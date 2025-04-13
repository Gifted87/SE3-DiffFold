import torch
import torch.nn.functional as F
import numpy as np

# Using torch_geometric Data object for consistency
from torch_geometric.data import Data

class SE3Diffusion:
    """
    Implements the variance schedule and forward diffusion process
    (adding noise to coordinates) for SE(3) coordinates.

    NOTE: This implements standard Euclidean coordinate diffusion (DDPM-style),
    often used as an approximation for manifold diffusion in practice.
    True SE(3) diffusion might involve noise on Lie Algebra.
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type="linear"):
        """
        Initializes the variance schedule.

        Args:
            num_timesteps (int): Number of diffusion steps.
            beta_start (float): Starting beta value.
            beta_end (float): Ending beta value.
            schedule_type (str): Type of beta schedule ('linear', 'cosine').
        """
        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            s = 0.008 # Offset s
            steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
            alpha_bar = torch.cos(((steps + s) / (1 + s)) * np.pi / 2) ** 2
            beta_t = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
            self.betas = torch.clamp(beta_t, min=0, max=0.999) # Clip betas
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        self.alphas = 1.0 - self.betas
        # alpha_bar or alphas_cumprod is the cumulative product of alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_bar_{t-1}

        # Precompute values needed for forward process q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Precompute values needed for reverse process q(x_{t-1} | x_t, x_0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # Clip variance to avoid instability when beta approaches 0
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)


    def _extract(self, a, t, x_shape):
        """ Utility function to extract coefficients for a batch of timesteps """
        batch_size = t.shape[0]
        out = a.to(t.device)[t].float() # Get coefficients for specific timesteps t
        # Reshape to (batch_size, 1, 1, ...) for broadcasting
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0).
        Adds noise to the initial data x_start to get noisy version x_t at timestep t.

        Args:
            x_start (Tensor): Initial protein coordinates (N, 3) or (B, N, 3).
            t (LongTensor): Timesteps for each sample in the batch (B,).
            noise (Tensor, optional): Pre-sampled noise, same shape as x_start. Defaults to standard Gaussian.

        Returns:
            Tensor: Noisy coordinates x_t at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get coefficients for the batch of timesteps t
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        # Apply diffusion formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_pos = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return noisy_pos

    # --- Methods for Inference (Reverse Process) ---

    def p_mean_variance(self, score_net, x_t_data, t, clip_denoised=True):
        """
        Calculate the mean and variance of the reverse diffusion step p(x_{t-1} | x_t).

        Args:
            score_net (nn.Module): The trained score network model.
            x_t_data (Data or Batch): The noisy graph data at timestep t (contains x_t coordinates).
            t (LongTensor): Current timestep (B,).
            clip_denoised (bool): Whether to clamp the predicted x_0 to [-1, 1] (if data was normalized).
                                  Less relevant for coordinates, maybe clip based on typical protein bounds?
                                  Let's skip coordinate clipping for now.

        Returns:
            dict: Containing 'mean', 'variance', 'log_variance', 'pred_xstart'.
        """
        x_t = x_t_data.pos
        # Predict the noise (epsilon) using the score network
        pred_epsilon = score_net(x_t_data, t) # Model predicts noise

        # Get coefficients for timestep t
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)

        # Calculate the mean of p(x_{t-1} | x_t) using the predicted noise
        # mean = sqrt(1/alpha_t) * (x_t - beta_t / sqrt(1 - alpha_bar_t) * epsilon_pred)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_epsilon / sqrt_one_minus_alphas_cumprod_t)

        # Calculate predicted x_0 (optional, useful for analysis or clipping)
        pred_xstart = self._predict_xstart_from_epsilon(x_t, t, pred_epsilon)
        # if clip_denoised:
             # Add clipping logic here if needed based on coordinate ranges

        # Get posterior variance and log variance for timestep t
        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_t = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return {
            "mean": model_mean,
            "variance": posterior_variance_t,
            "log_variance": posterior_log_variance_t,
            "pred_xstart": pred_xstart,
            "pred_epsilon": pred_epsilon
        }

    def _predict_xstart_from_epsilon(self, x_t, t, epsilon):
        """ Computes predicted x_0 given x_t and predicted epsilon """
        sqrt_recip_alphas_cumprod_t = self._extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(torch.sqrt(1.0 / self.alphas_cumprod - 1), t, x_t.shape)
        # x_0_pred = sqrt(1/alpha_bar_t) * x_t - sqrt(1/alpha_bar_t - 1) * epsilon
        pred_xstart = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * epsilon
        return pred_xstart


    @torch.no_grad()
    def p_sample(self, score_net, x_t_data, t):
        """
        Sample x_{t-1} from the model distribution p(x_{t-1} | x_t).

        Args:
            score_net (nn.Module): Trained score network.
            x_t_data (Data or Batch): Noisy graph data at timestep t.
            t (LongTensor): Current timestep (B,).

        Returns:
            Tensor: Sampled coordinates x_{t-1}.
        """
        x_t = x_t_data.pos
        out = self.p_mean_variance(score_net, x_t_data, t)
        model_mean = out["mean"]
        model_log_variance = out["log_variance"]

        noise = torch.randn_like(x_t)
        # No noise is added at t=0
        # WHAT IF t=0? Check condition.
        nonzero_mask = (t != 0).float().view(-1, *((1,) * (len(x_t.shape) - 1)))

        # Sample: x_{t-1} = mean + sqrt(variance) * noise
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample

    @torch.no_grad()
    def sample(self, score_net, shape, num_nodes, atom_types, device, batch_size=1):
        """
        Generate samples (protein coordinates) starting from noise.

        Args:
            score_net (nn.Module): Trained score network.
            shape (tuple): Shape of the output coordinates tensor per sample (num_nodes, 3).
            num_nodes (int): Number of nodes/residues in the protein.
            atom_types (Tensor): Tensor of atom/residue types (num_nodes,).
            device: Torch device.
            batch_size (int): Number of samples to generate in parallel.

        Returns:
            Tensor: Generated protein coordinates (batch_size, num_nodes, 3).
        """
        print(f"Sampling {batch_size} structures with {num_nodes} residues...")
        # Start from pure noise at timestep T (self.num_timesteps - 1)
        img_shape = (batch_size,) + shape # (B, N, 3)
        coords = torch.randn(img_shape, device=device)


        from torch_geometric.data import Batch
        data_list = [Data(x=atom_types.to(device), pos=coords[i]) for i in range(batch_size)]
        current_data = Batch.from_data_list(data_list)

        # Reverse diffusion loop
        for t_int in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            t = torch.full((batch_size,), t_int, device=device, dtype=torch.long)
            # Sample x_{t-1}
            coords = self.p_sample(score_net, current_data, t)
            # Update coordinates in the Data object for the next step
            current_data.pos = coords

        print("Sampling complete.")
        # Final result is coords at t=0
        return coords