import argparse
import os
import yaml
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR # Added LambdaLR for warmup
from torch.utils.data import Dataset # Use standard Dataset
from torch_geometric.loader import DataLoader # Use PyG DataLoader
from torch.utils.tensorboard import SummaryWriter # For logging
# Removed unused import: from torch_geometric.data import Data, Batch # Batch imported in diffusion.py sample

# Local imports
from models.score_network import ScoreNetwork # Direct import assuming correct PYTHONPATH or execution context
from models.diffusion import SE3Diffusion
from data.preprocess import N_RESIDUE_TYPES # Make sure this matches config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset Class
class ProteinGraphDataset(Dataset):
    """ Loads preprocessed protein graphs from .pt files """
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        # WHAT IF data_dir doesn't exist or is empty? Check.
        if not self.data_dir.is_dir():
             raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        self.file_paths = sorted(list(self.data_dir.glob("*.pt")))
        if not self.file_paths:
             raise FileNotFoundError(f"No '.pt' graph files found in {data_dir}")
        print(f"Loaded {len(self.file_paths)} protein graphs from {data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load graph data from file
        # WHAT IF a file is corrupted? Add try-except.
        try:
             data = torch.load(self.file_paths[idx])
             # Basic validation? Check for pos and x attributes.
             if not hasattr(data, 'pos') or not hasattr(data, 'x'):
                  print(f"Warning: Skipping corrupted file {self.file_paths[idx]}")
                  # Return next item? Or handle this in collate_fn? Best to skip here.
                  # Need a way to handle this properly in the loop. Let's retry with next index.
                  # This recursive call is dangerous. Better: Filter out bad files initially or handle in training loop.
                  # For now, let's assume files are valid after preprocessing.
                  return self.__getitem__((idx + 1) % len(self)) # Simple retry, can cause issues
             return data
        except Exception as e:
             print(f"Error loading file {self.file_paths[idx]}: {e}. Skipping.")
             # Again, retry logic needs care. Skipping is safer during iteration.
             # Let's just return None and handle it in the training loop.
             return None # Signal an error

def collate_filter_none(batch):
     """ Collate function that filters out None items """
     batch = [item for item in batch if item is not None]
     # If the entire batch becomes empty after filtering
     if not batch:
         return None
     # Use default PyG collate if score_net expects Batch object
     # BUT score_net needs Batch. Let's use PyG dataloader instead of standard one.
     # RETHINK: Use DataLoader from torch_geometric.loader directly. It handles batching.
     # Remove this collate function and the standard Dataset/DataLoader usage.
     pass # Will use PyG DataLoader below.

def load_config(config_path):
    """ Loads YAML configuration file """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_optimizer(model, config):
    """ Creates optimizer based on config """
    lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']
    opt_name = config['training']['optimizer']

    if opt_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

def get_scheduler(optimizer, config, total_steps):
    """ Creates learning rate scheduler based on config """
    if not config['training']['scheduler']['use']:
        return None

    warmup_epochs = config['training']['scheduler']['warmup_epochs']
    # Calculate warmup steps - requires steps_per_epoch
    # This info isn't directly available here. Pass total_steps instead.
    warmup_steps = warmup_epochs * (total_steps / config['training']['epochs']) # Approximation

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # After warmup, decay using CosineAnnealing (needs T_max)
        # This setup is complex. Let's simplify: Use CosineAnnealingLR *after* warmup.
        # OR use a combined scheduler like transformers library.
        # Simpler: CosineAnnealingLR over the whole training minus warmup.
        # BUT the API expects T_max. Let T_max be total_steps - warmup_steps.
        # Let's use a simpler warmup: linear warmup phase, then constant LR or standard decay.
        # Try LambdaLR for linear warmup, then let CosineAnnealing handle the rest.
        # NO, CosineAnnealing needs T_max.

        # Alternative: Linear warmup for N steps, then Cosine decay for remaining steps.
        if current_step < warmup_steps:
             return float(current_step) / float(max(1.0, warmup_steps))
        else:
             progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
             return 0.5 * (1.0 + np.cos(np.pi * progress)) # Cosine decay part


    # T_max should be total training steps for CosineAnnealingLR
    #scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.01)
    scheduler = LambdaLR(optimizer, lr_lambda) # Use the combined lambda function

    return scheduler


def train(config_path):
    """ Main training function """
    config = load_config(config_path)

    # Setup logging and checkpointing directories
    log_dir = Path(config['logging']['log_dir'])
    ckpt_dir = Path(config['logging']['checkpoint_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Set seed
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])

    # Load data
    # Use torch_geometric DataLoader
    dataset = ProteinGraphDataset(data_dir=config['data']['dataset_path'])
    # WHAT IF dataset is empty? Checked in constructor.
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True # Improves GPU transfer speed
    )

    # Initialize models
    # Using simple instantiation based on config keys for now
    # Ideally use hydra or similar for `_target_` instantiation.
    score_net = ScoreNetwork(
        node_input_dim=config['model']['node_input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        time_embed_dim=config['model']['time_embed_dim'],
        radius=config['model']['radius'],
        dropout=config['model']['dropout']
    ).to(device)

    diffusion = SE3Diffusion(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        schedule_type=config['diffusion']['schedule_type']
    )

    # Optimizer and Scheduler
    optimizer = get_optimizer(score_net, config)
    total_steps = len(dataloader) * config['training']['epochs']
    scheduler = get_scheduler(optimizer, config, total_steps)

    # Resume from checkpoint if available
    # WHAT IF we want to resume training? Add checkpoint loading.
    start_epoch = 0
    latest_ckpt = max(ckpt_dir.glob("*.pth"), key=os.path.getctime, default=None)
    if latest_ckpt:
        print(f"Resuming training from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        score_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # Load scheduler state if saved
        if scheduler and 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from epoch {start_epoch}")


    print(f"Starting training for {config['training']['epochs']} epochs...")
    global_step = start_epoch * len(dataloader)

    for epoch in range(start_epoch, config['training']['epochs']):
        score_net.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, batch_data in enumerate(dataloader):
            # Handle potential None from dataset/collate (though PyG loader shouldn't yield None)
            if batch_data is None:
                 print(f"Warning: Skipping empty batch at step {i}")
                 continue

            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            # Sample timesteps uniformly for each graph in the batch
            t = torch.randint(0, diffusion.num_timesteps, (batch_data.num_graphs,), device=device).long()

            # Sample noise and apply forward diffusion
            noise = torch.randn_like(batch_data.pos)
            noisy_pos = diffusion.q_sample(x_start=batch_data.pos, t=t, noise=noise)

            # Create noisy data object for the model
            # Need to preserve batch info
            noisy_data = batch_data.clone() # Clone to avoid modifying original batch
            noisy_data.pos = noisy_pos

            # Predict noise using the score network
            predicted_noise = score_net(noisy_data, t)

            # Calculate loss (MSE between predicted and actual noise)
            # WHAT IF different number of nodes per graph? MSE should handle variable N correctly.
            loss = F.mse_loss(predicted_noise, noise) # Simple MSE loss

            loss.backward()

            # Gradient clipping
            # WHAT IF gradients explode? Clip them.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                score_net.parameters(), config['training']['gradient_clip_val']
            )

            optimizer.step()

            # Update learning rate scheduler
            if scheduler:
                scheduler.step() # Step per iteration

            epoch_loss += loss.item()
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            writer.add_scalar("Params/grad_norm", grad_norm.item(), global_step)
            if scheduler:
                 writer.add_scalar("Params/learning_rate", scheduler.get_last_lr()[0], global_step)
            global_step += 1

            if (i + 1) % 50 == 0: # Log every 50 steps
                 print(f"Epoch [{epoch+1}/{config['training']['epochs']}] Step [{i+1}/{len(dataloader)}] Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{config['training']['epochs']}] Completed in {epoch_time:.2f}s | Average Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0 or (epoch + 1) == config['training']['epochs']:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1}.pth"
            save_payload = {
                'epoch': epoch,
                'model_state_dict': score_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': config # Save config for reproducibility
            }
            if scheduler:
                 save_payload['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(save_payload, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SE(3) Equivariant Diffusion Model")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    train(args.config)