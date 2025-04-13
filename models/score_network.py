# models/score_network.py
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import radius_graph
from .se3_gnn import SE3GNNLayer # Assuming se3_gnn.py is in the same directory
from data.preprocess import N_RESIDUE_TYPES # Get number of residue types

class SinusoidalTimeEmbedding(nn.Module):
    """ Sinusoidal time embedding """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: Tensor of shape (batch_size,) or scalar
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:  # zero pad if dim is odd
            emb = nn.functional.pad(emb, (0, 1))
        return emb

class ScoreNetwork(nn.Module):
    """
    SE(3)-Equivariant Score Network for Protein Structure Diffusion.
    Uses stacked SE3GNNLayers.
    Predicts the noise added at a given timestep.
    """
    def __init__(self, node_input_dim=N_RESIDUE_TYPES, hidden_dim=128, num_layers=4,
                 time_embed_dim=32, radius=10.0, dropout=0.1):
        super().__init__()

        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.radius = radius # Interaction radius for graph construction

        # Embedding for residue types (initial node features)
        self.residue_embedding = nn.Embedding(node_input_dim, hidden_dim)

        # Embedding for timestep t
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initial MLP to combine residue and time embeddings
        self.initial_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Combine residue and time embeds
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Stack of SE(3) GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            # Assuming pos_out_scalar_dim = 1, meaning each layer contributes one vector update
            self.gnn_layers.append(
                SE3GNNLayer(
                    node_dim=hidden_dim,
                    edge_hidden_dim=hidden_dim,
                    node_out_dim=hidden_dim,
                    pos_out_scalar_dim=1, # Predict one scalar per edge message for pos update
                    dropout=dropout
                )
            )

        # Final MLPs
        # Predict final scalar for position update magnitude from node features
        self.final_node_mlp = nn.Sequential(
             nn.Linear(hidden_dim, hidden_dim),
             nn.SiLU(),
             nn.Linear(hidden_dim, 1) # Output one scalar per node
        )

        # Layer norm for the final position update
        self.norm_pos_update = nn.LayerNorm(3) # Normalize the final 3D vector update

    def forward(self, data, t):
        """
        Forward pass of the score network.

        Args:
            data (torch_geometric.data.Data or Batch): Input graph data.
                Requires data.x (atom_types), data.pos (coordinates).
                data.batch is needed if using batches.
            t (Tensor): Timestep tensor, shape (num_graphs,) if batched, or scalar.

        Returns:
            Tensor: Predicted noise (score) vector for each node, shape (N, 3).
        """
        x, pos, batch = data.x, data.pos, data.batch

        # 1. Build graph dynamically based on current positions
        # WHAT IF dealing with batches? radius_graph needs batch argument.
        edge_index = radius_graph(pos, r=self.radius, batch=batch, loop=False) # No self-loops needed here

        # 2. Embed residues and timestep
        h = self.residue_embedding(x) # (N, hidden_dim)

        # Handle scalar t or batched t
        if t.numel() == 1:
            time_emb = self.time_embedding(t.expand(batch.max().item() + 1 if batch is not None else 1)) # (num_graphs, time_embed_dim)
        else:
            time_emb = self.time_embedding(t) # (num_graphs, time_embed_dim)

        time_emb = self.time_mlp(time_emb) # (num_graphs, hidden_dim)

        # Expand time_emb to match node batch dimension
        # WHAT IS HAPPENING NOW? Need to map graph-level time embedding to node-level.
        if batch is None: # Single graph case
             node_time_emb = time_emb.repeat(h.size(0), 1)
        else: # Batched case
             node_time_emb = time_emb[batch] # (N, hidden_dim)

        # Combine embeddings
        h = self.initial_mlp(torch.cat([h, node_time_emb], dim=-1)) # (N, hidden_dim)

        # Initialize aggregated position update
        total_pos_update = torch.zeros_like(pos) # (N, 3)

        # 3. Pass through SE(3) GNN layers
        for layer in self.gnn_layers:
             # Layer returns updated h and aggregated pos updates for this layer
             h_new, pos_update_agg = layer(h, pos, edge_index)
             # pos_update_agg has shape (N, 3 * pos_out_scalar_dim) = (N, 3) if dim=1

             # WAIT A MINUTE, the SE3GNNLayer returns the *aggregated* contribution.
             # How should these updates be combined across layers? Sum them up?
             # Let's sum the position updates from each layer.
             # Also, use the updated node features h_new for the next layer.
             # The position 'pos' itself is NOT updated within the loop, it remains the input noisy pos.
             # This is crucial for score matching - the network conditions on the *fixed* noisy input pos_t.
             h = h_new
             total_pos_update = total_pos_update + pos_update_agg.view(pos.shape[0], 3) # Reshape and add

        # 4. Final prediction
        # Use final node features `h` to predict a final scalar multiplier for the aggregated pos update.
        # This allows node features to gate the geometric update.
        final_scalar = self.final_node_mlp(h) # (N, 1)

        # Revised Final Layer:
        self.final_output_mlp = nn.Sequential(
             nn.Linear(hidden_dim, hidden_dim),
             nn.SiLU(),
             nn.Linear(hidden_dim, 3) # Output 3D score vector per node
        )
        # Remove self.final_node_mlp and self.norm_pos_update from __init__ and add self.final_output_mlp

        # Revised forward pass ending:
        # After GNN layers loop, `h` contains the final node features.
        predicted_score = self.final_output_mlp(h) # (N, 3)

        predicted_score = self.norm_pos_update(total_pos_update) # (N, 3)

        return predicted_score