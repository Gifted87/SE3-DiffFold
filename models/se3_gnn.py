import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph

class SE3GNNLayer(MessagePassing):
    """
    Single layer of the SE(3) Equivariant Graph Neural Network.
    Inspired by E(n)-GNNs and adapted based on the provided snippet structure.
    Updates both node features (invariant) and node positions (equivariant).
    """
    def __init__(self, node_dim, edge_hidden_dim, node_out_dim, pos_out_scalar_dim, aggr="mean", dropout=0.1):
        # node_dim: Dimension of input node features (h)
        # edge_hidden_dim: Hidden dimension for processing edge information
        # node_out_dim: Dimension of output node features (h')
        # pos_out_scalar_dim: Dimension of scalar multipliers for position updates
        # Using "mean" aggregation for messages by default. Sum is also common.
        super().__init__(aggr=aggr)

        self.node_dim = node_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.node_out_dim = node_out_dim
        self.pos_out_scalar_dim = pos_out_scalar_dim

        # MLP to process edge features (distance)
        # Input: 1 (distance norm)
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_hidden_dim),
            nn.SiLU(), # Swish activation is common
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU()
        )

        # MLP to generate messages
        # Input: node_dim (h_i) + node_dim (h_j) + edge_hidden_dim (e_ij)
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden_dim, edge_hidden_dim), # Output: message vector m_ij
            nn.SiLU()
        )

        # MLP to update node features
        # Input: node_dim (h_i) + aggregated message dim (edge_hidden_dim)
        self.node_update_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_hidden_dim, node_out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_out_dim, node_out_dim)
        )

        # MLP to compute scalar multiplier for coordinate updates (equivariant part)
        # Input: message vector m_ij (edge_hidden_dim)
        # Output: pos_out_scalar_dim scalars to scale relative position vector
        self.coord_update_mlp = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, pos_out_scalar_dim) # Predict scalar multiplier(s)
        )

        # Layer normalization for stability
        self.norm_h = nn.LayerNorm(node_out_dim)
        # WHAT IF pos updates explode? Maybe add LayerNorm or clipping to coord_update_mlp output? Let's skip for now.

    def forward(self, h, pos, edge_index):
        """
        Performs one layer of message passing.

        Args:
            h (Tensor): Node features (N, node_dim).
            pos (Tensor): Node coordinates (N, 3).
            edge_index (LongTensor): Edge index (2, E).

        Returns:
            Tuple[Tensor, Tensor]:
                - h_new (Tensor): Updated node features (N, node_out_dim).
                - pos_update (Tensor): Aggregated equivariant position updates (N, 3 * pos_out_scalar_dim).
                  Needs reshaping/summing downstream.
        """
        # Propagate messages: This calls message(), aggregate(), and update()
        # Pass necessary tensors to message function via propagate args
        aggregated_messages, pos_updates_agg = self.propagate(
            edge_index, h=h, pos=pos, size=None # size=None assumes square adjacency matrix logic
        )

        # Update node features (invariant update)
        h_new = self.node_update_mlp(torch.cat([h, aggregated_messages], dim=-1))
        h_new = self.norm_h(h + h_new) # Residual connection + Norm

        # Return updated features and the aggregated position updates
        return h_new, pos_updates_agg

    def message(self, h_i, h_j, pos_i, pos_j):
        """
        Computes messages between nodes i and j.

        Args:
            h_i (Tensor): Features of target nodes (E, node_dim).
            h_j (Tensor): Features of source nodes (E, node_dim).
            pos_i (Tensor): Coordinates of target nodes (E, 3).
            pos_j (Tensor): Coordinates of source nodes (E, 3).

        Returns:
            Tuple[Tensor, Tensor]:
                - message (Tensor): Message for node feature update (E, edge_hidden_dim).
                - pos_update (Tensor): Equivariant position update component (E, 3 * pos_out_scalar_dim).
        """
        # Calculate relative position and distance (invariant)
        rel_pos = pos_j - pos_i  # Vector pointing from i to j
        dist = rel_pos.norm(dim=-1, keepdim=True)

        # Avoid division by zero if dist is 0 (overlapping nodes)
        # WHAT OF EDGE CASES like overlapping nodes? Add small epsilon.
        dist = torch.clamp(dist, min=1e-6)

        # Compute edge embedding based on distance
        edge_attr = self.edge_mlp(dist) # (E, edge_hidden_dim)

        # Compute message using node features and edge embedding
        message_input = torch.cat([h_i, h_j, edge_attr], dim=-1)
        message = self.message_mlp(message_input) # (E, edge_hidden_dim)

        # Compute scalar multiplier(s) for coordinate update
        coord_scalar = self.coord_update_mlp(message) # (E, pos_out_scalar_dim)

        # Compute the equivariant position update component for this edge
        # Scale the normalized relative position vector
        # Using rel_pos / dist ensures the direction is preserved but magnitude is controlled by coord_scalar
        # We output multiple potential updates per edge if pos_out_scalar_dim > 1
        # Reshape coord_scalar to allow broadcasting: (E, pos_out_scalar_dim, 1)
        # Reshape rel_pos/dist: (E, 1, 3)
        pos_update = (rel_pos / dist).unsqueeze(1) * coord_scalar.unsqueeze(-1) # (E, pos_out_scalar_dim, 3)
        pos_update = pos_update.view(pos_update.size(0), -1) # Flatten to (E, pos_out_scalar_dim * 3)

        return message, pos_update

    def aggregate(self, inputs, index, dim_size=None):
        """
        Aggregates messages. Overrides default to handle multiple outputs from message().

        Args:
            inputs (Tuple[Tensor, Tensor]): Tuple containing:
                - message (Tensor): Messages for node features (E, edge_hidden_dim).
                - pos_update (Tensor): Position update components (E, pos_out_scalar_dim * 3).
            index (LongTensor): Index tensor mapping edges to target nodes.
            dim_size (int, optional): Number of nodes.

        Returns:
            Tuple[Tensor, Tensor]: Aggregated messages and position updates.
        """
        message_agg = super().aggregate(inputs[0], index, dim=self.node_dim, dim_size=dim_size)
        pos_update_agg = super().aggregate(inputs[1], index, dim=self.node_dim, dim_size=dim_size)
        return message_agg, pos_update_agg

    def update(self, inputs):
        """
        Update function - we do the update in forward after aggregation,
        so this just returns the aggregated values.
        """
        return inputs