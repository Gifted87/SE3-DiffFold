import unittest
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Data, Batch

# Assuming models are importable
from models.score_network import ScoreNetwork
from data.preprocess import N_RESIDUE_TYPES

# Use CPU for testing consistency
device = torch.device("cpu")

class TestEquivariance(unittest.TestCase):

    def setUp(self):
        """ Set up a dummy model and input data """
        self.hidden_dim = 16 # Smaller hidden dim for faster testing
        self.num_layers = 2
        self.time_embed_dim = 8
        self.radius = 5.0
        self.num_nodes = 10
        self.batch_size = 2 # Test with batching

        # Dummy model
        self.model = ScoreNetwork(
            node_input_dim=N_RESIDUE_TYPES,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            time_embed_dim=self.time_embed_dim,
            radius=self.radius
        ).to(device)
        self.model.eval() # Set to eval mode

        # Dummy input data (batch of two graphs)
        self.atom_types = torch.randint(0, N_RESIDUE_TYPES, (self.batch_size * self.num_nodes,), device=device)
        self.pos = torch.randn(self.batch_size * self.num_nodes, 3, device=device) * 5.0 # Scale positions
        self.batch_index = torch.arange(self.batch_size, device=device).repeat_interleave(self.num_nodes)
        self.t = torch.randint(0, 1000, (self.batch_size,), device=device).long() # Dummy timesteps per graph

        self.data = Batch(x=self.atom_types, pos=self.pos, batch=self.batch_index)

        # Generate random SE(3) transformation (per batch item is harder, apply same to all for now)
        # Or, apply per item and check relative outputs within batch. Let's apply same first.
        self.rotation_matrix = torch.tensor(R.random().as_matrix(), dtype=torch.float32, device=device)
        self.translation_vector = torch.randn(3, device=device) * 2.0 # Random translation

    def transform_pos(self, pos, rot, trans):
        """ Applies rotation and translation to coordinates """
        # pos shape: (N_total, 3)
        # rot shape: (3, 3)
        # trans shape: (3,)
        # Need to handle batches if applying different transforms per graph.
        # For now, apply same transform:
        pos_rotated = torch.matmul(pos, rot.T) # Matmul expects (..., N, D), Rot (D, D) -> use transpose
        pos_transformed = pos_rotated + trans.unsqueeze(0) # Add translation broadcasted
        return pos_transformed

    def test_se3_equivariance(self):
        """ Test if the ScoreNetwork output transforms correctly under SE(3) """

        # Apply transformation to input positions
        pos_transformed = self.transform_pos(self.pos, self.rotation_matrix, self.translation_vector)
        data_transformed = Batch(x=self.atom_types, pos=pos_transformed, batch=self.batch_index)

        # Pass original and transformed data through the model
        with torch.no_grad():
            score_original = self.model(self.data, self.t)
            score_transformed = self.model(data_transformed, self.t)

        # The predicted score (noise vector) should rotate with the input coordinates.
        # Translation should not affect the score vector itself (it's a difference vector).
        # Rotate the original score vector using the same rotation matrix.
        score_original_rotated = torch.matmul(score_original, self.rotation_matrix.T)

        # Check if the rotated original score matches the score from the transformed input
        # Use torch.allclose for numerical tolerance
        # WHAT IF the tolerance is too strict/loose? Start with default atol/rtol.
        self.assertTrue(
            torch.allclose(score_transformed, score_original_rotated, atol=1e-5),
            f"Equivariance test failed.\n"
            f"Max difference: {(score_transformed - score_original_rotated).abs().max().item()}\n"
            # f"Score Transformed:\n{score_transformed[:5]}\n" # Optional: Print parts for debugging
            # f"Score Original Rotated:\n{score_original_rotated[:5]}"
        )
        print("SE(3) Equivariance test passed.")

if __name__ == '__main__':
    unittest.main()