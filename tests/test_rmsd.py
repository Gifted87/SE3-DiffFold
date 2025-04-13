import unittest
import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader # Using PyG loader
from tqdm import tqdm
import biotite.structure as struc

# Local imports
from experiments.train import ProteinGraphDataset # Reuse dataset class
from experiments.inference import load_model_from_checkpoint, load_diffusion_from_config, save_graph_to_pdb
from data.preprocess import RESIDUE_MAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_rmsd(coords_pred: np.ndarray, coords_true: np.ndarray) -> float:
    """
    Calculates C-alpha RMSD between predicted and true coordinates after alignment.

    Args:
        coords_pred (np.ndarray): Predicted coordinates (N, 3).
        coords_true (np.ndarray): Ground truth coordinates (N, 3).

    Returns:
        float: C-alpha RMSD value.
    """
    if coords_pred.shape != coords_true.shape:
        raise ValueError("Predicted and true coordinate shapes must match.")
    if coords_pred.shape[0] == 0:
        return 0.0 # Or raise error? RMSD is 0 for empty structure.

    # Use biotite's superposition function for alignment
    try:
        # Align predicted onto true
        coords_pred_aligned, transform = struc.superimpose(coords_true, coords_pred)
        # Calculate RMSD
        diff = coords_pred_aligned - coords_true
        rmsd = np.sqrt(np.sum(diff * diff) / coords_pred.shape[0])
        return float(rmsd)
    except Exception as e:
        print(f"Warning: Superposition failed: {e}. Returning NaN.")
        return np.nan


def evaluate_rmsd(args):
    """ Evaluates RMSD on a test dataset """
    # Load config (either from checkpoint or specified)
    if not args.config:
         try:
             checkpoint = torch.load(args.checkpoint, map_location='cpu')
             config = checkpoint['config']
             print("Loaded configuration from checkpoint.")
         except Exception as e:
             raise ValueError(f"Config file must be provided if not found in checkpoint: {e}")
    else:
         with open(args.config, 'r') as f:
             config = yaml.safe_load(f)

    # Load Model and Diffusion
    score_net = load_model_from_checkpoint(args.checkpoint, config)
    diffusion = load_diffusion_from_config(config)

    # Load Test Dataset
    # WHAT IF test dataset path is different from training? Use args.test_data_dir.
    test_dataset = ProteinGraphDataset(data_dir=args.test_data_dir)
    # Use DataLoader only for iterating if needed, but sampling is per-item.
    # We'll iterate through the dataset directly.

    results = {"pdb_id": [], "rmsd": [], "num_residues": []}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating RMSD on {len(test_dataset)} structures from {args.test_data_dir}...")

    for i in tqdm(range(len(test_dataset)), desc="Evaluating RMSD"):
        try:
            ground_truth_data = test_dataset[i]
            if ground_truth_data is None: continue # Skip if loading failed

            pdb_id = ground_truth_data.name if hasattr(ground_truth_data, 'name') else f"test_{i}"
            num_residues = ground_truth_data.num_nodes
            atom_types = ground_truth_data.x # Shape (N,)
            shape = (num_residues, 3)

            # Generate structure using the model
            # Sample only one structure per test case for evaluation
            generated_coords = diffusion.sample(
                score_net,
                shape=shape,
                num_nodes=num_residues,
                atom_types=atom_types, # Use ground truth atom types
                device=device,
                batch_size=1 # Sample one at a time
            )[0] # Get the single sample from the batch (N, 3)

            # Get ground truth coordinates
            coords_true_np = ground_truth_data.pos.cpu().numpy()
            coords_pred_np = generated_coords.cpu().numpy()

            # Calculate RMSD
            rmsd = calculate_rmsd(coords_pred_np, coords_true_np)

            if not np.isnan(rmsd):
                 results["pdb_id"].append(pdb_id)
                 results["rmsd"].append(rmsd)
                 results["num_residues"].append(num_residues)

                 # Optionally save the generated structure for inspection
                 if args.save_pdbs:
                      output_pdb_path = output_dir / f"{pdb_id}_generated.pdb"
                      atom_types_np = atom_types.cpu().numpy()
                      save_graph_to_pdb(coords_pred_np, atom_types_np, output_pdb_path)
            else:
                 print(f"Skipping RMSD calculation for {pdb_id} due to alignment error.")

        except Exception as e:
            pdb_id_err = f"test_{i}"
            if 'ground_truth_data' in locals() and hasattr(ground_truth_data, 'name'):
                pdb_id_err = ground_truth_data.name
            print(f"Error processing structure {pdb_id_err}: {e}")
            continue # Skip to next structure

    # Calculate and print average RMSD
    if results["rmsd"]:
        average_rmsd = np.mean(results["rmsd"])
        median_rmsd = np.median(results["rmsd"])
        print("\n--- Evaluation Summary ---")
        print(f"Evaluated on: {len(results['rmsd'])} structures")
        print(f"Average C-alpha RMSD: {average_rmsd:.4f} Å")
        print(f"Median C-alpha RMSD:  {median_rmsd:.4f} Å")
        print("------------------------")

        # Save detailed results
        import pandas as pd
        df = pd.DataFrame(results)
        results_file = output_dir / "rmsd_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Detailed results saved to: {results_file}")

    else:
        print("No RMSD results were calculated.")

# Using unittest structure for potential integration, but running as script for now
class TestRMSDPerformance(unittest.TestCase):
     # This test would require command line args or hardcoded paths
     # @unittest.skip("Requires trained model and test data - run as script")
     def test_placeholder(self):
          # To make `unittest.main()` runnable without errors if no CLI args provided
          print("RMSD evaluation needs to be run as a script with arguments.")
          self.assertTrue(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate model RMSD on a test set.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--config", type=str, help="Path to the config YAML file (optional if saved in ckpt).")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Directory containing preprocessed test graph (.pt) files.")
    parser.add_argument("--output_dir", type=str, default="rmsd_evaluation", help="Directory to save RMSD results and optional PDBs.")
    parser.add_argument("--save_pdbs", action='store_true', help="Save generated PDB files for inspection.")
    # Add batch size if sampling becomes slow? Sampling is already per structure.

    args = parser.parse_args()
    evaluate_rmsd(args)
    # unittest.main() # Keep this commented out unless structuring as proper tests