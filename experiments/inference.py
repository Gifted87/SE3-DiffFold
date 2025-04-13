import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from torch_geometric.data import Data, Batch # Need Batch for sampling multiple

# Local imports
from models.score_network import ScoreNetwork # Assuming execution context allows direct import
from models.diffusion import SE3Diffusion
from data.preprocess import RESIDUE_MAP, N_RESIDUE_TYPES # For mapping indices back

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Reverse mapping from index to residue name (3-letter code)
IDX_TO_RES = {v: k for k, v in RESIDUE_MAP.items()}

def load_model_from_checkpoint(ckpt_path, config):
    """ Loads the ScoreNetwork model from a checkpoint file. """
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Re-create model architecture from config saved in checkpoint or provided separately
    # WHAT IF config in checkpoint differs from provided config? Prioritize checkpoint's?
    model_config = checkpoint.get('config', {}).get('model', config['model']) # Use checkpoint config if available

    # Ensure node_input_dim matches N_RESIDUE_TYPES used during training
    node_input_dim_train = model_config.get('node_input_dim', N_RESIDUE_TYPES) # Default if not in old ckpt

    model = ScoreNetwork(
        node_input_dim=node_input_dim_train, # Use the dimension from training
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        time_embed_dim=model_config['time_embed_dim'],
        radius=model_config['radius'],
        dropout=model_config.get('dropout', 0.0) # Handle missing dropout in old checkpoints
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode
    print(f"Loaded model state dict from epoch {checkpoint.get('epoch', 'N/A')}")
    return model

def load_diffusion_from_config(config):
     """ Instantiates SE3Diffusion from config """
     diff_config = config['diffusion']
     return SE3Diffusion(
         num_timesteps=diff_config['num_timesteps'],
         beta_start=diff_config['beta_start'],
         beta_end=diff_config['beta_end'],
         schedule_type=diff_config['schedule_type']
     )

def save_graph_to_pdb(coords, atom_types, output_pdb_file, chain_id="A"):
    """
    Saves generated coordinates and atom types as a PDB file (C-alpha trace).

    Args:
        coords (np.ndarray): Coordinates (N, 3).
        atom_types (np.ndarray): Integer atom/residue types (N,).
        output_pdb_file (str or Path): Path to save the PDB file.
        chain_id (str): Chain identifier.
    """
    if coords.shape[0] != atom_types.shape[0]:
        raise ValueError("Coordinates and atom types must have the same length.")

    # Create biotite AtomArray
    num_atoms = coords.shape[0]
    atom_array = struc.AtomArray(num_atoms)

    # Set coordinates
    atom_array.coord = coords

    # Set atom names (all C-alpha)
    atom_array.atom_name = ["CA"] * num_atoms

    # Set residue IDs (sequential 1 to N)
    atom_array.res_id = np.arange(1, num_atoms + 1)

    # Set residue names from atom_types indices
    # WHAT IF index is out of bounds? Use 'UNK'.
    atom_array.res_name = [IDX_TO_RES.get(idx, 'UNK') for idx in atom_types]

    # Set element type (Carbon for C-alpha)
    atom_array.element = ["C"] * num_atoms

    # Set chain ID
    atom_array.chain_id = [chain_id] * num_atoms

    # Set hetero property (False for standard amino acids)
    atom_array.hetero = np.zeros(num_atoms, dtype=bool)

    # Write PDB file
    # WHAT IF directory doesn't exist? Ensure parent dir exists.
    Path(output_pdb_file).parent.mkdir(parents=True, exist_ok=True)
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atom_array)
    try:
        pdb_file.write(str(output_pdb_file))
        print(f"Saved C-alpha trace PDB to: {output_pdb_file}")
    except Exception as e:
        print(f"Error writing PDB file {output_pdb_file}: {e}")


def generate_structure(args):
    """ Main function to generate structures """
    # Load configuration used during training (from checkpoint or specified)
    # For simplicity, assume config matching the checkpoint is provided via args.config
    if not args.config:
         # Try to load config from checkpoint if available
         try:
             checkpoint = torch.load(args.checkpoint, map_location='cpu') # Load to CPU first
             config = checkpoint['config']
             print("Loaded configuration from checkpoint.")
         except Exception as e:
             raise ValueError(f"Config file must be provided if not found in checkpoint: {e}")
    else:
         with open(args.config, 'r') as f:
             config = yaml.safe_load(f)

    # Load model
    score_net = load_model_from_checkpoint(args.checkpoint, config)

    # Load diffusion settings
    diffusion = load_diffusion_from_config(config)

    # Prepare input: Determine sequence length and atom types
    # Option 1: Fixed length
    # Option 2: Input sequence string -> convert to indices
    # Let's support both: specify length or sequence
    if args.sequence:
        # Convert sequence string to atom type indices
        try:
             atom_types = torch.tensor([RESIDUE_MAP[res.upper()] for res in args.sequence], dtype=torch.long)
             num_residues = len(args.sequence)
             print(f"Generating structure for sequence: {args.sequence} ({num_residues} residues)")
        except KeyError as e:
             raise ValueError(f"Invalid residue in sequence: {e}. Allowed: {list(RESIDUE_MAP.keys())}")
    elif args.length:
        num_residues = args.length
        # Generate random sequence or use poly-Alanine? Let's use poly-Alanine as default.
        ala_index = RESIDUE_MAP['ALA']
        atom_types = torch.full((num_residues,), ala_index, dtype=torch.long)
        print(f"Generating structure of length {num_residues} (Poly-Alanine)")
    else:
        raise ValueError("Either --sequence or --length must be provided.")

    # Define the shape of the coordinates tensor
    shape = (num_residues, 3)

    # Generate structures
    generated_coords_batch = diffusion.sample(
        score_net,
        shape=shape,
        num_nodes=num_residues,
        atom_types=atom_types, # Pass the determined atom types
        device=device,
        batch_size=args.num_samples
    ) # Output shape: (B, N, 3)

    # Save generated structures
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    atom_types_np = atom_types.cpu().numpy() # Convert atom types once for saving

    for i in range(args.num_samples):
        coords_np = generated_coords_batch[i].cpu().numpy() # (N, 3)
        # Determine filename
        base_name = Path(args.checkpoint).stem # Use checkpoint name as base
        if args.sequence:
             output_filename = output_dir / f"{args.sequence}_{i+1}.pdb"
        else:
             output_filename = output_dir / f"{base_name}_len{num_residues}_{i+1}.pdb"

        # Save as PDB
        save_graph_to_pdb(coords_np, atom_types_np, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate protein structures using SE(3) Diffusion Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--config", type=str, help="Path to the config YAML file used for training (optional if saved in ckpt).")
    parser.add_argument("--output_dir", type=str, default="generated_structures", help="Directory to save generated PDB files.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of structures to generate.")

    # Input specification group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", type=str, help="Amino acid sequence string (e.g., 'GAVLIMFWPSTCYNQDERKH').")
    group.add_argument("--length", type=int, help="Length of the protein sequence to generate (uses Poly-Alanine).")

    args = parser.parse_args()
    generate_structure(args)