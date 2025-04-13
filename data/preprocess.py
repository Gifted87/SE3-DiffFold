import os
import argparse
import warnings
from pathlib import Path
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import torch
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

# Define mapping for common amino acids to integers (can be expanded)
# Using a simple index for now. Could use properties later.
RESIDUE_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,
    'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13,
    'PRO': 14, 'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    # Add 'UNK' or handle others if needed
    'UNK': 20
}
N_RESIDUE_TYPES = len(RESIDUE_MAP)

def get_alpha_carbon_coordinates(pdb_file: Path) -> tuple[np.ndarray | None, list[int] | None]:
    """
    Extracts C-alpha coordinates and residue type indices from a PDB file.

    Args:
        pdb_file (Path): Path to the PDB file.

    Returns:
        tuple[np.ndarray | None, list[int] | None]:
            - Numpy array of C-alpha coordinates (N, 3) or None if error.
            - List of residue type integers (N,) or None if error.
    """
    try:
        # Use biotite for robust PDB parsing
        pdb_obj = pdb.PDBFile.read(str(pdb_file))
        
        # Get the first model (handling multi-model files)
        # BUT what if the PDB has no models? Check structure length.
        if pdb_obj.get_model_count() == 0:
            warnings.warn(f"No models found in PDB: {pdb_file.name}")
            return None, None
            
        struct = pdb_obj.get_structure(model=1)
        
        # Select C-alpha atoms
        ca_mask = (struct.atom_name == "CA") & struc.filter_amino_acids(struct)
        ca_atoms = struct[ca_mask]

        # WHAT IF no C-alpha atoms are found (e.g., DNA/RNA file, or non-protein)?
        if len(ca_atoms) == 0:
            warnings.warn(f"No C-alpha atoms found in PDB: {pdb_file.name}")
            return None, None

        coords = ca_atoms.coord
        res_names = ca_atoms.res_name

        # Map residue names to integers
        # BUT what if a residue name isn't in our map? Use 'UNK'.
        res_indices = [RESIDUE_MAP.get(name, RESIDUE_MAP['UNK']) for name in res_names]

        # Check for consistency
        if coords.shape[0] != len(res_indices):
             warnings.warn(f"Coordinate and residue count mismatch in {pdb_file.name}. Skipping.")
             return None, None

        return coords, res_indices

    except Exception as e:
        warnings.warn(f"Error processing PDB file {pdb_file.name}: {e}")
        return None, None

def create_protein_graph(coords: np.ndarray, res_indices: list[int], pdb_id: str) -> Data:
    """
    Creates a PyTorch Geometric Data object for a protein.

    Args:
        coords (np.ndarray): C-alpha coordinates (N, 3).
        res_indices (list[int]): List of residue type integers (N,).
        pdb_id (str): Identifier for the protein (e.g., PDB filename stem).

    Returns:
        Data: PyTorch Geometric Data object.
    """
    # Use residue indices as initial node features 'x'
    # WHAT ELSE could be node features? Maybe residue properties, secondary structure?
    # For now, keep it simple as required by the model structure later.
    node_features = torch.tensor(res_indices, dtype=torch.long)

    # Positions are the coordinates
    positions = torch.tensor(coords, dtype=torch.float32)

    # Create the Data object
    # We don't precompute edges here; the GNN will use radius_graph
    # WAIT A MINUTE, should we add sequence info or other metadata? Yes, good practice.
    data = Data(
        x=node_features,      # Node features (residue types)
        pos=positions,        # Node positions (C-alpha coordinates)
        num_nodes=len(res_indices),
        name=pdb_id           # Store the name/ID
    )
    return data

def main(args):
    """
    Main preprocessing script. Finds PDBs, processes them, saves graphs.
    """
    pdb_dir = Path(args.pdb_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # WHAT IF pdb_dir doesn't exist? Add check.
    if not pdb_dir.is_dir():
        print(f"Error: PDB directory not found: {pdb_dir}")
        return

    pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif")) # Allow CIF too? Let's stick to PDB for now.
    pdb_files = list(pdb_dir.glob("*.pdb"))

    # WHAT IF no PDB files are found?
    if not pdb_files:
        print(f"No PDB files found in {pdb_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files. Processing...")

    processed_count = 0
    skipped_count = 0
    for pdb_file in tqdm(pdb_files, desc="Processing PDBs"):
        coords, res_indices = get_alpha_carbon_coordinates(pdb_file)

        if coords is not None and res_indices is not None:
            # Check for minimum length? Maybe later in dataset loading.
            # WHAT IF the protein is very small (e.g., < 3 residues)? Might cause issues in GNN.
            if len(res_indices) < 3:
                 warnings.warn(f"Protein {pdb_file.stem} too short ({len(res_indices)} residues). Skipping.")
                 skipped_count += 1
                 continue

            protein_graph = create_protein_graph(coords, res_indices, pdb_file.stem)
            output_path = output_dir / f"{pdb_file.stem}.pt"
            torch.save(protein_graph, output_path)
            processed_count += 1
        else:
            skipped_count += 1

    print(f"\nPreprocessing complete.")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped/Errors: {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PDB files into PyTorch Geometric graphs.")
    parser.add_argument("--pdb_dir", type=str, required=True, help="Directory containing PDB files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed graph (.pt) files.")
    # Add more args if needed, e.g., --atom_filter (CA, CB, backbone...)
    args = parser.parse_args()
    main(args)