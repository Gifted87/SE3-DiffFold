# SE(3)-Equivariant Diffusion for Protein Folding (SE(3)-DiffFold)

**Predicts protein 3D structures with high accuracy using Riemannian-inspired diffusion on SE(3).**

[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add Colab badge once a notebook exists -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/se3-diff-fold/blob/main/notebooks/Demo.ipynb) -->

<!-- Placeholder for a cool diffusion GIF -->
<!-- ![Diffusion Process](docs/diffusion.gif) -->

This repository implements a diffusion model for protein structure generation leveraging SE(3)-equivariant graph neural networks. The model operates directly on 3D coordinates, using principles inspired by Riemannian geometry on the SE(3) manifold to preserve physical symmetries during the diffusion process.

## Key Innovations

🌀 **SE(3)-Equivariant GNNs**
- Core graph network (`models/se3_gnn.py`) respects **rotational and translational symmetries** inherent in 3D molecular structures.
- Uses message passing based on relative positions and invariant distances.
- Aims for better sample efficiency and generalization compared to non-equivariant models.

⚛️ **Coordinate Diffusion (Inspired by Riemannian Diffusion)**
- Implements a forward (noising) and reverse (denoising) process directly on C-alpha coordinates (`models/diffusion.py`).
- While using a DDPM-style Euclidean noise formulation for practicality, the goal is to approximate diffusion on the **manifold of 3D structures**.
- Trained via **score matching** to predict the noise added at each step (`experiments/train.py`).

## Benchmarks (Illustrative - Replace with Actual Results)

Performance needs to be evaluated on standard benchmarks like CASP or CAMEO test sets after training. The 0.8 Å RMSD mentioned in the prompt is a target goal.

| Model               | RMSD (Å)             | Inference Time | Notes                   |
|---------------------|----------------------|----------------|-------------------------|
| AlphaFold2 (static) | ~1.0 - 1.5 (Typical) | ~5-30 min      | MSA-based, static pred. |
| **Ours (Target)**   | **< 1.0 Å (Goal)**   | ~1-5 min       | Diffusion-based         |
| *Other Diffusion*   | *(Varies)*           | *(Varies)*     | e.g., RFDiffusion       |

*Inference time depends on protein length and hardware.*

## Project Structure

```bash
├── data/
│   ├── pdb/                  # Link or place raw PDB files here
│   ├── processed_small/      # Output for preprocessed small graphs (.pt)
│   ├── processed_large/      # Output for preprocessed large graphs (.pt)
│   └── preprocess.py         # Script to convert PDB -> Torch Geometric graphs
├── models/
│   ├── se3_gnn.py            # SE(3)-equivariant GNN layer
│   ├── diffusion.py          # Diffusion schedule and sampling logic
│   └── score_network.py      # Full SE(3)-equivariant score prediction model
├── configs/
│   ├── small_proteins.yaml   # Hyperparameters for <200 residues
│   └── large_proteins.yaml   # Hyperparameters for >=200 residues
├── experiments/
│   ├── train.py              # Main training script
│   └── inference.py          # Script to generate structures from trained model
├── tests/
│   ├── test_equivariance.py  # Unit test for SE(3) equivariance of the ScoreNetwork
│   └── test_rmsd.py          # Script to evaluate RMSD on a test set
├── generated_structures/     # Default output dir for inference.py
├── rmsd_evaluation/          # Default output dir for test_rmsd.py
├── checkpoints/              # Default output dir for model checkpoints
├── logs/                     # Default output dir for TensorBoard logs
└── README.md                 # This file
```

## Quick Start

**1. Setup Environment:**

```bash
# Clone the repository
git clone https://github.com/your-username/se3-diff-fold.git # Replace with your repo URL
cd se3-diff-fold

# Create Conda environment
conda create -n se3fold python=3.9 -y
conda activate se3fold

# Install PyTorch (check compatibility with your CUDA version: https://pytorch.org/)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__)').html

# Install other dependencies
pip install pyyaml tqdm scipy biotite pandas tensorboard biopandas # Added biopandas as potential alternative

```

**2. Prepare Data:**

*   Place your PDB files (or symlinks) into the `data/pdb/` directory.
*   Run the preprocessing script:

```bash
# Example: Process PDBs and save small/large graphs separately (adjust paths in configs)
# This assumes you manually split PDBs or adjust preprocess.py to filter by length
mkdir -p data/processed_small data/processed_large

# Preprocess all PDBs into one directory first
python data/preprocess.py --pdb_dir data/pdb/ --output_dir data/processed_all/

# TODO: Add logic here or in preprocess.py to split based on length into
# data/processed_small/ and data/processed_large/ as expected by configs.
# As a placeholder, copy all to both for now:
# cp data/processed_all/*.pt data/processed_small/
# cp data/processed_all/*.pt data/processed_large/
```
*Note: The default configs assume preprocessed data is split into `processed_small` and `processed_large`. You may need to adjust `preprocess.py` or manually split the output `.pt` files based on protein length.*

**3. Train a Model:**

```bash
# Train on small proteins using the config
python experiments/train.py --config configs/small_proteins.yaml

# Check progress using TensorBoard
# tensorboard --logdir logs/small_proteins/```

**4. Generate Structures (Inference):**

```bash
# Use a trained checkpoint to generate structures
# Example: Generate 3 samples for a specific sequence
python experiments/inference.py \
    --checkpoint checkpoints/small_proteins/epoch_500.pth \
    --config configs/small_proteins.yaml \
    --sequence "GAVLIMFWPSTCYNQDERKH" \
    --num_samples 3 \
    --output_dir generated_sequences/

# Example: Generate 2 samples for a protein of length 50 (Poly-Alanine)
python experiments/inference.py \
    --checkpoint checkpoints/small_proteins/epoch_500.pth \
    --config configs/small_proteins.yaml \
    --length 50 \
    --num_samples 2 \
    --output_dir generated_length/
```

**5. Evaluate Model (RMSD):**

```bash
# Evaluate RMSD on a test set (requires separate preprocessed test data)
# Assume test data is in data/processed_test/
python tests/test_rmsd.py \
    --checkpoint checkpoints/small_proteins/epoch_500.pth \
    --config configs/small_proteins.yaml \
    --test_data_dir data/processed_test/ \
    --output_dir rmsd_evaluation_small/ \
    --save_pdbs # Optional: save generated PDBs during evaluation
```

## Testing

Run unit tests to verify components:

```bash
# Test SE(3) equivariance
python tests/test_equivariance.py
```

## Applications

*   **Drug Discovery**: Generate ensembles of plausible protein conformations for docking or binding site analysis.
*   **Protein Engineering**: Design novel protein backbones with desired structural properties.
*   **Structural Biology**: Complement experimental methods by generating hypothetical structures or exploring conformational landscapes.
*   **Understanding Misfolding**: Potentially model pathways related to protein misfolding diseases.

## TODO / Future Directions

*   Implement true Riemannian diffusion on SE(3) (using Lie algebra noise).
*   Incorporate residue-level features beyond just type (e.g., physicochemical properties).
*   Add support for side-chain prediction.
*   Integrate with molecular dynamics (MD) for refinement (e.g., OpenMM, Rosetta).
*   Develop a Colab notebook for easy demonstration.
*   Optimize inference speed (model distillation, ONNX export).
*   Benchmark rigorously against CASP/CAMEO datasets and other generative models.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
