#!/bin/bash

#SBATCH --partition=study
#SBATCH --job-name=linkpred
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00

# Checkout right git branch

# Load the conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate LiteralE

python3 main.py --lit_mode all --scoring ConvE --epochs 300 --eta 50 --emb_dim 100
