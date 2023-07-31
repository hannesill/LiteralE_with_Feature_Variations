#!/bin/bash

#SBATCH --partition=study
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=linkpred
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00

# Checkout right git branch
rm data/fb15k-237/processed.pt
# Load the conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate LiteralE

python3 main.py --lit_mode num --eta 5
