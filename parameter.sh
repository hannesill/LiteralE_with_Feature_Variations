#!/bin/bash

#SBATCH --job-name=DistMultModel
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -w handy
#SBATCH --time=02:00:00

# Load the conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate LiteralE

python main.py
