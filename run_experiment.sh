#!/bin/bash

#SBATCH --job-name=unlearn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=02:59:00
#SBATCH --output=slurm_logs/run_%A_%a.out
#SBATCH --array=1
#SBATCH --job-name=TTA

# Load necessary modules or activate virtual environment
source activate llm-tta

# Run the Python script with Hydra's grid search
# The SLURM_ARRAY_TASK_ID environment variable will be different for each job in the array
# python main.py --multirun edit_set=$SLURM_ARRAY_TASK_ID number_of_edits=50 edit=True\
#  compress=True save_ckpt=False method=prune sparsity_ratio=0.35\
#  tag=exp_memit_wanda35
# python main.py seed=$SLURM_ARRAY_TASK_ID $@
make $1 