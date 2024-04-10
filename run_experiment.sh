#!/bin/bash

#SBATCH --job-name=unlearn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=250GB
#SBATCH --output=slurm_logs/run_%A_%a.out
#SBATCH --array=1
#SBATCH --job-name=TTA

export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1


source activate llm-tta
experiment_make=$1
seed=$2
make $experiment_make SEED=$seed