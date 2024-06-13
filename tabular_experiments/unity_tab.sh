#!/bin/bash
#SBATCH -c 48  # Number of Cores per Task
#SBATCH --mem=10GB  # Requested Memory
#SBATCH -p cpu  # Partition
##SBATCH -G 1  # Number of GPUs
#SBATCH -t 23:00:00  # Job time limit
#SBATCH -o outfiles/%j.out  # %j = job ID

#SBATCH --array=1-5

# module load cuda/10.1.243
# /modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
# Load the conda environment:
module load miniconda/22.11.1-1
# eval "$(conda shell.bash hook)"
conda activate /home/jacob_adamczyk001_umb_edu/.conda/envs/rlenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib


python tabular_lr_sensitivity.py