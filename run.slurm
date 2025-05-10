#!/bin/bash
#SBATCH --job-name=ddpm_cifar100lt
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=swanb045@umn.edu

module load cuda
module load anaconda3
source activate ddpm
cd capstone

# Run your script
python -u train_ddpm_cifar100lt.py
