#!/bin/sh

#SBATCH -p gpu
#SBATCH -G rtx5000:1
#SBATCH -C scratch
#SBATCH -o job-%J.out
#SBATCH -t 8:00:00
#SBATCH -c 8
#SBATCH --array=1-40


wandb agent --count 1 username/project_name/$1
