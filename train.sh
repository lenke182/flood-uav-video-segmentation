#!/bin/sh

#SBATCH -p gpu
#SBATCH -G rtx5000:1
#SBATCH -C scratch
#SBATCH -o job-%J.out
#SBATCH -t 12:00:00
#SBATCH -c 8


module purge
module load python/3.9.0 cuda/11.5.1 gcc/10.2.0

export CXX=g++
export WANDB_DISABLE_SERVICE=True
pipenv run $1 fit  --config "configs/train_base.yaml" --config "configs/train_$1.yaml" --config "dataset/$2/config.yaml" --tag $1 --tag $2 "${@:3}"
