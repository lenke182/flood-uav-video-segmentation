#!/bin/sh

#SBATCH -p gpu
#SBATCH -G rtx5000:1
#SBATCH -C scratch
#SBATCH -o job-%J.out
#SBATCH -t 20:00:00
#SBATCH -c 8


module purge
module load python/3.9.0 cuda/11.5.1 gcc/10.2.0

export CXX=g++
export WANDB_DISABLE_SERVICE=True
pipenv run $1 test --config "configs/train_base.yaml" --config "configs/train_$1.yaml" --tag $2 --config "logs/$3/version_0/config.yaml" --runid $3 --ckpt_path "logs/$3/last.ckpt" --config "dataset/$2/config.yaml" --wandb "" "${@:4}"