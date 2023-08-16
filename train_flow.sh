#!/bin/sh

#SBATCH -p gpu
#SBATCH -G rtx5000:1
#SBATCH -C scratch
#SBATCH -o job-%J.out
#SBATCH -t 12:00:00
#SBATCH -c 8


./train.sh $1 flow "${@:2}"
