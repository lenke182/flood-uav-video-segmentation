#!/bin/sh

#SBATCH -p gpu
#SBATCH -G rtx5000:1
#SBATCH -C scratch
#SBATCH -o job-%J.out
#SBATCH -t 0:40:00
#SBATCH -c 8


./predict.sh $1 flow "${@:2}"
