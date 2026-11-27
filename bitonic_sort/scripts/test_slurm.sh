#!/bin/bash
#SBATCH --job-name=bitonic_sort
#SBATCH --output=output/%j.out
#SBATCH --error=error/%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu02

set -e
module purge
module load cuda/11.3


./bitonic_sort

echo "Testing completed"