#!/bin/bash
#SBATCH --job-name=radix_sort_grp18
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

echo "======================================================="
echo " CS5658 GPU Programming — Group 18"
echo " Parallel Radix Sort on NVIDIA CUDA"
echo "======================================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "======================================================="

nvidia-smi

echo ""
echo "=== Running Radix Sort Test Suite ==="
./radix_sort

echo ""
echo "Job completed"
echo "Finished : $(date)"
