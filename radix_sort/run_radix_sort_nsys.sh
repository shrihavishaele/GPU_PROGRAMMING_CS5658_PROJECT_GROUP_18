#!/bin/bash
#SBATCH --job-name=radix_sort_prof
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

NSYS=/opt/nvidia/nsight-systems/2023.1.2/target-linux-x64/nsys
"$NSYS" profile \
    --trace=cuda \
    --force-overwrite=true \
    --output="output/radix_sort_report_${SLURM_JOB_ID}" \
    ./radix_sort

echo "Job completed"
