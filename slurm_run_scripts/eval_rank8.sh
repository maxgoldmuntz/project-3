#!/bin/bash
#SBATCH --job-name="eval_r8"
#SBATCH --output="logs/eval_rank8.out"
#SBATCH --error="logs/eval_rank8.err"
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

# Fix memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting evaluation for Rank 8..."
uv run eval_task2.py --adapter_model ./results/code_r8_sz1.0 --model_name JetBrains/Mellum-4b-base
