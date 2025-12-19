#!/bin/bash
#SBATCH --job-name="eval_others"
#SBATCH --output="logs/eval_others.out"
#SBATCH --error="logs/eval_others.err"
#SBATCH --time=01:00:00
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

# Fix memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> Evaluating Rank 16..."
uv run eval_task2.py --adapter_model ./results/code_r16_sz1.0 --model_name JetBrains/Mellum-4b-base

echo "----------------------------------------------------"

echo ">>> Evaluating Rank 32..."
uv run eval_task2.py --adapter_model ./results/code_r32_sz1.0 --model_name JetBrains/Mellum-4b-base

echo ">>> Done."
