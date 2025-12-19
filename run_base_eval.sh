#!/bin/bash
#SBATCH --job-name="eval_base"
#SBATCH --output="logs/eval_base.out"
#SBATCH --error="logs/eval_base.err"
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting Base Model Evaluation..."
uv run eval_base.py
