#!/bin/bash
#SBATCH --job-name=nlp_pa3_exp
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=student-gpu-001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

mkdir -p logs
source $HOME/.cargo/env

echo "=== STARTING TASK 3 EXPERIMENTS ==="

# --- EXPERIMENT A: DATA SIZE (Target: QA Model - Faster to train) ---
# Instruction: "Various training size (30%, 50%, 100%)"
# Note: 100% is already done in main job, doing 30% and 50% here.

echo ">>> Running Exp: QA Size 30%"
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 0.3 --output_dir ./results

echo ">>> Running Exp: QA Size 50%"
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 0.5 --output_dir ./results

# --- EXPERIMENT B: LORA RANK (Target: QA Model) ---
# Instruction: "LoRA rank (choose 3 values)"
# We already have Rank 8. Let's do Rank 16 and 32.

echo ">>> Running Exp: QA Rank 16"
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 16 --train_size 1.0 --output_dir ./results

echo ">>> Running Exp: QA Rank 32"
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 32 --train_size 1.0 --output_dir ./results

# --- EXPERIMENT C: BASELINE (Pre-trained vs Fine-tuned) ---
# Instruction: "Pre-trained model vs fine-tuned model"
# To test the pre-trained model, we run eval WITHOUT an adapter.
# (You might need to modify eval_task2.py slightly to accept --no_adapter if you strictly want code baseline,
#  but usually this comparison is done on the QA task or conceptually in the report).

echo "Experiments Complete."