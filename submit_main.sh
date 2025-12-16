#!/bin/bash
#SBATCH --job-name=nlp_pa3_main
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err
#SBATCH --partition=student-gpu-001   # Or 'regular' or 'gpu48g'
#SBATCH --gres=gpu:1                  # Mandatory: Request 1 GPU
#SBATCH --cpus-per-task=4             # Request 4 CPUs
#SBATCH --mem=32G                     # Request 32GB RAM
#SBATCH --time=04:00:00               # Max duration

# 1. Setup
mkdir -p logs
echo "Job running on node: $(hostname)"
source $HOME/.cargo/env  # Ensure uv is loaded

# 2. Task 1: Question Answering (RoBERTa)
# Instruction: "fine-tune RoBERTa with LoRA on the SQUAD-v2 dataset"
echo ">>> [Task 1] Starting QA Training..."
uv run train_task1.py \
    --task qa \
    --model_name roberta-base \
    --lora_rank 8 \
    --train_size 1.0 \
    --output_dir ./results

# 3. Task 2: Code Generation (Mellum)
# Instruction: "fine-tune... JetBrains/Mellum-4b-base... using trl's SFTTrainer"
echo ">>> [Task 2] Starting CodeGen Training..."
uv run train_task2.py \
    --task codegen \
    --model_name JetBrains/Mellum-4b-base \
    --lora_rank 8 \
    --train_size 1.0 \
    --output_dir ./results

# 4. Task 2: Evaluation
# Instruction: "Export the container... Read the exported file and start your evaluation"
echo ">>> [Task 2] Starting Evaluation..."
# Note: We pass the specific adapter path created in step 3
uv run eval_task2.py \
    --adapter_model ./results/code_r8_sz1.0 \
    --model_name JetBrains/Mellum-4b-base

echo "Main Job Complete."