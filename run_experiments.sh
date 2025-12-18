#!/bin/bash
# ====================================================
#  PART 1: ABLATION EXPERIMENTS ONLY
# ====================================================

# Cleanup logs for just these experiments
rm -f logs/exp_*.out logs/exp_*.err

echo ">>> Launching Experiments (Batch Size 2, Safe)..."

# --- QA Size Ablation ---
sbatch --job-name="exp_qa_30" --output="logs/exp_qa_sz30.out" --error="logs/exp_qa_sz30.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 0.3 --output_dir ./results/qa_r8_sz0.3"

sbatch --job-name="exp_qa_50" --output="logs/exp_qa_sz50.out" --error="logs/exp_qa_sz50.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 0.5 --output_dir ./results/qa_r8_sz0.5"

# --- QA Rank Variation ---
sbatch --job-name="exp_qa_r16" --output="logs/exp_qa_r16.out" --error="logs/exp_qa_r16.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task1.py --task qa --model_name roberta-base --lora_rank 16 --train_size 1.0 --output_dir ./results/qa_r16_sz1.0"

sbatch --job-name="exp_qa_r32" --output="logs/exp_qa_r32.out" --error="logs/exp_qa_r32.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task1.py --task qa --model_name roberta-base --lora_rank 32 --train_size 1.0 --output_dir ./results/qa_r32_sz1.0"

# --- Code Rank Variation ---
sbatch --job-name="exp_cd_r16" --output="logs/exp_code_r16.out" --error="logs/exp_code_r16.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task2.py --task codegen --model_name JetBrains/Mellum-4b-base --lora_rank 16 --output_dir ./results/code_r16_sz1.0 --batch_size 2"

sbatch --job-name="exp_cd_r32" --output="logs/exp_code_r32.out" --error="logs/exp_code_r32.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task2.py --task codegen --model_name JetBrains/Mellum-4b-base --lora_rank 32 --output_dir ./results/code_r32_sz1.0 --batch_size 2"

echo ">>> Experiments submitted."
