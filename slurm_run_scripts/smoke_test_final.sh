#!/bin/bash
set -e
echo "========================================================"
echo "FINAL INTEGRATION TEST: FAST MODE"
echo "========================================================"

rm -rf ./debug_results
mkdir -p ./debug_results

# --- PART 1: MAIN FLOW ---
echo ">>> [1/4] Training QA (Task 1)..."
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 0.01 --output_dir ./debug_results --epochs 1 --max_steps 5 --max_eval_samples 10

echo ">>> [2/4] Training CodeGen (Task 2)..."
uv run train_task2.py --task codegen --model_name JetBrains/Mellum-4b-base --lora_rank 8 --train_size 0.01 --output_dir ./debug_results --epochs 1 --max_steps 5 --max_eval_samples 10

echo ">>> [3/4] Evaluating CodeGen..."
uv run eval_task2.py --model_name JetBrains/Mellum-4b-base --adapter_model ./debug_results/code_r8_sz0.01

# --- PART 2: EXPERIMENTS ---
echo ">>> [4/4] Testing Experiments..."
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 0.3 --output_dir ./debug_results --max_steps 2 --max_eval_samples 10
uv run train_task1.py --task qa --model_name roberta-base --lora_rank 16 --train_size 0.01 --output_dir ./debug_results --max_steps 2 --max_eval_samples 10

echo "ALL TESTS PASSED."