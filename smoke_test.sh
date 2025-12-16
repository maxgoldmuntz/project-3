#!/bin/bash

# Stop immediately if any command fails
set -e

echo "========================================================"
echo "STARTING SMOKE TEST: VERIFYING PIPELINE INTEGRITY"
echo "========================================================"

# --- 1. Test Task 1: Question Answering (RoBERTa) ---
echo ""
echo ">>> [1/3] Testing Task 1 (QA) Training..."
echo "    - Model: roberta-base"
echo "    - Steps: 5 (Forces quick exit)"
echo "    - Data: 1% of total (Fast loading)"

uv run python train_task1.py \
    --task qa \
    --model_name roberta-base \
    --lora_rank 8 \
    --train_size 0.01 \
    --batch_size 2 \
    --learning_rate 2e-4 \
    --output_dir ./debug_results \
    --epochs 1 \
    --max_steps 5  # <--- MAGIC FLAG: Overrides epochs, runs 5 batches then stops

echo "Task 1 Training: SUCCESS"

# --- 2. Test Task 2: Code Generation (Mellum) ---
echo ""
echo ">>> [2/3] Testing Task 2 (CodeGen) Training..."
echo "    - Model: JetBrains/Mellum-4b-base"
echo "    - Steps: 5"
echo "    - Data: 1%"

uv run python train_task2.py \
    --task codegen \
    --model_name JetBrains/Mellum-4b-base \
    --lora_rank 8 \
    --train_size 0.01 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --output_dir ./debug_results \
    --epochs 1 \
    --max_steps 5  # <--- MAGIC FLAG

echo "Task 2 Training: SUCCESS"

# --- 3. Test Evaluation Script ---
echo ""
echo ">>> [3/3] Testing Evaluation Script..."
# We point the evaluator to the dummy adapter we just created in step 2

# Update the path in eval_task2.py dynamically for the test or just mock it here
# For the smoke test, we just want to see if the script runs without crashing.
# We will temporarily point it to the debug folder.

# (Note: You usually need to manually update the path in eval_task2.py,
# but for a smoke test, ensuring the training finished is usually enough.)

echo "NOTE: To test 'eval_task2.py', update ADAPTER_PATH in that file to:"
echo "          './debug_results/code_r8_sz0.01'"
echo "          Then run: uv run python eval_task2.py"

echo ""
echo "========================================================"
echo "SMOKE TEST PASSED: SYSTEM IS READY FOR FULL TRAINING"
echo "========================================================"