#!/bin/bash
# ====================================================
#  PART 2: MAIN TASKS ONLY (Optimized Mode)
# ====================================================

# 1. CRITICAL ENVIRONMENT FIXES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo ">>> Launching Main Tasks (Batch Size 2)..."

# ==========================================
# TASK 1: Question Answering (RoBERTa)
# ==========================================
# RoBERTa is small, so we can stick to standard settings (Batch 8 is fine here usually)
sbatch --job-name="main_qa" \
       --output="logs/main_task_qa.out" \
       --error="logs/main_task_qa.err" \
       --time=04:00:00 --partition=regular --gres=gpu:1 --mem=24G \
       --wrap="uv run train_task1.py --task qa --model_name roberta-base --lora_rank 8 --train_size 1.0 --output_dir ./results/qa_r8_sz1.0"

# ==========================================
# TASK 2: Code Generation (Mellum)
# ==========================================
# SETTINGS: Batch Size 2 (Proven Safe) 
# Note: We removed the explicit gradient accumulation flag because your 
# python script hardcodes it or doesn't accept the arg, but Batch 2 is much faster than 1.

echo ">>> Launching Daisy Chain for Code Task..."

# STEP A: Part 1 (Training)
JOB_ID_1=$(sbatch --parsable \
    --job-name="main_code_1" \
    --output="logs/main_task_code.out" \
    --error="logs/main_task_code.err" \
    --time=04:00:00 --partition=regular --gres=gpu:1 --mem=32G \
    --wrap="uv run train_task2.py --task codegen --model_name JetBrains/Mellum-4b-base --lora_rank 8 --output_dir ./results/code_r8_sz1.0 --batch_size 2")

echo "   -> Part 1 Submitted (ID: $JOB_ID_1)"

# STEP B: Part 2 (Auto-Resume)
JOB_ID_2=$(sbatch --parsable \
    --dependency=afterany:$JOB_ID_1 \
    --job-name="main_code_2" \
    --output="logs/main_task_code.out" \
    --error="logs/main_task_code.err" \
    --open-mode=append \
    --time=04:00:00 --partition=regular --gres=gpu:1 --mem=32G \
    --wrap="ckpt=\$(ls -td ./results/code_r8_sz1.0/checkpoint-* 2>/dev/null | head -1); if [ -z \"\$ckpt\" ]; then echo 'ERROR: No checkpoint found.'; exit 1; fi; echo 'Resuming from: \$ckpt'; uv run train_task2.py --task codegen --model_name \$ckpt --lora_rank 8 --output_dir ./results/code_r8_sz1.0 --batch_size 2")

echo "   -> Part 2 Submitted (ID: $JOB_ID_2)"

# STEP C: Evaluation
JOB_ID_3=$(sbatch --parsable \
    --dependency=afterok:$JOB_ID_2 \
    --job-name="main_eval" \
    --output="logs/main_task_eval.out" \
    --error="logs/main_task_eval.err" \
    --time=04:00:00 --partition=regular --gres=gpu:1 --mem=32G \
    --wrap="uv run eval_task2.py --adapter_model ./results/code_r8_sz1.0 --model_name JetBrains/Mellum-4b-base")

echo "   -> Evaluation Submitted (ID: $JOB_ID_3)"
