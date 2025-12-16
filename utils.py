import argparse
import re
import evaluate
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description="NLP PA3")

    # Task selection
    parser.add_argument("--task", type=str, required=False, choices=["qa", "codegen"])
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--output_dir", type=str, default="./results")

    # Experiments
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--train_size", type=float, default=1.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_steps", type=int, default=-1, help="Smoke test limit")

    # Limit validation size for smoke tests
    parser.add_argument("--max_eval_samples", type=int, default=-1, help="Limit val set size")

    # Evaluation
    parser.add_argument("--adapter_model", type=str, default=None, help="Path to trained adapter for eval")

    return parser.parse_args()

def extract_python_code(text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.strip()

def compute_bleu(predictions, references):
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=predictions, references=references)["bleu"]