import utils
import torch
import subprocess
import tempfile
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

BASE_MODEL = "JetBrains/Mellum-4b-base"
OUTPUT_DIR = "results/base_model_baseline"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def execute_python_code(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_name = f.name
    try:
        # 5 second timeout
        result = subprocess.run(['python', temp_name], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False
    finally:
        if os.path.exists(temp_name): os.remove(temp_name)

def main():
    print(f"Evaluating BASE MODEL: {BASE_MODEL}")

    # Load Tokenizer & Model (NO ADAPTER)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    model.eval()

    # Same 50 examples
    dataset = load_dataset("flytech/python-codes-25k")
    test_data = dataset["train"].train_test_split(test_size=0.2, seed=42)["test"].train_test_split(test_size=0.5, seed=42)["test"].select(range(50))

    results_container = []
    print(f"Generating for {len(test_data)} examples...")

    for i, example in enumerate(test_data):
        instruction = example['text']
        truth = example['output']
        prompt = f"Instruction: {instruction}\nResponse:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150)

        generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = utils.extract_python_code(generated_full)
        
        if (i + 1) % 10 == 0: print(f"  Processed {i + 1}/{len(test_data)}")

        results_container.append({
            "instruction": instruction, "truth": truth, 
            "generated_code": generated_code
        })

    # Save outputs
    with open(os.path.join(OUTPUT_DIR, "codegen_results.json"), "w") as f:
        json.dump(results_container, f, indent=4)

    # Calculate Metrics
    predictions = [item['generated_code'] for item in results_container]
    references = [[item['truth']] for item in results_container]
    
    metrics = {}
    try:
        metrics["bleu"] = utils.compute_bleu(predictions, references)
    except:
        metrics["bleu"] = 0.0
    
    exec_count = sum([1 for item in results_container if execute_python_code(item['generated_code'])])
    metrics["executability"] = exec_count / len(results_container)

    print(f"FINAL BASE METRICS: {metrics}")

    # Save Summary
    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
