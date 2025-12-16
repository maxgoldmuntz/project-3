import utils
import torch
import subprocess
import tempfile
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

BASE_MODEL = "JetBrains/Mellum-4b-base"
OUTPUT_FILE = "codegen_results.json"

def execute_python_code(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_name = f.name
    try:
        result = subprocess.run(['python', temp_name], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if os.path.exists(temp_name): os.remove(temp_name)

def main():
    args = utils.get_args()

    # CRITICAL: Validate that we have a model to test
    if not args.adapter_model or not os.path.exists(args.adapter_model):
        raise ValueError(f"Invalid adapter path: {args.adapter_model}. Did training finish?")

    print(f"Evaluating Adapter: {args.adapter_model}")

    if torch.cuda.is_available(): device = "cuda"
    elif torch.backends.mps.is_available(): device = "mps"
    else: device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base_model, args.adapter_model)
    model.eval()

    dataset = load_dataset("flytech/python-codes-25k")
    test_data = dataset["train"].train_test_split(test_size=0.2, seed=42)["test"].train_test_split(test_size=0.5, seed=42)["test"].select(range(10))

    results_container = []
    print(f"Generating for {len(test_data)} examples...")

    for example in test_data:
        instruction = example['text']
        truth = example['output']
        prompt = f"Instruction: {instruction}\nResponse:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150)

        generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_code = utils.extract_python_code(generated_full)

        results_container.append({
            "instruction": instruction, "truth": truth,
            "generated_code": generated_code, "full_response": generated_full
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_container, f, indent=4)

    with open(OUTPUT_FILE, "r") as f: data = json.load(f)
    predictions = [item['generated_code'] for item in data]
    references = [[item['truth']] for item in data]

    try:
        # Check if we have enough data for BLEU
        bleu = utils.compute_bleu(predictions, references)
        print(f"BLEU: {bleu}")
    except:
        print("BLEU calculation skipped (not enough samples or error)")

    executable = sum([1 for item in data if execute_python_code(item['generated_code'])])
    print(f"Executable Rate: {executable / len(data):.2%}")

if __name__ == "__main__":
    main()