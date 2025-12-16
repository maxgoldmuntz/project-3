import utils
import data_loader
from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
import torch
import os

def main():
    args = utils.get_args()
    use_cuda = torch.cuda.is_available()
    is_mac_mps = torch.backends.mps.is_available()

    if use_cuda:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", quantization_config=bnb_config)
    elif is_mac_mps:
        print("⚠️  Mac M-Series detected: Optimizing for memory...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="mps", torch_dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    train_data, val_data, formatting_func = data_loader.get_codegen_dataset(args.model_name, args.train_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # NEW: Safety Limit for Smoke Test
    if args.max_eval_samples > 0:
        print(f"⚠️ Limiting validation set to {args.max_eval_samples} samples for testing.")
        val_data = val_data.select(range(min(len(val_data), args.max_eval_samples)))

    peft_config = LoraConfig(r=args.lora_rank, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    output_path = os.path.join(args.output_dir, f"code_r{args.lora_rank}_sz{args.train_size}")

    actual_batch_size = 1 if is_mac_mps else args.batch_size
    use_gradient_checkpointing = True if is_mac_mps else False

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=use_cuda, bf16=is_mac_mps,
        max_steps=args.max_steps,
        use_cpu=False,
        gradient_checkpointing=use_gradient_checkpointing,
    )

    trainer = SFTTrainer(
        model=model, train_dataset=train_data, eval_dataset=val_data,
        formatting_func=formatting_func, peft_config=peft_config,
        processing_class=tokenizer, args=training_args,
    )
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()