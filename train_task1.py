import utils
import data_loader
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import os

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions_start = np.argmax(logits[0], axis=-1)
    predictions_end = np.argmax(logits[1], axis=-1)
    acc_start = (predictions_start == labels[0]).mean()
    acc_end = (predictions_end == labels[1]).mean()
    return {"accuracy": (acc_start + acc_end) / 2}

def main():
    args = utils.get_args()
    print(f"Loading SQUAD for {args.model_name}...")
    train_dataset, val_dataset, tokenizer = data_loader.get_qa_dataset(args.model_name, args.train_size)

    # NEW: Safety Limit for Smoke Test
    if args.max_eval_samples > 0:
        print(f"⚠️ Limiting validation set to {args.max_eval_samples} samples for testing.")
        val_dataset = val_dataset.select(range(min(len(val_dataset), args.max_eval_samples)))

    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    peft_config = LoraConfig(task_type=TaskType.QUESTION_ANS, inference_mode=False, r=args.lora_rank, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    output_path = args.output_dir #os.path.join(args.output_dir, f"qa_r{args.lora_rank}_sz{args.train_size}")

    training_args = TrainingArguments(
        output_dir=output_path,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=8,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        remove_unused_columns=False,
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model()
    # --- SAVE SCORES AUTOMATICALLY
    print(">>> Saving Final Scores...")
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    trainer.save_state()
    # ---------------------------------------------

if __name__ == "__main__":
    main()
