from datasets import load_dataset
from transformers import AutoTokenizer

def get_qa_dataset(model_name, train_size=1.0):
    dataset = load_dataset("squad_v2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_set = split_dataset["train"]
    val_set = split_dataset["test"]

    if train_size < 1.0:
        train_set = train_set.select(range(int(len(train_set) * train_size)))

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions, examples["context"],
            max_length=384, truncation="only_second",
            return_overflowing_tokens=True, return_offsets_mapping=True,
            stride=128, padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0] if len(answer["answer_start"]) > 0 else 0
            end_char = start_char + len(answer["text"][0]) if len(answer["text"]) > 0 else 0
            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1: idx += 1
            context_start = idx
            while sequence_ids[idx] == 1: idx += 1
            context_end = idx - 1

            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char: idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_train = train_set.map(preprocess_function, batched=True, remove_columns=train_set.column_names)
    tokenized_val = val_set.map(preprocess_function, batched=True, remove_columns=val_set.column_names)
    return tokenized_train, tokenized_val, tokenizer

def get_codegen_dataset(model_name, train_size=1.0):
    dataset = load_dataset("flytech/python-codes-25k")
    train_test = dataset["train"].train_test_split(test_size=0.2, seed=42)
    test_val = train_test["test"].train_test_split(test_size=0.5, seed=42)
    train_data = train_test["train"]
    val_data = test_val["train"]

    if train_size < 1.0:
        train_data = train_data.select(range(int(len(train_data) * train_size)))

    def formatting_prompts_func(example):
        text_data = example['text']
        output_data = example['output']
        if isinstance(text_data, list):
            output_texts = []
            for text, output in zip(text_data, output_data):
                output_texts.append(f"Instruction: {text}\nResponse:\n```python\n{output}\n```")
            return output_texts
        else:
            return f"Instruction: {text_data}\nResponse:\n```python\n{output_data}\n```"

    return train_data, val_data, formatting_prompts_func