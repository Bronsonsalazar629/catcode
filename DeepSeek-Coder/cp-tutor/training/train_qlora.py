#!/usr/bin/env python3
"""QLoRA finetuning script for CP Sensei — competitive programming tutor model.

Loads DeepSeek-Coder in 4-bit quantization and trains LoRA adapters on
competitive programming instruction/output pairs.
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

SYSTEM_PROMPT = """You are an elite competitive programming tutor trained on solutions and \
techniques from IOI gold medalists. You teach using plain English, real-world analogies, and \
step-by-step reasoning. You give hints by default, building understanding progressively from \
brute force to IOI-level optimized solutions. You only provide complete solutions when explicitly \
asked. Your language is C++. You make even the hardest algorithms feel approachable."""

IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"


def build_prompt(instruction: str) -> str:
    return f"""{SYSTEM_PROMPT}
### Instruction:
{instruction.strip()}
### Response:
"""


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def load_quantized_model(model_name: str, quant_bits: int):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(quant_bits == 4),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    return model


def apply_lora(model, config: dict):
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tokenize_dataset(dataset, tokenizer, max_seq_length: int):
    def tokenize_fn(examples):
        prompts = [build_prompt(inst) for inst in examples["instruction"]]
        completions = [f"{out}\n{EOT_TOKEN}" for out in examples["output"]]
        full_texts = [p + c for p, c in zip(prompts, completions)]

        model_inputs = tokenizer(
            full_texts,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
        )

        # Mask the prompt tokens in labels (only train on completion)
        labels = []
        for i, prompt in enumerate(prompts):
            prompt_ids = tokenizer(prompt, truncation=True, max_length=max_seq_length)["input_ids"]
            prompt_len = len(prompt_ids)
            input_len = len(model_inputs["input_ids"][i])

            label = [IGNORE_INDEX] * prompt_len + model_inputs["input_ids"][i][prompt_len:]
            # Pad label to match input length
            if len(label) < input_len:
                label = label + [IGNORE_INDEX] * (input_len - len(label))
            elif len(label) > input_len:
                label = label[:input_len]
            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    return dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def main():
    parser = argparse.ArgumentParser(description="QLoRA finetune for CP Sensei")
    parser.add_argument("--config", type=str, default="training/qlora_config.json")
    parser.add_argument("--data", type=str, default="data/train.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {config['model_name']}")
    model = load_quantized_model(config["model_name"], config["quant_bits"])
    model = apply_lora(model, config)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading training data: {args.data}")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Training samples: {len(dataset)}")

    tokenized = tokenize_dataset(dataset, tokenizer, config["max_seq_length"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler"],
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
