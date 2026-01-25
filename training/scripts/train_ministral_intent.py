#!/usr/bin/env python3
"""
Train Ministral for intent classification using QLoRA with Unsloth.

Uses Unsloth's FastLanguageModel for optimized training (~2x faster, lower memory).

Usage:
    python -m training.scripts.train_ministral_intent \
        --config training/config/ministral_intent.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import yaml
from datasets import Dataset
from trl import SFTTrainer
from unsloth import FastVisionModel

from training.utils.callbacks import LoggingCallback, MemoryCallback


# Prompt template for intent classification
INTENT_PROMPT = """[INST] Classifie l'intention: TRIP (voyage) ou NOT_TRIP (autre).
Texte: {text}
Intention: [/INST]"""


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_intent_dataset(
    file_path: Path,
    tokenizer,
    num_samples: Optional[int] = None,
) -> Dataset:
    """
    Prepare dataset for generative intent classification.

    Args:
        file_path: Path to JSON data file
        tokenizer: Tokenizer instance
        num_samples: Optional limit on samples

    Returns:
        HuggingFace Dataset with formatted texts
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if num_samples:
        data = data[:num_samples]

    formatted_texts = []
    for sample in data:
        prompt = INTENT_PROMPT.format(text=sample["sentence"])
        target = sample["intent"]
        full_text = prompt + target + tokenizer.eos_token
        formatted_texts.append(full_text)

    return Dataset.from_dict({"text": formatted_texts})


def main(config_path: str) -> None:
    """Run QLoRA training for intent classification using Unsloth."""
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Extract config sections
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    data_config = config["data"]
    debug_config = config.get("debug", {})

    print(f"\nModel: {model_config['name']}")
    print(f"Output dir: {training_config['output_dir']}")

    # Load model and tokenizer with Unsloth (FastVisionModel for Ministral 3)
    print("\nLoading model with Unsloth FastVisionModel...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_config["name"],
        max_seq_length=model_config.get("max_seq_length", 2048),
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Setup padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA with Unsloth (optimized for vision model)
    print("Applying LoRA with Unsloth optimizations...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Text-only task, skip vision
        finetune_language_layers=True,  # Fine-tune language layers
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        lora_dropout=0,  # MUST be 0 for Unsloth fast patching
        target_modules=lora_config["target_modules"],
        bias=lora_config["bias"],
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=42,
    )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = prepare_intent_dataset(
        Path(data_config["train_file"]),
        tokenizer,
        num_samples=debug_config.get("num_train_samples"),
    )
    val_dataset = prepare_intent_dataset(
        Path(data_config["val_file"]),
        tokenizer,
        num_samples=debug_config.get("num_val_samples"),
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Training arguments (using trl's SFTConfig via SFTTrainer)
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        fp16=training_config["fp16"],
        logging_steps=training_config["logging_steps"],
        eval_strategy=training_config["eval_strategy"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        save_total_limit=training_config["save_total_limit"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        report_to=training_config.get("report_to", "none"),
        optim="adamw_8bit",  # Unsloth recommended optimizer
        seed=42,
    )

    # Callbacks
    callbacks = [LoggingCallback(), MemoryCallback()]

    # Initialize SFTTrainer (optimized for instruction fine-tuning)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=data_config["max_length"],
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 50)
    print("Starting training with Unsloth...")
    print("=" * 50 + "\n")

    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {training_config['output_dir']}...")
    model.save_pretrained(training_config["output_dir"])
    tokenizer.save_pretrained(training_config["output_dir"])

    print("\nTraining complete!")
    print(f"LoRA adapter saved to: {training_config['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Ministral for intent classification with Unsloth"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/config/ministral_intent.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    main(args.config)
