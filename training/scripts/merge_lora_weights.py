#!/usr/bin/env python3
"""
Merge LoRA adapter weights with the base model.

After training, this script merges the LoRA weights with the base model
to create a standalone model that can be used without PEFT.

Usage:
    python -m training.scripts.merge_lora_weights \
        --base-model unsloth/Ministral-3-3B-Base-2512-bnb-4bit \
        --adapter-path models/ministral-intent-lora \
        --output-dir models/ministral-intent-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_weights(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
) -> None:
    """
    Merge LoRA adapter with base model and save.

    Args:
        base_model_name: HuggingFace model name
        adapter_path: Path to LoRA adapter
        output_dir: Directory to save merged model
        push_to_hub: Whether to push to HuggingFace Hub
        hub_repo_id: Repository ID for Hub
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if push_to_hub and hub_repo_id:
        print(f"Pushing to HuggingFace Hub: {hub_repo_id}")
        merged_model.push_to_hub(hub_repo_id)
        tokenizer.push_to_hub(hub_repo_id)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/Ministral-3-3B-Base-2512-bnb-4bit",
        help="Base model name",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID",
    )

    args = parser.parse_args()

    merge_lora_weights(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        args.push_to_hub,
        args.hub_repo_id,
    )


if __name__ == "__main__":
    main()
