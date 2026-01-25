"""
Dataset preparation for Ministral entity extraction training.

Uses a generative approach: the model learns to output structured JSON
containing departure, destination, and intermediate stations.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


# Prompt template for entity extraction (Mistral format)
EXTRACTION_TEMPLATE = (
    "[INST] Extrait les entités de voyage du texte.\n"
    'JSON: {{"departure": "...", "destination": "...", "intermediate": [...]}}\n\n'
    "Texte: {sentence} [/INST]"
)

# Target format
TARGET_TEMPLATE = (
    '{{"departure": {departure}, "destination": {destination}, ' '"intermediate": {intermediate}}}'
)


def format_target(sample: Dict) -> str:
    """
    Format target JSON from sample.

    Args:
        sample: Dataset sample with departure, destination, intermediate

    Returns:
        Formatted JSON string
    """
    departure = f'"{sample["departure"]}"' if sample["departure"] else "null"
    destination = f'"{sample["destination"]}"' if sample["destination"] else "null"

    intermediate = sample.get("intermediate", "")
    if intermediate and isinstance(intermediate, str) and intermediate.strip():
        inter_list = [f'"{s.strip()}"' for s in intermediate.split(",") if s.strip()]
        intermediate_str = f"[{', '.join(inter_list)}]"
    else:
        intermediate_str = "[]"

    return TARGET_TEMPLATE.format(
        departure=departure,
        destination=destination,
        intermediate=intermediate_str,
    )


def load_raw_dataset(file_path: Path) -> List[Dict]:
    """Load raw dataset from JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def load_entity_dataset(
    file_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    num_samples: Optional[int] = None,
    trip_only: bool = True,
) -> Dataset:
    """
    Load and prepare entity extraction dataset for causal LM training.

    Args:
        file_path: Path to JSON dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_samples: Optional limit on samples
        trip_only: Only use TRIP samples (recommended)

    Returns:
        HuggingFace Dataset ready for training
    """
    data = load_raw_dataset(file_path)

    # Filter to TRIP samples only (they have entities)
    if trip_only:
        data = [s for s in data if s.get("intent") == "TRIP"]

    if num_samples:
        data = data[:num_samples]

    prompts = []
    targets = []

    for sample in data:
        prompt = EXTRACTION_TEMPLATE.format(sentence=sample["sentence"])
        target = format_target(sample)
        prompts.append(prompt)
        targets.append(target)

    def tokenize_function(examples: Dict) -> Dict:
        # Combine prompt + target for causal LM training
        full_texts = [
            p + t + tokenizer.eos_token for p, t in zip(examples["prompt"], examples["target"])
        ]

        model_inputs = tokenizer(
            full_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Create labels (mask prompt tokens with -100)
        labels = []
        for i, (prompt, input_ids) in enumerate(zip(examples["prompt"], model_inputs["input_ids"])):
            # Get prompt length in tokens
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_tokens)

            # Create label: -100 for prompt tokens, actual ids for target
            label = list(input_ids)
            for j in range(min(prompt_len, len(label))):
                label[j] = -100

            labels.append(label)

        model_inputs["labels"] = labels
        return model_inputs

    dataset = Dataset.from_dict({"prompt": prompts, "target": targets})
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "target"],
    )

    return tokenized


def load_entity_datasets(
    train_path: Path,
    val_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
) -> tuple:
    """
    Load train and validation datasets for entity extraction.

    Args:
        train_path: Path to training JSON
        val_path: Path to validation JSON
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_train_samples: Optional limit for training samples
        num_val_samples: Optional limit for validation samples

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = load_entity_dataset(train_path, tokenizer, max_length, num_train_samples)
    val_dataset = load_entity_dataset(val_path, tokenizer, max_length, num_val_samples)

    return train_dataset, val_dataset
