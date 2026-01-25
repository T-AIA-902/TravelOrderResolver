"""
Dataset preparation for Ministral intent classification training.

Loads existing datasets from datasets/base/ or datasets/augmented/
and prepares them for sequence classification fine-tuning.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizer


# Label mapping for intent classification
LABEL_MAP = {"TRIP": 0, "NOT_TRIP": 1, "UNKNOWN": 2}
LABEL_NAMES = ["TRIP", "NOT_TRIP", "UNKNOWN"]


def load_raw_dataset(file_path: Path) -> List[Dict]:
    """
    Load raw dataset from JSON file.

    Args:
        file_path: Path to JSON dataset file

    Returns:
        List of sample dictionaries
    """
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def load_intent_dataset(
    file_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    num_samples: Optional[int] = None,
) -> Dataset:
    """
    Load and tokenize intent classification dataset.

    Args:
        file_path: Path to JSON dataset (e.g., datasets/base/train.json)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_samples: Optional limit on number of samples (for testing)

    Returns:
        HuggingFace Dataset ready for training
    """
    data = load_raw_dataset(file_path)

    if num_samples:
        data = data[:num_samples]

    texts = [sample["sentence"] for sample in data]
    labels = [LABEL_MAP.get(sample["intent"], 2) for sample in data]

    def tokenize_function(examples: Dict) -> Dict:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.remove_columns(["text"])

    return tokenized


def load_intent_datasets(
    train_path: Path,
    val_path: Path,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
) -> tuple:
    """
    Load train and validation datasets.

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
    train_dataset = load_intent_dataset(train_path, tokenizer, max_length, num_train_samples)
    val_dataset = load_intent_dataset(val_path, tokenizer, max_length, num_val_samples)

    return train_dataset, val_dataset


def get_label_names() -> List[str]:
    """Return list of label names in order."""
    return LABEL_NAMES


def get_num_labels() -> int:
    """Return number of labels."""
    return len(LABEL_MAP)
