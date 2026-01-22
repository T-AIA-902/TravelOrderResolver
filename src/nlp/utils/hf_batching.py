"""HuggingFace batching utilities for efficient GPU pipeline processing."""

from typing import Any, List


def texts_to_dataset(texts: List[str]) -> Any:
    """
    Convert list of texts to HuggingFace Dataset.

    Args:
        texts: List of input texts

    Returns:
        HuggingFace Dataset with "text" column
    """
    from datasets import Dataset  # type: ignore[attr-defined]

    return Dataset.from_dict({"text": texts})


def run_pipeline_batched(
    pipeline: Any,
    texts: List[str],
    batch_size: int = 32,
    **kwargs: Any,
) -> List[Any]:
    """
    Run HuggingFace pipeline with Dataset for optimal GPU throughput.

    Using a Dataset instead of a list enables HuggingFace to use efficient
    DataLoader-based batching with proper GPU memory prefetching.

    Args:
        pipeline: HuggingFace pipeline instance
        texts: Input texts to process
        batch_size: Batch size for processing (default: 32)
        **kwargs: Additional pipeline arguments (e.g., candidate_labels)

    Returns:
        List of pipeline outputs in input order
    """
    if not texts:
        return []

    dataset = texts_to_dataset(texts)
    results = list(pipeline(dataset["text"], batch_size=batch_size, **kwargs))
    return results
