"""Reusable preprocessing functions for evaluation.

This module provides preprocessing functions that can be imported by:
- CLI (src/evaluation/cli.py)
- Notebooks (notebooks/evaluation.ipynb)
- Dataset scripts (datasets/scripts/)
"""

import copy

from src.nlp.pre import Preprocessor, PreprocessorConfig, STTArtifactFilter


def apply_preprocessing(
    data: list[dict], verbose: bool = True, inplace: bool = False
) -> list[dict]:
    """Apply STT filter + text normalization to dataset.

    Args:
        data: List of samples with "sentence" key
        verbose: Print progress information
        inplace: If True, modify data in place. If False, return a copy.

    Returns:
        List with preprocessed sentences (new list if inplace=False)
    """
    stt_filter = STTArtifactFilter()
    preprocessor = Preprocessor(
        PreprocessorConfig(
            lowercase=False,  # Preserve case for language detection
            remove_accents=False,  # Keep accents for French
            normalize_whitespace=True,
            remove_punctuation=False,
            normalize_apostrophes=True,
            normalize_hyphens=True,
            strip_extra_spaces=True,
        )
    )

    if inplace:
        processed = data
    else:
        processed = copy.deepcopy(data)

    noise_count = 0
    for sample in processed:
        # Step 1: Clean STT artifacts
        cleaned, is_noise = stt_filter.filter(sample["sentence"])
        if is_noise:
            noise_count += 1

        # Step 2: Normalize text (unicode, apostrophes, hyphens, whitespace)
        sample["sentence"] = preprocessor.preprocess(cleaned)

    if verbose:
        print(f"Pre-processed {len(processed)} samples ({noise_count} pure noise)")

    return processed
