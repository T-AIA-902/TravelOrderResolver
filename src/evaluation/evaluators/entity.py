"""Entity extractor evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, overload

from ..data_loader import (
    is_lowercase_sample,
    is_misspelled_sample,
    normalize_fuzzy_station,
    normalize_location,
)
from ..metrics import EntityResults, update_entity_metrics
from ..progress import print_progress


@dataclass
class EntityPredictions:
    """Container for entity extraction predictions (for confusion matrices)."""

    departure_true: list[str]
    departure_pred: list[str]
    destination_true: list[str]
    destination_pred: list[str]


# Type alias for predictions dict
EntityPredictionsDict = dict[str, EntityPredictions]


@overload
def evaluate_entity_extractors(  # type: ignore[overload-overlap]
    extractors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    fuzzy_post: Any = ...,
    normalize_fuzzy: bool = ...,
    batch_size: int = ...,
    progress_callback: Callable[[int, int], None] | None = ...,
    return_predictions: Literal[False] = ...,
) -> dict[str, EntityResults]:
    """Evaluate entity extractors (return_predictions=False)."""
    ...


@overload
def evaluate_entity_extractors(
    extractors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    fuzzy_post: Any = ...,
    normalize_fuzzy: bool = ...,
    batch_size: int = ...,
    progress_callback: Callable[[int, int], None] | None = ...,
    return_predictions: Literal[True] = ...,
) -> tuple[dict[str, EntityResults], EntityPredictionsDict]:
    """Evaluate entity extractors (return_predictions=True)."""
    ...


def evaluate_entity_extractors(
    extractors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    fuzzy_post: Any = None,
    normalize_fuzzy: bool = False,
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
    return_predictions: bool = False,
) -> dict[str, EntityResults] | tuple[dict[str, EntityResults], EntityPredictionsDict]:
    """
    Evaluate all entity extractors.

    Args:
        extractors: List of (name, extractor) tuples
        data: List of samples with 'sentence', 'intent', 'departure', 'destination' keys
        fuzzy_post: Optional fuzzy post-processor
        normalize_fuzzy: Whether to normalize station names for comparison
        batch_size: Batch size for batched inference
        progress_callback: Optional callback for progress updates
        return_predictions: If True, also return predictions for confusion matrices

    Returns:
        If return_predictions=False: Dictionary mapping extractor names to EntityResults
        If return_predictions=True: Tuple of (results_dict, predictions_dict)
            where predictions_dict[name] contains EntityPredictions with
            departure_true, departure_pred, destination_true, destination_pred
    """
    results: dict[str, EntityResults] = {}
    all_predictions: EntityPredictionsDict = {}
    callback = progress_callback or print_progress

    # Filter TRIP samples
    trip_samples = [s for s in data if s["intent"] == "TRIP"]
    sentences = [s["sentence"] for s in trip_samples]

    for name, extractor in extractors:
        suffix = " + Fuzzy" if fuzzy_post else ""
        display_name = f"{name}{suffix}"
        print(f"  Evaluating entity: {display_name}...")

        r = EntityResults(name=display_name)
        preds = EntityPredictions(
            departure_true=[],
            departure_pred=[],
            destination_true=[],
            destination_pred=[],
        )

        if hasattr(extractor, "extract_batch"):
            print(f"    Using batched inference (batch_size={batch_size})...")
            start = time.perf_counter()
            all_entities = extractor.extract_batch(
                sentences,
                batch_size=batch_size,
                progress_callback=callback,
            )
            extraction_time_ms = (time.perf_counter() - start) * 1000

            # Time fuzzy post-processing if enabled
            fuzzy_time_ms = 0.0
            if fuzzy_post:
                start = time.perf_counter()
                all_entities = [fuzzy_post.process(e, s) for e, s in zip(all_entities, sentences)]
                fuzzy_time_ms = (time.perf_counter() - start) * 1000

            total_time_ms = extraction_time_ms + fuzzy_time_ms
            avg_latency = total_time_ms / len(sentences) if sentences else 0

            for sample, entities in zip(trip_samples, all_entities):
                # Pass None for fuzzy_post since already applied above
                _process_sample(r, sample, entities, None, normalize_fuzzy, avg_latency, preds)
        else:
            # Sequential extraction
            total = len(trip_samples)
            for i, sample in enumerate(trip_samples):
                if i % batch_size == 0:
                    callback(i, total)

                sentence = sample["sentence"]

                start = time.perf_counter()
                entities = extractor.extract(sentence)
                if fuzzy_post:
                    entities = fuzzy_post.process(entities, sentence)
                end = time.perf_counter()

                latency = (end - start) * 1000
                _process_sample(r, sample, entities, None, normalize_fuzzy, latency, preds)

            callback(total, total)

        results[display_name] = r
        all_predictions[display_name] = preds
        print(f"    Done. Accuracy: {r.accuracy*100:.1f}%")

    if return_predictions:
        return results, all_predictions
    return results


def _process_sample(
    r: EntityResults,
    sample: dict[str, Any],
    entities: dict[str, Any],
    fuzzy_post: Any,
    normalize_fuzzy: bool,
    latency: float,
    preds: EntityPredictions | None = None,
) -> None:
    """Process a single sample and update results."""
    sentence = sample["sentence"]
    orig_true_dep = normalize_location(sample.get("departure"))
    orig_true_dest = normalize_location(sample.get("destination"))
    true_dep = orig_true_dep
    true_dest = orig_true_dest

    # Apply fuzzy post-processing if provided
    if fuzzy_post:
        entities = fuzzy_post.process(entities, sentence)

    r.latencies.append(latency)

    # Get predictions
    pred_dep = normalize_location(entities.get("departure"))
    pred_dest = normalize_location(entities.get("destination"))

    # Normalize for fuzzy comparison
    if normalize_fuzzy:
        pred_dep = normalize_fuzzy_station(pred_dep)
        pred_dest = normalize_fuzzy_station(pred_dest)
        true_dep = normalize_fuzzy_station(true_dep)
        true_dest = normalize_fuzzy_station(true_dest)

    # Store predictions for confusion matrix
    if preds is not None:
        preds.departure_true.append(true_dep or "NONE")
        preds.departure_pred.append(pred_dep or "NONE")
        preds.destination_true.append(true_dest or "NONE")
        preds.destination_pred.append(pred_dest or "NONE")

    r.total += 1

    # Update metrics
    dep_correct = update_entity_metrics(r.departure_metrics, pred_dep, true_dep)
    dest_correct = update_entity_metrics(r.destination_metrics, pred_dest, true_dest)

    both_correct = dep_correct and dest_correct
    if both_correct:
        r.correct += 1

    # Category metrics
    if is_misspelled_sample(sentence, orig_true_dep or "", orig_true_dest or ""):
        r.misspelling_total += 1
        if both_correct:
            r.misspelling_correct += 1

    if is_lowercase_sample(sentence):
        r.lowercase_total += 1
        if both_correct:
            r.lowercase_correct += 1
