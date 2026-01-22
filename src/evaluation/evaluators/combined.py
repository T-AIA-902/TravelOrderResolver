"""
Combined pipeline evaluation (intent + entity).
"""

import time
from typing import Any, Callable

from ..data_loader import normalize_fuzzy_station, normalize_location
from ..metrics import CombinedResults
from ..progress import print_progress


def evaluate_combined(
    intent_classifiers: list[tuple[str, Any]],
    entity_extractors: list[tuple[str, Any]],
    data: list[dict[str, Any]],
    fuzzy_post: Any = None,
    normalize_fuzzy: bool = False,
    batch_size: int = 128,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[CombinedResults]:
    """
    Evaluate all combinations of intent classifiers and entity extractors.

    Args:
        intent_classifiers: List of (name, classifier) tuples
        entity_extractors: List of (name, extractor) tuples
        data: List of samples
        fuzzy_post: Optional fuzzy post-processor
        normalize_fuzzy: Whether to normalize station names for comparison
        batch_size: Batch size for progress reporting
        progress_callback: Optional callback for progress updates

    Returns:
        List of CombinedResults for each combination
    """
    results: list[CombinedResults] = []
    callback = progress_callback or print_progress

    for intent_name, intent_clf in intent_classifiers:
        for entity_name, entity_ext in entity_extractors:
            # Clear fuzzy cache for fair comparison between pipeline combinations
            if fuzzy_post and hasattr(fuzzy_post, "matcher"):
                fuzzy_post.matcher.clear_cache()

            suffix = " + Fuzzy" if fuzzy_post else ""
            print(f"  Evaluating combined: {intent_name} + {entity_name}{suffix}...")

            r = CombinedResults(
                intent_name=intent_name,
                entity_name=entity_name,
                with_fuzzy=fuzzy_post is not None,
            )

            total = len(data)
            for i, sample in enumerate(data):
                if i % batch_size == 0:
                    callback(i, total)

                sentence = sample["sentence"]
                true_intent = sample["intent"]

                # Get ground truth
                orig_true_dep = normalize_location(sample.get("departure"))
                orig_true_dest = normalize_location(sample.get("destination"))
                true_dep = orig_true_dep
                true_dest = orig_true_dest

                start = time.perf_counter()

                # Classify intent
                pred_intent, _ = intent_clf.classify(sentence)

                # Extract entities
                entities = entity_ext.extract(sentence)
                if fuzzy_post:
                    entities = fuzzy_post.process(entities, sentence)

                end = time.perf_counter()
                r.latencies.append((end - start) * 1000)

                # Intent evaluation
                r.intent_total += 1
                if pred_intent == true_intent:
                    r.intent_correct += 1

                # Entity evaluation (only on TRIP samples)
                if true_intent == "TRIP":
                    pred_dep = normalize_location(entities.get("departure"))
                    pred_dest = normalize_location(entities.get("destination"))

                    if normalize_fuzzy:
                        pred_dep = normalize_fuzzy_station(pred_dep)
                        pred_dest = normalize_fuzzy_station(pred_dest)
                        true_dep = normalize_fuzzy_station(true_dep)
                        true_dest = normalize_fuzzy_station(true_dest)

                    r.entity_total += 1
                    if pred_dep == true_dep and pred_dest == true_dest:
                        r.entity_correct += 1

            callback(total, total)
            results.append(r)
            print(
                f"    Done. Intent: {r.intent_accuracy*100:.1f}%, "
                f"Entity: {r.entity_accuracy*100:.1f}%"
            )

    return results
