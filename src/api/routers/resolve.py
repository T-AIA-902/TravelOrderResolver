"""Resolve endpoint: NLP pipeline + pathfinding in one call."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import (
    get_entity_extractors,
    get_fuzzy_post,
    get_graph,
    get_intent_classifiers,
    get_language_detectors,
    get_metrics_logger,
)
from src.api.utils import build_route_response
from src.nlp.pipeline import NLPPipeline, PipelineConfig

router = APIRouter()


class ResolveRequest(BaseModel):
    text: str
    intent_model: str | None = None
    entity_model: str | None = None
    use_fuzzy: bool = True


def _find_component(name: str | None, components: list[tuple[str, object]]) -> tuple[str, object]:
    """Find a component by name (case-insensitive) or return the first one."""
    if name:
        lower = name.lower()
        for comp_name, comp in components:
            if comp_name.lower() == lower:
                return comp_name, comp
        available = [n for n, _ in components]
        raise HTTPException(404, f"Model '{name}' not found. Available: {available}")
    return components[0]


@router.post("/resolve")
def resolve(
    body: ResolveRequest,
    graph=Depends(get_graph),
    fuzzy_post=Depends(get_fuzzy_post),
    metrics_logger=Depends(get_metrics_logger),
    intent_classifiers=Depends(get_intent_classifiers),
    entity_extractors=Depends(get_entity_extractors),
    language_detectors=Depends(get_language_detectors),
):
    # Select components
    classifier_name, classifier = _find_component(body.intent_model, intent_classifiers)
    extractor_name, extractor = _find_component(body.entity_model, entity_extractors)
    detector_name, detector = language_detectors[0]

    # Build pipeline — disable built-in station matching, use fuzzy_post instead
    config = PipelineConfig(use_station_matching=False)
    pipeline = NLPPipeline(
        config=config,
        language_detector=detector,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
        entity_extractor=extractor,  # type: ignore[arg-type]
    )

    # Run NLP
    start = time.time()
    result = pipeline.process(body.text)
    latency_ms = round((time.time() - start) * 1000, 2)

    # Fuzzy match raw entities to real station names
    raw_entities = {
        "departure": result.departure or None,
        "destination": result.destination or None,
        "intermediate": result.intermediates or [],
    }
    if body.use_fuzzy:
        matched = fuzzy_post.process(raw_entities, body.text)
    else:
        matched = raw_entities

    departure_matched = str(matched["departure"]) if matched.get("departure") else None
    destination_matched = str(matched["destination"]) if matched.get("destination") else None
    raw_intermediates = matched.get("intermediate", [])
    intermediates_matched: list[str] = list(raw_intermediates)  # type: ignore[arg-type]

    def _build_entity(raw: str | None, mtch: str | None) -> dict | None:
        if not raw:
            return None
        return {"raw": raw, "matched": mtch, "confidence": 1.0 if mtch else 0.0}

    raw_dep = str(raw_entities["departure"]) if raw_entities.get("departure") else None
    raw_dest = str(raw_entities["destination"]) if raw_entities.get("destination") else None
    raw_inter: list[str] = list(raw_entities.get("intermediate", []))  # type: ignore[arg-type]

    dep_entity = _build_entity(raw_dep, departure_matched)
    dest_entity = _build_entity(raw_dest, destination_matched)
    intermediates = [_build_entity(r, m) for r, m in zip(raw_inter, intermediates_matched) if r]

    nlp_response = {
        "language": {
            "detected": result.language.value,
            "confidence": result.language_confidence,
            "model": detector_name,
        },
        "intent": {
            "value": result.intent.value,
            "confidence": result.intent_confidence,
            "model": classifier_name,
        },
        "entities": {
            "departure": dep_entity,
            "destination": dest_entity,
            "intermediates": intermediates,
            "model": extractor_name,
            "fuzzy_enabled": body.use_fuzzy,
        },
        "processed_text": result.processed_text,
        "latency_ms": latency_ms,
    }

    # Pathfinding — only if intent is TRIP or UNKNOWN (benefit of the doubt)
    pathfinding_response = None
    intent_ok = result.intent.value != "NOT_TRIP"
    if intent_ok and departure_matched and destination_matched:
        simplified, error, full_uic_path = graph.get_path(
            departure_matched,
            destination_matched,
            intermediates=intermediates_matched or None,
            algorithm="astar",
        )
        pathfinding_response = build_route_response(graph, simplified, error, full_uic_path)

    # Log metrics
    with metrics_logger.start_request(body.text, result.model_name) as timer:
        timer.set_output(result.to_dict())

    return {"nlp": nlp_response, "pathfinding": pathfinding_response}
