"""Resolve endpoint: NLP pipeline + pathfinding in one call."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import (
    get_entity_extractors,
    get_graph,
    get_intent_classifiers,
    get_language_detectors,
    get_metrics_logger,
    get_station_db,
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
    station_db=Depends(get_station_db),
    metrics_logger=Depends(get_metrics_logger),
    intent_classifiers=Depends(get_intent_classifiers),
    entity_extractors=Depends(get_entity_extractors),
    language_detectors=Depends(get_language_detectors),
):
    # Select components
    classifier_name, classifier = _find_component(body.intent_model, intent_classifiers)
    extractor_name, extractor = _find_component(body.entity_model, entity_extractors)
    detector_name, detector = language_detectors[0]

    # Build pipeline
    config = PipelineConfig(use_station_matching=body.use_fuzzy)
    pipeline = NLPPipeline(
        config=config,
        station_db=station_db if body.use_fuzzy else None,
        language_detector=detector,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
        entity_extractor=extractor,  # type: ignore[arg-type]
    )

    # Run NLP
    start = time.time()
    result = pipeline.process(body.text)
    latency_ms = round((time.time() - start) * 1000, 2)

    # Build entity response helpers
    def _entity_for_role(role: str) -> dict | None:
        for e in result.entities:
            if e.role == role:
                matched = {
                    "DEPARTURE": result.departure,
                    "DESTINATION": result.destination,
                }.get(role)
                return {"raw": e.text, "matched": matched, "confidence": e.confidence}
        return None

    dep_entity = _entity_for_role("DEPARTURE")
    if not dep_entity and result.departure:
        dep_entity = {"raw": result.departure, "matched": result.departure, "confidence": 1.0}

    dest_entity = _entity_for_role("DESTINATION")
    if not dest_entity and result.destination:
        dest_entity = {"raw": result.destination, "matched": result.destination, "confidence": 1.0}

    intermediates = []
    intermediate_entities = [e for e in result.entities if e.role == "INTERMEDIATE"]
    if intermediate_entities:
        for idx, e in enumerate(intermediate_entities):
            matched = result.intermediates[idx] if idx < len(result.intermediates) else e.text
            intermediates.append({"raw": e.text, "matched": matched, "confidence": e.confidence})
    elif result.intermediates:
        intermediates = [{"raw": s, "matched": s, "confidence": 1.0} for s in result.intermediates]

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
        },
        "processed_text": result.processed_text,
        "latency_ms": latency_ms,
    }

    # Pathfinding
    pathfinding_response = None
    if result.intent.value == "TRIP" and result.departure and result.destination:
        simplified, error, full_uic_path = graph.get_path(
            result.departure,
            result.destination,
            intermediates=result.intermediates or None,
            algorithm="astar",
        )
        pathfinding_response = build_route_response(graph, simplified, error, full_uic_path)

    # Log metrics
    with metrics_logger.start_request(body.text, result.model_name) as timer:
        timer.set_output(result.to_dict())

    return {"nlp": nlp_response, "pathfinding": pathfinding_response}
