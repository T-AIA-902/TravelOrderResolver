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
    algorithm: str = "astar"


def _normalize_name(name: str) -> str:
    """Normalize model name for comparison: lowercase, strip hyphens/underscores/spaces."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


# Lazy-loaded Mistral instances (shared between intent and entity)
_mistral_entity = None
_mistral_intent = None


def _get_mistral_entity():
    global _mistral_entity
    if _mistral_entity is None:
        from src.nlp.entity import MistralEntityExtractor
        _mistral_entity = MistralEntityExtractor()
    return _mistral_entity


def _get_mistral_intent():
    global _mistral_intent
    if _mistral_intent is None:
        from src.nlp.intent import MistralIntentClassifier
        _mistral_intent = MistralIntentClassifier(ner_model=_get_mistral_entity())
    return _mistral_intent


def _find_component(name: str | None, components: list[tuple[str, object]], kind: str = "") -> tuple[str, object]:
    """Find a component by name (case-insensitive, ignoring hyphens) or return the first one.
    Lazy-loads Mistral if requested but not in the preloaded list."""
    if name:
        normalized = _normalize_name(name)
        for comp_name, comp in components:
            if _normalize_name(comp_name) == normalized:
                return comp_name, comp
        # Lazy-load models on demand
        if normalized == "mistrallora" or normalized == "mistral":
            if kind == "intent":
                return "Mistral-LoRA", _get_mistral_intent()
            else:
                return "Mistral-LoRA", _get_mistral_entity()
        if normalized == "mistralbase" or normalized == "mistral(base)":
            if kind == "intent":
                from src.nlp.intent.mistral_intent import MistralIntentClassifier
                return "Mistral (base)", MistralIntentClassifier(adapter_path=None)
            else:
                from src.nlp.entity.mistral_entity import MistralEntityExtractor
                return "Mistral (base)", MistralEntityExtractor(adapter_path=None)
        if normalized == "camembertbase" or normalized == "camembert(base)":
            if kind == "intent":
                from src.nlp.intent.camembert_base_intent import CamembertBaseIntentClassifier
                return "CamemBERT (base)", CamembertBaseIntentClassifier()
            else:
                from src.nlp.entity.camembert_base_entity import CamembertBaseEntityExtractor
                return "CamemBERT (base)", CamembertBaseEntityExtractor()
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
    classifier_name, classifier = _find_component(body.intent_model, intent_classifiers, kind="intent")
    extractor_name, extractor = _find_component(body.entity_model, entity_extractors, kind="entity")
    detector_name, detector = language_detectors[0]

    # Build pipeline — disable built-in station matching, use fuzzy_post instead
    config = PipelineConfig(use_station_matching=False)
    pipeline = NLPPipeline(
        config=config,
        language_detector=detector,  # type: ignore[arg-type]
        intent_classifier=classifier,  # type: ignore[arg-type]
        entity_extractor=extractor,  # type: ignore[arg-type]
    )

    with metrics_logger.start_request(body.text, f"{classifier_name}+{extractor_name}") as timer:
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
            algo = body.algorithm.lower()
            if algo not in ("dijkstra", "astar", "moastar"):
                algo = "astar"

            if algo == "moastar":
                from src.pathfinding.route_optimizer import Algorithm, RouteOptimizer
                optimizer = RouteOptimizer(graph)
                intermediate = intermediates_matched[0] if intermediates_matched else None
                moa_result = optimizer.find_route(
                    departure_matched, destination_matched,
                    intermediate=intermediate,
                    algorithm=Algorithm.MOASTAR,
                )
                pathfinding_response = build_route_response(graph, moa_result.simplified_path, moa_result.error, moa_result.full_uic_path)
                pathfinding_response["algorithm"] = algo
                if moa_result.pareto_paths:
                    pareto_routes = []
                    for p in moa_result.pareto_paths:
                        p_response = build_route_response(graph, graph._simplify_path(p.uic_path), None, p.uic_path)
                        pareto_routes.append({
                            "path": graph._simplify_path(p.uic_path),
                            "cost": {"time_h": round(p.cost.time, 2), "distance_km": round(p.cost.distance, 1), "transfers": p.cost.transfers},
                            "route_details": p_response["route_details"],
                        })
                    pathfinding_response["pareto_routes"] = pareto_routes
            else:
                simplified, error, full_uic_path = graph.get_path(
                    departure_matched,
                    destination_matched,
                    intermediates=intermediates_matched or None,
                    algorithm=algo,
                )
                pathfinding_response = build_route_response(graph, simplified, error, full_uic_path)
                pathfinding_response["algorithm"] = algo

        timer.set_output(result.to_dict())

    return {"nlp": nlp_response, "pathfinding": pathfinding_response}
