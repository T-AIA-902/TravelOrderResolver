"""NLP debug endpoints: run all models and compare results."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import (
    get_entity_extractors,
    get_fuzzy_post,
    get_intent_classifiers,
    get_language_detectors,
)

router = APIRouter()


class TextRequest(BaseModel):
    text: str


@router.post("/nlp/language")
def detect_language(
    body: TextRequest,
    language_detectors=Depends(get_language_detectors),
):
    results = []
    for name, detector in language_detectors:
        start = time.time()
        lang, conf = detector.detect(body.text)
        latency_ms = round((time.time() - start) * 1000, 2)
        results.append(
            {"model": name, "detected": lang, "confidence": conf, "latency_ms": latency_ms}
        )
    return {"results": results}


@router.post("/nlp/intent")
def classify_intent(
    body: TextRequest,
    intent_classifiers=Depends(get_intent_classifiers),
):
    results = []
    for name, classifier in intent_classifiers:
        start = time.time()
        intent, conf = classifier.classify(body.text)
        latency_ms = round((time.time() - start) * 1000, 2)
        results.append(
            {"model": name, "intent": intent, "confidence": conf, "latency_ms": latency_ms}
        )
    return {"results": results}


@router.post("/nlp/entities")
def extract_entities(
    body: TextRequest,
    entity_extractors=Depends(get_entity_extractors),
    fuzzy_post=Depends(get_fuzzy_post),
):
    def _build(raw_val: str | None, matched_val: str | None) -> dict:
        return {
            "raw": raw_val,
            "matched": matched_val,
            "confidence": 1.0 if raw_val else 0.0,
        }

    results = []
    for name, extractor in entity_extractors:
        start = time.time()
        entities = extractor.extract(body.text)
        matched = fuzzy_post.process(entities, body.text)
        latency_ms = round((time.time() - start) * 1000, 2)

        raw_intermediates = entities.get("intermediate", [])
        matched_intermediates = matched.get("intermediate", [])
        intermediates = [
            _build(raw, mtch) for raw, mtch in zip(raw_intermediates, matched_intermediates)
        ]

        results.append(
            {
                "model": name,
                "fuzzy": True,
                "departure": _build(entities.get("departure"), matched.get("departure")),
                "destination": _build(entities.get("destination"), matched.get("destination")),
                "intermediates": intermediates,
                "latency_ms": latency_ms,
            }
        )
    return {"results": results}
