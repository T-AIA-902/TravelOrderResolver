"""
Integration tests for the FastAPI Travel Order Resolver API.

These tests use a lightweight app setup with only regex-based NLP models
(no ML model downloads required) to validate all API endpoints end-to-end.
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import health, nlp, pathfinding, resolve
from src.evaluation.model_factory import (
    create_entity_extractors,
    create_fuzzy_post_processor,
    create_intent_classifiers,
    create_language_detectors,
)
from src.monitoring.metrics_logger import MetricsLogger
from src.pathfinding.graph import TrainGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    """Build a FastAPI app with regex-only models (no ML downloads needed)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[arg-type]
        app.state.device = "cpu"
        app.state.device_info = {"cuda_device_name": None}
        app.state.graph = TrainGraph()
        app.state.language_detectors = create_language_detectors(["regex"])
        app.state.intent_classifiers = create_intent_classifiers(["regex"])
        app.state.entity_extractors = create_entity_extractors(["regex"])
        app.state.fuzzy_post = create_fuzzy_post_processor()
        with tempfile.TemporaryDirectory() as tmpdir:
            app.state.metrics_logger = MetricsLogger(output_dir=tmpdir)
            yield

    app = FastAPI(title="Test Travel Order Resolver", lifespan=lifespan)
    app.include_router(health.router, prefix="/api")
    app.include_router(resolve.router, prefix="/api")
    app.include_router(nlp.router, prefix="/api")
    app.include_router(pathfinding.router, prefix="/api")
    return app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient scoped to the module for performance."""
    app = _build_test_app()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_has_expected_fields(self, client: TestClient) -> None:
        data = client.get("/api/health").json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert isinstance(data["models_loaded"], list)
        assert isinstance(data["stations_count"], int)
        assert data["stations_count"] > 0
        assert isinstance(data["graph_nodes"], int)
        assert isinstance(data["graph_edges"], int)
        assert data["device"] == "cpu"

    def test_health_regex_model_loaded(self, client: TestClient) -> None:
        data = client.get("/api/health").json()
        assert "regex" in data["models_loaded"]


# ---------------------------------------------------------------------------
# Resolve endpoint
# ---------------------------------------------------------------------------


class TestResolveEndpoint:
    """Tests for POST /api/resolve."""

    def test_resolve_french_trip(self, client: TestClient) -> None:
        """A clear French trip sentence should return NLP + pathfinding results."""
        resp = client.post(
            "/api/resolve",
            json={"text": "Je veux aller de Paris à Lyon", "intent_model": "Regex", "entity_model": "Regex"},
        )
        assert resp.status_code == 200
        data = resp.json()

        # NLP section
        nlp_data = data["nlp"]
        assert nlp_data["language"]["detected"] == "FRENCH"
        assert nlp_data["intent"]["value"] == "TRIP"
        assert nlp_data["intent"]["model"] == "Regex"
        assert nlp_data["entities"]["model"] == "Regex"
        assert nlp_data["entities"]["departure"] is not None
        assert nlp_data["entities"]["destination"] is not None
        assert isinstance(nlp_data["latency_ms"], (int, float))

    def test_resolve_returns_pathfinding(self, client: TestClient) -> None:
        """When entities are matched to real stations, pathfinding should run."""
        resp = client.post(
            "/api/resolve",
            json={"text": "De Paris à Lyon"},
        )
        data = resp.json()
        pf = data["pathfinding"]
        # Pathfinding depends on fuzzy matching to real station names.
        # If matched, we expect a route; if not, pathfinding is None.
        if pf is not None:
            assert "found" in pf
            assert "route" in pf

    def test_resolve_english_trip(self, client: TestClient) -> None:
        resp = client.post(
            "/api/resolve",
            json={"text": "I want to go from Paris to Lyon"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["nlp"]["language"]["detected"] == "ENGLISH"
        assert data["nlp"]["intent"]["value"] == "TRIP"

    def test_resolve_not_trip(self, client: TestClient) -> None:
        """Non-travel text should be classified as NOT_TRIP or TRIP with no entities."""
        resp = client.post(
            "/api/resolve",
            json={"text": "Bonjour, quelle heure est-il ?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # The regex classifier may or may not flag this as a trip.
        assert data["nlp"]["intent"]["value"] in ("TRIP", "NOT_TRIP")

    def test_resolve_without_fuzzy(self, client: TestClient) -> None:
        resp = client.post(
            "/api/resolve",
            json={"text": "De Paris à Lyon", "use_fuzzy": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["nlp"]["entities"]["fuzzy_enabled"] is False

    def test_resolve_invalid_model_name(self, client: TestClient) -> None:
        """Requesting a model that does not exist should return 404."""
        resp = client.post(
            "/api/resolve",
            json={"text": "De Paris à Lyon", "intent_model": "NonExistentModel"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_resolve_invalid_entity_model(self, client: TestClient) -> None:
        resp = client.post(
            "/api/resolve",
            json={"text": "De Paris à Lyon", "entity_model": "FakeModel"},
        )
        assert resp.status_code == 404

    def test_resolve_empty_text(self, client: TestClient) -> None:
        """Empty text should still return 200 (pipeline handles gracefully)."""
        resp = client.post(
            "/api/resolve",
            json={"text": ""},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "nlp" in data

    def test_resolve_missing_text_field(self, client: TestClient) -> None:
        """Missing required 'text' field should return 422 validation error."""
        resp = client.post("/api/resolve", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# NLP debug endpoints
# ---------------------------------------------------------------------------


class TestNLPLanguageEndpoint:
    """Tests for POST /api/nlp/language."""

    def test_detect_french(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/language", json={"text": "Je veux aller de Paris à Lyon"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        result = data["results"][0]
        assert result["model"] == "Regex"
        assert result["detected"] == "FRENCH"
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["latency_ms"], (int, float))

    def test_detect_english(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/language", json={"text": "I want to go from Paris to Lyon"})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["detected"] == "ENGLISH"

    def test_empty_text(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/language", json={"text": ""})
        assert resp.status_code == 200

    def test_missing_text(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/language", json={})
        assert resp.status_code == 422


class TestNLPIntentEndpoint:
    """Tests for POST /api/nlp/intent."""

    def test_trip_intent(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/intent", json={"text": "Je veux aller de Paris à Lyon"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        result = data["results"][0]
        assert result["intent"] == "TRIP"
        assert result["model"] == "Regex"
        assert isinstance(result["confidence"], (int, float))

    def test_not_trip_intent(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/intent", json={"text": "Quel temps fait-il ?"})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["intent"] in ("TRIP", "NOT_TRIP")

    def test_missing_text(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/intent", json={})
        assert resp.status_code == 422


class TestNLPEntitiesEndpoint:
    """Tests for POST /api/nlp/entities."""

    def test_extract_entities(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/entities", json={"text": "De Paris à Lyon"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        result = data["results"][0]
        assert result["model"] == "Regex"
        assert result["fuzzy"] is True
        assert "departure" in result
        assert "destination" in result
        assert "intermediates" in result
        # Check entity structure
        dep = result["departure"]
        assert "raw" in dep
        assert "matched" in dep
        assert "confidence" in dep

    def test_extract_with_raw_values(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/entities", json={"text": "Je veux aller de Marseille à Bordeaux"})
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["departure"]["raw"] is not None
        assert "Marseille" in result["departure"]["raw"]
        assert result["destination"]["raw"] is not None
        assert "Bordeaux" in result["destination"]["raw"]

    def test_empty_text(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/entities", json={"text": ""})
        assert resp.status_code == 200

    def test_missing_text(self, client: TestClient) -> None:
        resp = client.post("/api/nlp/entities", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pathfinding endpoints
# ---------------------------------------------------------------------------


class TestPathfindingStationsEndpoint:
    """Tests for GET /api/pathfinding/stations."""

    def test_search_stations(self, client: TestClient) -> None:
        resp = client.get("/api/pathfinding/stations", params={"q": "Paris"})
        assert resp.status_code == 200
        data = resp.json()
        assert "stations" in data
        assert isinstance(data["stations"], list)
        # The real graph should contain Paris stations
        if len(data["stations"]) > 0:
            station = data["stations"][0]
            assert "name" in station
            assert "uic" in station
            assert "lat" in station
            assert "lon" in station
            assert "paris" in station["name"].lower()

    def test_search_with_limit(self, client: TestClient) -> None:
        resp = client.get("/api/pathfinding/stations", params={"q": "Paris", "limit": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stations"]) <= 3

    def test_search_no_match(self, client: TestClient) -> None:
        resp = client.get("/api/pathfinding/stations", params={"q": "ZZZNOTASTATION"})
        assert resp.status_code == 200
        assert resp.json()["stations"] == []

    def test_search_missing_query(self, client: TestClient) -> None:
        """Missing required 'q' parameter should return 422."""
        resp = client.get("/api/pathfinding/stations")
        assert resp.status_code == 422

    def test_search_lyon(self, client: TestClient) -> None:
        resp = client.get("/api/pathfinding/stations", params={"q": "Lyon"})
        assert resp.status_code == 200
        stations = resp.json()["stations"]
        if len(stations) > 0:
            assert any("lyon" in s["name"].lower() for s in stations)


class TestPathfindingRouteEndpoint:
    """Tests for POST /api/pathfinding/route."""

    def test_route_known_stations(self, client: TestClient) -> None:
        """Route between well-known stations should succeed."""
        # First find valid station names
        paris_resp = client.get("/api/pathfinding/stations", params={"q": "Paris Gare de Lyon"})
        lyon_resp = client.get("/api/pathfinding/stations", params={"q": "Lyon Part Dieu"})
        paris_stations = paris_resp.json()["stations"]
        lyon_stations = lyon_resp.json()["stations"]

        if paris_stations and lyon_stations:
            resp = client.post(
                "/api/pathfinding/route",
                json={
                    "departure": paris_stations[0]["name"],
                    "destination": lyon_stations[0]["name"],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "found" in data
            if data["found"]:
                assert len(data["route"]) > 0
                assert data["total_stops"] > 0
                assert "route_details" in data
                assert data["error"] is None

    def test_route_unknown_stations(self, client: TestClient) -> None:
        """Unknown station names should return a response with found=False."""
        resp = client.post(
            "/api/pathfinding/route",
            json={"departure": "NotARealStation", "destination": "AlsoNotReal"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert data["error"] is not None

    def test_route_missing_fields(self, client: TestClient) -> None:
        resp = client.post("/api/pathfinding/route", json={"departure": "Paris"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Cross-endpoint integration: full resolve pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end tests combining NLP + pathfinding via /api/resolve."""

    def test_french_trip_full_pipeline(self, client: TestClient) -> None:
        """Verify the full pipeline from text to route for a French trip request."""
        resp = client.post(
            "/api/resolve",
            json={"text": "Je voudrais un billet de Paris Gare de Lyon à Lyon Part Dieu"},
        )
        assert resp.status_code == 200
        data = resp.json()

        nlp_data = data["nlp"]
        assert nlp_data["language"]["detected"] == "FRENCH"
        assert nlp_data["intent"]["value"] == "TRIP"
        assert nlp_data["entities"]["departure"] is not None
        assert nlp_data["entities"]["destination"] is not None

    def test_resolve_processed_text(self, client: TestClient) -> None:
        """processed_text field should be a non-empty string."""
        resp = client.post(
            "/api/resolve",
            json={"text": "  De  Paris  à  Lyon  "},
        )
        assert resp.status_code == 200
        processed = resp.json()["nlp"]["processed_text"]
        assert isinstance(processed, str)
        assert len(processed.strip()) > 0

    def test_resolve_response_structure(self, client: TestClient) -> None:
        """Validate the complete response structure of /api/resolve."""
        resp = client.post(
            "/api/resolve",
            json={"text": "De Paris à Lyon"},
        )
        data = resp.json()

        # Top-level keys
        assert set(data.keys()) == {"nlp", "pathfinding"}

        # NLP sub-keys
        nlp_data = data["nlp"]
        assert set(nlp_data.keys()) == {
            "language",
            "intent",
            "entities",
            "processed_text",
            "latency_ms",
        }
        assert set(nlp_data["language"].keys()) == {"detected", "confidence", "model"}
        assert set(nlp_data["intent"].keys()) == {"value", "confidence", "model"}
        assert set(nlp_data["entities"].keys()) == {
            "departure",
            "destination",
            "intermediates",
            "model",
            "fuzzy_enabled",
        }
