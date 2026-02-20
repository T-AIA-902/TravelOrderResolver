"""Pathfinding endpoints: station search and route calculation."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_graph
from src.api.utils import build_route_response

router = APIRouter(prefix="/pathfinding", tags=["pathfinding"])


@router.get("/stations")
def search_stations(q: str, limit: int = 10, graph=Depends(get_graph)):
    """Search graph nodes for stations matching *q* by name."""
    query = q.lower()
    exact, prefix, contains = [], [], []

    for uic, data in graph.graph.nodes(data=True):
        name: str = data.get("name", "")
        name_lower = name.lower()
        pos = data.get("pos", (0.0, 0.0))
        entry = {"name": name, "uic": str(uic), "lat": pos[0], "lon": pos[1]}

        if name_lower == query:
            exact.append(entry)
        elif name_lower.startswith(query):
            prefix.append(entry)
        elif query in name_lower:
            contains.append(entry)

    stations = (exact + prefix + contains)[:limit]
    return {"stations": stations}


class RouteRequest(BaseModel):
    departure: str
    destination: str
    intermediates: list[str] = []


@router.post("/route")
def compute_route(body: RouteRequest, graph=Depends(get_graph)):
    """Compute a route between two stations with optional intermediates."""
    simplified, error, full_uic_path = graph.get_path(
        body.departure,
        body.destination,
        body.intermediates or None,
        algorithm="astar",
    )
    return build_route_response(graph, simplified, error, full_uic_path)
