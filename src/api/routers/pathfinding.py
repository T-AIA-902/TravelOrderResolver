"""Pathfinding endpoints: station search and route calculation."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_graph
from src.api.utils import build_route_response
from src.pathfinding.route_optimizer import Algorithm, RouteOptimizer

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
    algorithm: str = "astar"


@router.post("/route")
def compute_route(body: RouteRequest, graph=Depends(get_graph)):
    """Compute a route between two stations with optional intermediates."""
    algo = body.algorithm.lower()
    if algo not in ("dijkstra", "astar", "moastar"):
        algo = "astar"

    if algo == "moastar":
        optimizer = RouteOptimizer(graph)
        intermediate = body.intermediates[0] if body.intermediates else None
        result = optimizer.find_route(
            body.departure, body.destination,
            intermediate=intermediate,
            algorithm=Algorithm.MOASTAR,
        )
        response = build_route_response(graph, result.simplified_path, result.error, result.full_uic_path)
        response["algorithm"] = algo
        if result.pareto_paths:
            pareto_routes = []
            for p in result.pareto_paths:
                p_response = build_route_response(graph, graph._simplify_path(p.uic_path), None, p.uic_path)
                pareto_routes.append({
                    "path": graph._simplify_path(p.uic_path),
                    "cost": {"time_h": round(p.cost.time, 2), "distance_km": round(p.cost.distance, 1), "transfers": p.cost.transfers},
                    "route_details": p_response["route_details"],
                })
            response["pareto_routes"] = pareto_routes
        return response

    simplified, error, full_uic_path = graph.get_path(
        body.departure,
        body.destination,
        body.intermediates or None,
        algorithm=algo,
    )
    response = build_route_response(graph, simplified, error, full_uic_path)
    response["algorithm"] = algo
    return response
