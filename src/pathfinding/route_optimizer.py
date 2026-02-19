from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import networkx as nx

from .algorithms.astar import astar_path
from .algorithms.moa_star import CostVector, PathResult, moastar_paths
from .graph import TrainGraph


class Algorithm(str, Enum):
    DIJKSTRA = "dijkstra"
    ASTAR = "astar"
    MOASTAR = "moastar"


@dataclass
class RouteResult:
    simplified_path: Optional[List[str]] = None
    full_uic_path: Optional[List[str]] = None
    error: Optional[str] = None
    pareto_paths: Optional[List[PathResult]] = None

    @property
    def success(self) -> bool:
        return self.error is None and (
            self.full_uic_path is not None or self.pareto_paths is not None
        )


class RouteOptimizer:
    def __init__(self, train_graph: TrainGraph) -> None:
        self.tg = train_graph

    def find_route(
        self,
        departure: str,
        destination: str,
        intermediate: Optional[str] = None,
        algorithm: Algorithm = Algorithm.ASTAR,
    ) -> RouteResult:
        dep_uic = self.tg._find_uic_by_name(departure)
        dest_uic = self.tg._find_uic_by_name(destination)

        if not dep_uic:
            return RouteResult(error=f"Departure not found: {departure}")
        if not dest_uic:
            return RouteResult(error=f"Destination not found: {destination}")

        if intermediate:
            mid_uic = self.tg._find_uic_by_name(intermediate)
            if not mid_uic:
                return RouteResult(
                    error=f"Intermediate city not found: {intermediate}"
                )
            return self._route_via_waypoint(dep_uic, mid_uic, dest_uic, algorithm)

        return self._compute_route(dep_uic, dest_uic, algorithm)

    def _route_via_waypoint(
        self,
        dep_uic: str,
        mid_uic: str,
        dest_uic: str,
        algorithm: Algorithm,
    ) -> RouteResult:
        leg1 = self._compute_route(dep_uic, mid_uic, algorithm)
        if leg1.error:
            return RouteResult(error=f"Leg 1 failed: {leg1.error}")

        leg2 = self._compute_route(mid_uic, dest_uic, algorithm)
        if leg2.error:
            return RouteResult(error=f"Leg 2 failed: {leg2.error}")

        if algorithm == Algorithm.MOASTAR:
            if not leg1.pareto_paths or not leg2.pareto_paths:
                return RouteResult(error="No Pareto paths found for one leg")

            combined = []
            for p1 in leg1.pareto_paths:
                for p2 in leg2.pareto_paths:
                    merged_uics = p1.uic_path + p2.uic_path[1:]
                    merged_cost = p1.cost + p2.cost
                    combined.append(PathResult(uic_path=merged_uics, cost=merged_cost))

            combined.sort(key=lambda p: (p.cost.time, p.cost.distance, p.cost.transfers))

            pareto: List[PathResult] = []
            for candidate in combined:
                if not any(existing.cost.dominates(candidate.cost) for existing in pareto):
                    pareto.append(candidate)
                if len(pareto) >= 3:
                    break

            best = pareto[0]
            simplified = self.tg._simplify_path(best.uic_path)
            return RouteResult(
                simplified_path=simplified,
                full_uic_path=best.uic_path,
                pareto_paths=pareto,
            )

        path1 = leg1.full_uic_path or []
        path2 = leg2.full_uic_path or []
        combined_uics = path1 + path2[1:]

        simplified = self.tg._simplify_path(combined_uics)
        return RouteResult(
            simplified_path=simplified, full_uic_path=combined_uics
        )

    def _compute_route(
        self, start_uic: str, goal_uic: str, algorithm: Algorithm
    ) -> RouteResult:
        graph = self.tg.graph

        if algorithm == Algorithm.DIJKSTRA:
            return self._run_dijkstra(graph, start_uic, goal_uic)
        elif algorithm == Algorithm.ASTAR:
            return self._run_astar(graph, start_uic, goal_uic)
        elif algorithm == Algorithm.MOASTAR:
            return self._run_moastar(graph, start_uic, goal_uic)
        else:
            return RouteResult(error=f"Unknown algorithm: {algorithm}")

    def _run_dijkstra(
        self, graph: nx.MultiGraph, start: str, goal: str
    ) -> RouteResult:
        try:
            uic_path = nx.shortest_path(graph, start, goal, weight="weight")
            simplified = self.tg._simplify_path(uic_path)
            return RouteResult(simplified_path=simplified, full_uic_path=uic_path)
        except nx.NetworkXNoPath:
            return RouteResult(error="No path found (Dijkstra)")

    def _run_astar(
        self, graph: nx.MultiGraph, start: str, goal: str
    ) -> RouteResult:
        uic_path = astar_path(graph, start, goal)
        if uic_path is None:
            return RouteResult(error="No path found (A*)")
        simplified = self.tg._simplify_path(uic_path)
        return RouteResult(simplified_path=simplified, full_uic_path=uic_path)

    def _run_moastar(
        self, graph: nx.MultiGraph, start: str, goal: str
    ) -> RouteResult:
        pareto = moastar_paths(graph, start, goal)
        if not pareto:
            return RouteResult(error="No path found (MOA*)")

        best = pareto[0]
        simplified = self.tg._simplify_path(best.uic_path)
        return RouteResult(
            simplified_path=simplified,
            full_uic_path=best.uic_path,
            pareto_paths=pareto,
        )
