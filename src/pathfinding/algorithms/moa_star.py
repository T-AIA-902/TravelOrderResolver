from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from geopy.distance import geodesic

SPEED_LGV = 280.0
SPEED_DEFAULT = 80.0
SPEED_WALK = 5.0


@dataclass(frozen=True)
class CostVector:
    time: float = 0.0
    distance: float = 0.0
    transfers: int = 0

    def __add__(self, other: CostVector) -> CostVector:
        return CostVector(
            time=self.time + other.time,
            distance=self.distance + other.distance,
            transfers=self.transfers + other.transfers,
        )

    def dominates(self, other: CostVector) -> bool:
        leq = (
            self.time <= other.time
            and self.distance <= other.distance
            and self.transfers <= other.transfers
        )
        lt = (
            self.time < other.time
            or self.distance < other.distance
            or self.transfers < other.transfers
        )
        return leq and lt


@dataclass
class PathResult:
    uic_path: List[str]
    cost: CostVector


class _ParetoFrontier:
    def __init__(self) -> None:
        self.costs: List[CostVector] = []

    def is_dominated(self, cost: CostVector) -> bool:
        return any(existing.dominates(cost) for existing in self.costs)

    def add(self, cost: CostVector) -> bool:
        if self.is_dominated(cost):
            return False
        self.costs = [c for c in self.costs if not cost.dominates(c)]
        self.costs.append(cost)
        return True


def _edge_costs(
    graph: nx.MultiGraph,
    u: str,
    v: str,
    prev_line: Optional[str],
) -> List[Tuple[CostVector, str]]:
    edge_data = graph.get_edge_data(u, v)
    results = []

    for key in edge_data:
        edata = edge_data[key]
        line_code = edata.get("line", "UNKNOWN")
        dist_km = edata.get("dist_km", 1.0)
        speed = edata.get("speed", SPEED_DEFAULT)
        time_h = dist_km / speed
        transfer = 1 if (prev_line is not None and line_code != prev_line) else 0

        results.append(
            (
                CostVector(time=time_h, distance=dist_km, transfers=transfer),
                line_code,
            )
        )

    return results


def _heuristic_vector(
    graph: nx.MultiGraph,
    node: str,
    goal: str,
) -> CostVector:
    pos_node = graph.nodes[node]["pos"]
    pos_goal = graph.nodes[goal]["pos"]
    geo_km = geodesic(pos_node, pos_goal).km
    return CostVector(
        time=geo_km / SPEED_LGV,
        distance=geo_km,
        transfers=0,
    )


def moastar_paths(
    graph: nx.MultiGraph,
    start: str,
    goal: str,
    max_solutions: int = 3,
) -> List[PathResult]:
    if start not in graph or goal not in graph:
        return []

    counter = 0
    zero_cost = CostVector()
    h = _heuristic_vector(graph, start, goal)

    open_set: list = []
    heapq.heappush(
        open_set,
        (
            h.time,
            counter,
            zero_cost,
            start,
            None,
            [start],
        ),
    )

    frontiers: Dict[str, _ParetoFrontier] = {}
    frontiers[start] = _ParetoFrontier()
    frontiers[start].add(zero_cost)

    solutions = _ParetoFrontier()
    results: List[PathResult] = []

    max_iterations = 200000

    while open_set and len(results) < max_solutions:
        max_iterations -= 1
        if max_iterations <= 0:
            break

        _, _, g_cost, current, prev_line, path = heapq.heappop(open_set)

        if current == goal:
            if solutions.add(g_cost):
                results.append(PathResult(uic_path=list(path), cost=g_cost))
            continue

        if current in frontiers and frontiers[current].is_dominated(g_cost):
            continue

        for neighbor in graph.neighbors(current):
            edge_options = _edge_costs(graph, current, neighbor, prev_line)

            for edge_cost, line_code in edge_options:
                new_g = g_cost + edge_cost

                if solutions.is_dominated(new_g):
                    continue

                if neighbor not in frontiers:
                    frontiers[neighbor] = _ParetoFrontier()

                if frontiers[neighbor].add(new_g):
                    h_vec = _heuristic_vector(graph, neighbor, goal)
                    f_time = new_g.time + h_vec.time
                    counter += 1
                    heapq.heappush(
                        open_set,
                        (
                            f_time,
                            counter,
                            new_g,
                            neighbor,
                            line_code,
                            path + [neighbor],
                        ),
                    )

    return results
