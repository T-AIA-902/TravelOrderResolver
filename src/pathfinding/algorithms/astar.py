import heapq
from typing import Dict, List, Optional, Set

import networkx as nx
from geopy.distance import geodesic


def _geodesic_heuristic(graph: nx.MultiGraph, node: str, goal: str) -> float:
    pos_node = graph.nodes[node]["pos"]
    pos_goal = graph.nodes[goal]["pos"]
    return geodesic(pos_node, pos_goal).km / 3.0


def astar_path(
    graph: nx.MultiGraph,
    start: str,
    goal: str,
) -> Optional[List[str]]:
    if start not in graph or goal not in graph:
        return None

    open_set: list = []
    counter = 0
    heapq.heappush(open_set, (0.0, counter, start))

    came_from: Dict[str, str] = {}
    g_score: Dict[str, float] = {start: 0.0}
    visited: Set[str] = set()

    while open_set:
        f, _, current = heapq.heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        if current in visited:
            continue
        visited.add(current)

        current_g = g_score[current]

        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue

            edge_data = graph.get_edge_data(current, neighbor)
            min_weight = min(
                edge_data[k].get("weight", 1.0) for k in edge_data
            )

            tentative_g = current_g + min_weight

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                h = _geodesic_heuristic(graph, neighbor, goal)
                counter += 1
                heapq.heappush(
                    open_set, (tentative_g + h, counter, neighbor)
                )

    return None


def _reconstruct_path(came_from: Dict[str, str], current: str) -> List[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
