import heapq
from typing import Dict, List, Optional, Set

import networkx as nx


def dijkstra_path(
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
    dist: Dict[str, float] = {start: 0.0}
    visited: Set[str] = set()

    while open_set:
        d, _, current = heapq.heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in graph.neighbors(current):
            if neighbor in visited:
                continue

            edge_data = graph.get_edge_data(current, neighbor)
            min_weight = min(edge_data[k].get("weight", 1.0) for k in edge_data)

            tentative_dist = dist[current] + min_weight

            if tentative_dist < dist.get(neighbor, float("inf")):
                came_from[neighbor] = current
                dist[neighbor] = tentative_dist
                counter += 1
                heapq.heappush(open_set, (tentative_dist, counter, neighbor))

    return None


def _reconstruct_path(came_from: Dict[str, str], current: str) -> List[str]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
