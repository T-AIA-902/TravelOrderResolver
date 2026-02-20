"""Shared utilities for API response building."""


def build_route_response(graph, simplified, error, full_uic_path):
    """Build pathfinding response dict from TrainGraph.get_path() results."""
    if error or not full_uic_path:
        return {
            "found": False,
            "route": [],
            "route_details": [],
            "total_stops": 0,
            "transfers": 0,
            "error": error,
        }

    route_details = []
    transfers = 0
    prev_line = None

    for i in range(len(full_uic_path) - 1):
        u, v = full_uic_path[i], full_uic_path[i + 1]
        edge_data = graph.graph.get_edge_data(u, v)[0]  # [0] for MultiGraph
        line = edge_data.get("line", "UNKNOWN")
        seg_type = edge_data.get("type", "TRAIN")

        if prev_line is not None and line != prev_line:
            transfers += 1
        prev_line = line

        # Use node positions (not line geometry which covers the whole line)
        geometry = [
            list(graph.graph.nodes[u]["pos"]),
            list(graph.graph.nodes[v]["pos"]),
        ]

        route_details.append(
            {
                "from_station": graph.graph.nodes[u]["name"],
                "from_uic": u,
                "to_station": graph.graph.nodes[v]["name"],
                "to_uic": v,
                "line": line,
                "type": seg_type,
                "geometry": geometry,
            }
        )

    return {
        "found": True,
        "route": simplified,
        "route_details": route_details,
        "total_stops": len(simplified),
        "transfers": transfers,
        "error": None,
    }
