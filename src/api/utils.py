"""Shared utilities for API response building."""

import math


def _extract_segment_geometry(full_geometry, pos_u, pos_v):
    """Extract the portion of a line geometry between two stations.

    Given the full geometry of a railway line and the positions of two
    stations, returns the subset of coordinates that lies between them.
    Falls back to straight line if extraction fails.
    """
    if not full_geometry or len(full_geometry) < 2:
        return [list(pos_u), list(pos_v)]

    def _dist(p1, p2):
        dlat = (p1[0] - p2[0]) * 111.0
        dlon = (p1[1] - p2[1]) * 111.0 * math.cos(math.radians((p1[0] + p2[0]) / 2))
        return math.sqrt(dlat * dlat + dlon * dlon)

    # Find closest point in geometry to each station
    best_i_u, best_d_u = 0, float("inf")
    best_i_v, best_d_v = 0, float("inf")

    for idx, pt in enumerate(full_geometry):
        du = _dist(pt, pos_u)
        dv = _dist(pt, pos_v)
        if du < best_d_u:
            best_d_u = du
            best_i_u = idx
        if dv < best_d_v:
            best_d_v = dv
            best_i_v = idx

    # Ensure correct order
    start = min(best_i_u, best_i_v)
    end = max(best_i_u, best_i_v)

    segment = full_geometry[start : end + 1]

    if len(segment) < 2:
        return [list(pos_u), list(pos_v)]

    # Ensure segment starts/ends near the actual stations
    segment[0] = list(pos_u)
    segment[-1] = list(pos_v)

    return segment


def _resolve_hub(graph, uic, toward_uic=None):
    """If uic is a hub node (HUB_*), return the real station closest to toward_uic.

    When toward_uic is given, picks the local station that minimises geographic
    distance to the target — this avoids routing through the wrong side of a
    city for geometry building.  Falls back to highest-degree neighbor.
    """
    if not str(uic).startswith("HUB_"):
        return uic

    # Collect local (WALK/TRANSFERT) neighbors
    locals_ = []
    for neighbor in graph.graph.neighbors(uic):
        if str(neighbor).startswith("HUB_"):
            continue
        edge_data = graph.graph.get_edge_data(uic, neighbor)
        if not edge_data:
            continue
        is_local = any(
            e.get("type") == "WALK" or e.get("line") == "TRANSFERT"
            for e in edge_data.values()
        )
        if is_local:
            locals_.append(neighbor)

    if not locals_:
        return uic

    # If we know the destination, pick the closest station to it
    if toward_uic and toward_uic in graph.graph:
        target_pos = graph.graph.nodes[toward_uic]["pos"]

        def _dist_sq(n):
            p = graph.graph.nodes[n]["pos"]
            dlat = (p[0] - target_pos[0]) * 111.0
            dlon = (p[1] - target_pos[1]) * 111.0 * math.cos(
                math.radians((p[0] + target_pos[0]) / 2)
            )
            return dlat * dlat + dlon * dlon

        return min(locals_, key=_dist_sq)

    # Fallback: highest degree
    return max(locals_, key=lambda u: graph.graph.degree(u))


_track_graph_cache = None


def _get_track_graph(graph):
    """Get or build a graph with only track-level edges (no TGV/CONNEXION)."""
    global _track_graph_cache
    if _track_graph_cache is not None:
        return _track_graph_cache

    import networkx as nx
    tg = nx.MultiGraph()
    for n, data in graph.graph.nodes(data=True):
        if not str(n).startswith("HUB_"):
            tg.add_node(n, **data)
    for a, b, data in graph.graph.edges(data=True):
        if str(a).startswith("HUB_") or str(b).startswith("HUB_"):
            continue
        if data.get("type") != "SERVICE":
            tg.add_edge(a, b, **data)
    _track_graph_cache = tg
    return tg


def _find_nearest_track_node(graph, track_graph, uic):
    """Find the nearest node in the track graph to the given station."""
    if uic in track_graph and track_graph.degree(uic) > 0:
        return uic
    pos = graph.graph.nodes[uic]["pos"]
    best_uic = None
    best_dist = float("inf")
    for n, data in track_graph.nodes(data=True):
        if track_graph.degree(n) == 0:
            continue
        np = data.get("pos")
        if not np:
            continue
        dlat = (pos[0] - np[0]) * 111.0
        dlon = (pos[1] - np[1]) * 111.0 * math.cos(math.radians((pos[0] + np[0]) / 2))
        d = dlat * dlat + dlon * dlon
        if d < best_dist:
            best_dist = d
            best_uic = n
    return best_uic


def _build_via_geometry(graph, u, v):
    """Build geometry for service/TGV edges by following actual tracks.

    Finds the track-level path and collects the real rail geometry
    for each segment, producing a smooth trace along the rails.
    Falls back to nearest track stations if endpoints aren't in the track graph.
    """
    import networkx as nx

    try:
        track_graph = _get_track_graph(graph)
        # Find nearest track nodes for stations not in track graph
        track_u = _find_nearest_track_node(graph, track_graph, u)
        track_v = _find_nearest_track_node(graph, track_graph, v)
        if track_u and track_v:
            path = nx.shortest_path(track_graph, track_u, track_v, weight="weight")
            points = []
            for i in range(len(path) - 1):
                seg_u, seg_v = path[i], path[i + 1]
                edge_data = track_graph.get_edge_data(seg_u, seg_v)
                if edge_data:
                    edge = edge_data[0] if isinstance(edge_data, dict) else edge_data
                    geo = edge.get("geometry")
                    if geo and len(geo) > 2:
                        # Use real rail geometry for this segment
                        pos_u = graph.graph.nodes[seg_u]["pos"]
                        pos_v = graph.graph.nodes[seg_v]["pos"]
                        seg_points = _extract_segment_geometry(geo, pos_u, pos_v)
                        # Avoid duplicating the junction point
                        if points and seg_points:
                            points.extend(seg_points[1:])
                        else:
                            points.extend(seg_points)
                        continue
                # Fallback: straight line for this segment
                p = list(graph.graph.nodes[seg_u]["pos"])
                if not points or points[-1] != p:
                    points.append(p)
            # Add last point
            last_p = list(graph.graph.nodes[path[-1]]["pos"])
            if not points or points[-1] != last_p:
                points.append(last_p)
            if len(points) >= 2:
                # Ensure endpoints match the actual stations (not nearest proxy)
                points[0] = list(graph.graph.nodes[u]["pos"])
                points[-1] = list(graph.graph.nodes[v]["pos"])
                return points
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass

    return [list(graph.graph.nodes[u]["pos"]), list(graph.graph.nodes[v]["pos"])]


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
    prev_type = None

    for i in range(len(full_uic_path) - 1):
        u, v = full_uic_path[i], full_uic_path[i + 1]
        edge_dict = graph.graph.get_edge_data(u, v)
        if not edge_dict:
            continue
        # Pick the edge with lowest weight (best route)
        edge_data = min(edge_dict.values(), key=lambda e: e.get("weight", float("inf")))

        # Skip hub-to-station walk edges (internal city transfers)
        if edge_data.get("line") == "TRANSFERT" and (
            str(u).startswith("HUB_") or str(v).startswith("HUB_")
        ):
            continue
        line = edge_data.get("line", "UNKNOWN")
        seg_type = edge_data.get("type", "TRAIN")

        # Count transfers: walk segments always count; for SERVICE/TGV edges
        # only count when the axe changes (same axe = same train line)
        is_transfer = False
        if prev_line is not None:
            if seg_type == "WALK" or prev_type == "WALK":
                is_transfer = True
            elif line != prev_line:
                # Same axe prefix means same high-speed corridor, not a transfer
                prev_axe = prev_line.split("-", 1)[1] if "-" in prev_line else prev_line
                curr_axe = line.split("-", 1)[1] if "-" in line else line
                if prev_axe != curr_axe or seg_type not in ("SERVICE", "TRAIN"):
                    is_transfer = True
        if is_transfer:
            transfers += 1
        prev_line = line
        prev_type = seg_type

        # Resolve hub nodes to the real station closest to the other endpoint
        real_u = _resolve_hub(graph, u, toward_uic=v)
        real_v = _resolve_hub(graph, v, toward_uic=u)

        pos_u = graph.graph.nodes[real_u]["pos"]
        pos_v = graph.graph.nodes[real_v]["pos"]

        # Use line geometry if available, extract segment between stations
        edge_geometry = edge_data.get("geometry")
        if edge_geometry and len(edge_geometry) > 2:
            geometry = _extract_segment_geometry(edge_geometry, pos_u, pos_v)
        elif line.startswith("TGV") or line == "CONNEXION" or seg_type == "SERVICE":
            geometry = _build_via_geometry(graph, real_u, real_v)
        else:
            geometry = [list(pos_u), list(pos_v)]

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
        "total_stops": len(simplified) if simplified else 0,
        "transfers": transfers,
        "error": None,
    }
