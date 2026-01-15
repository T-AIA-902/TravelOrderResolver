"""
Railway network graph for pathfinding.

This module builds a graph of the French railway network from SNCF data
and provides shortest path computation using Dijkstra's algorithm.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from geopy.distance import geodesic
from shapely.geometry import Point, shape
from shapely.ops import substring

# Default paths to SNCF data files (JSON format)
DEFAULT_GARES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "datasets",
    "raw",
    "sncf",
    "gares-de-voyageurs.json",
)
DEFAULT_LIGNES_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "datasets",
    "raw",
    "sncf",
    "lignes-par-type.json",
)


class TrainGraph:
    """
    Railway network graph with Dijkstra pathfinding.

    Builds a MultiGraph from SNCF station and line data, with optimized
    weights for high-speed lines (LGV). Provides shortest path computation
    and path simplification to show only key stops.
    """

    def __init__(
        self,
        gares_json: Optional[str] = None,
        lignes_json: Optional[str] = None,
    ) -> None:
        """
        Initialize the railway graph.

        Args:
            gares_json: Path to stations JSON file
                (default: datasets/raw/sncf/gares-de-voyageurs.json)
            lignes_json: Path to lines JSON file
                (default: datasets/raw/sncf/lignes-par-type.json)
        """
        gares_json = gares_json or DEFAULT_GARES_JSON
        lignes_json = lignes_json or DEFAULT_LIGNES_JSON

        print(
            "Building railway graph (HD geometry with LGV optimization)...",
            file=sys.stderr,
        )
        self.graph: nx.MultiGraph = nx.MultiGraph()
        self.stations: List[Dict[str, Any]] = []

        self._load_stations(gares_json)
        self._map_lines(lignes_json)
        self._add_city_transfers()
        self._ensure_connectivity()

        print(
            f"Graph ready: {self.graph.number_of_nodes()} stations connected.",
            file=sys.stderr,
        )

    def _load_stations(self, json_file: str) -> None:
        """Load station data from JSON file."""
        print("   - Loading stations...", file=sys.stderr)
        with open(json_file, "r", encoding="utf-8") as f:
            stations_data = json.load(f)

        for row in stations_data:
            try:
                pos = row.get("position_geographique")
                if not pos:
                    continue

                lon = pos.get("lon")
                lat = pos.get("lat")
                if lon is None or lat is None:
                    continue

                name = str(row.get("nom", ""))
                uic = str(row.get("codes_uic", ""))

                if not name or not uic:
                    continue

                station_data = {
                    "uic": uic,
                    "name": name,
                    "point": Point(lon, lat),
                    "coords": (lat, lon),
                }
                self.stations.append(station_data)
                self.graph.add_node(uic, name=name, pos=(lat, lon))
            except Exception:
                continue

    def _map_lines(self, lignes_json: str) -> None:
        """Map railway lines and create edges between stations."""
        print(
            "   - Mapping line geometries (LGV optimization enabled)...",
            file=sys.stderr,
        )
        with open(lignes_json, "r", encoding="utf-8") as f:
            lignes_data = json.load(f)

        tolerance_deg = 0.002

        for row in lignes_data:
            try:
                geo_shape = row.get("geo_shape", {})
                geometry = geo_shape.get("geometry")
                if not geometry:
                    continue

                line_obj = shape(geometry)
                line_code = str(row.get("code_ligne", ""))
                line_name = str(row.get("lib_ligne", "")).upper()

                # High-speed lines get reduced weight (3x faster)
                is_lgv = "LGV" in line_name or "VITESSE" in line_name

                minx, miny, maxx, maxy = line_obj.bounds
                possible_stations = [
                    s
                    for s in self.stations
                    if minx - tolerance_deg <= s["point"].x <= maxx + tolerance_deg
                    and miny - tolerance_deg <= s["point"].y <= maxy + tolerance_deg
                ]

                stations_on_line: List[Tuple[Dict[str, Any], float]] = []
                for s in possible_stations:
                    dist_proj = line_obj.project(s["point"])
                    if line_obj.distance(s["point"]) < tolerance_deg:
                        stations_on_line.append((s, dist_proj))

                stations_on_line.sort(key=lambda x: x[1])

                for i in range(len(stations_on_line) - 1):
                    s1, dist1 = stations_on_line[i]
                    s2, dist2 = stations_on_line[i + 1]

                    if dist1 != dist2:
                        segment = substring(line_obj, dist1, dist2)
                        curve_coords = [[y, x] for x, y in segment.coords]
                    else:
                        curve_coords = [s1["coords"], s2["coords"]]

                    dist_km = geodesic(s1["coords"], s2["coords"]).km
                    weight = dist_km / 3.0 if is_lgv else dist_km

                    self.graph.add_edge(
                        s1["uic"],
                        s2["uic"],
                        weight=weight,
                        line=line_code,
                        type="TRAIN",
                        geometry=curve_coords,
                    )

            except Exception:
                continue

    def _add_city_transfers(self) -> None:
        """Add walking transfer edges between stations in the same city hub."""
        hubs = ["Paris", "Lyon", "Lille", "Marseille", "Bordeaux", "Nantes"]
        for hub in hubs:
            candidates: List[str] = []
            for uic, data in self.graph.nodes(data=True):
                if hub.lower() in data["name"].lower():
                    candidates.append(uic)

            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    u1, u2 = candidates[i], candidates[j]
                    d1 = self.graph.nodes[u1]["pos"]
                    d2 = self.graph.nodes[u2]["pos"]
                    dist = geodesic(d1, d2).km

                    # Only add transfer if stations are within 8km
                    if dist < 8:
                        self.graph.add_edge(u1, u2, weight=15, line="TRANSFERT", type="WALK")

    def _ensure_connectivity(self) -> None:
        """Placeholder for future connectivity improvements."""
        # Could add logic to connect orphan nodes
        pass

    def get_path(
        self, dep_name: str, dest_name: str
    ) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
        """
        Find shortest path between two stations.

        Args:
            dep_name: Departure station name
            dest_name: Destination station name

        Returns:
            Tuple of (simplified_path, error_message, full_uic_path)
            - simplified_path: List of key station names on the route
            - error_message: Error string if path not found, None otherwise
            - full_uic_path: Complete list of station UICs for visualization
        """
        start_uic = self._find_uic_by_name(dep_name)
        end_uic = self._find_uic_by_name(dest_name)

        if not start_uic:
            return None, f"Departure not found: {dep_name}", None
        if not end_uic:
            return None, f"Destination not found: {dest_name}", None

        try:
            full_path_uics = nx.shortest_path(self.graph, start_uic, end_uic, weight="weight")
            simplified_names = self._simplify_path(full_path_uics)
            return simplified_names, None, full_path_uics

        except nx.NetworkXNoPath:
            return None, "No path found", None

    def _simplify_path(self, path_uics: List[str]) -> List[str]:
        """
        Simplify path to show only key stops (hubs and line changes).

        Args:
            path_uics: Full list of station UICs

        Returns:
            Simplified list of station names
        """
        if not path_uics:
            return []

        full_names = [self.graph.nodes[u]["name"] for u in path_uics]
        final_stops = [full_names[0]]
        prev_line: Optional[str] = None

        for i in range(len(path_uics) - 1):
            u, v = path_uics[i], path_uics[i + 1]
            edge_data = self.graph.get_edge_data(u, v)[0]
            curr_line = edge_data.get("line", "UNKNOWN")
            u_name = self.graph.nodes[u]["name"]

            is_hub = any(x in u_name for x in ["Paris", "Lyon", "Lille", "Bordeaux", "Marseille"])

            # Add stop if line changes or at major hub
            if (prev_line and curr_line != prev_line) or (is_hub and "TGV" in u_name):
                if final_stops[-1] != u_name:
                    final_stops.append(u_name)
            prev_line = curr_line

        if final_stops[-1] != full_names[-1]:
            final_stops.append(full_names[-1])
        return final_stops

    def _find_uic_by_name(self, search_name: str) -> Optional[str]:
        """
        Find station UIC by name (exact match first, then partial).

        Args:
            search_name: Station name to search

        Returns:
            Station UIC if found, None otherwise
        """
        if not search_name:
            return None
        s = search_name.lower().strip()

        # Try exact match first
        for uic, data in self.graph.nodes(data=True):
            if data["name"].lower() == s:
                return str(uic)

        # Fall back to partial match
        for uic, data in self.graph.nodes(data=True):
            if s in data["name"].lower():
                return str(uic)

        return None
