import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from geopy.distance import geodesic

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "datasets",
    "raw",
    "sncf",
)
DEFAULT_GARES_JSON = os.path.join(_DATA_DIR, "gares-de-voyageurs.json")
DEFAULT_LIGNES_JSON = os.path.join(_DATA_DIR, "lignes-par-type.json")
DEFAULT_LISTE_GARES_JSON = os.path.join(_DATA_DIR, "liste-des-gares.json")


def _parse_pk(pk_str: str) -> Optional[float]:
    """Parse '602+834' -> 602.834 km. Returns None if unparseable."""
    if not pk_str or not isinstance(pk_str, str):
        return None
    try:
        parts = pk_str.split("+")
        km = float(parts[0])
        m = float(parts[1]) / 1000.0 if len(parts) > 1 else 0.0
        return km + m
    except (ValueError, IndexError):
        return None


class TrainGraph:
    def __init__(
        self,
        gares_json: Optional[str] = None,
        lignes_json: Optional[str] = None,
        liste_gares_json: Optional[str] = None,
    ) -> None:
        gares_json = gares_json or DEFAULT_GARES_JSON
        lignes_json = lignes_json or DEFAULT_LIGNES_JSON
        liste_gares_json = liste_gares_json or DEFAULT_LISTE_GARES_JSON

        print(
            "Building railway graph (hybrid: topology + geometry)...",
            file=sys.stderr,
        )
        self.graph: nx.MultiGraph = nx.MultiGraph()
        self.stations: List[Dict[str, Any]] = []

        self._load_stations(gares_json)
        self._build_topology(liste_gares_json, lignes_json)
        self._add_city_transfers()

        print(
            f"Graph ready: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges.",
            file=sys.stderr,
        )

    def _load_stations(self, json_file: str) -> None:
        print("   - Loading passenger stations...", file=sys.stderr)
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
                    "coords": (lat, lon),
                }
                self.stations.append(station_data)
                self.graph.add_node(uic, name=name, pos=(lat, lon))
            except Exception:
                continue

    def _build_topology(self, liste_gares_json: str, lignes_json: str) -> None:
        print("   - Building topology from station-line mapping...", file=sys.stderr)

        lgv_lines = self._detect_lgv_lines(lignes_json)
        line_geometries = self._load_line_geometries(lignes_json)

        with open(liste_gares_json, "r", encoding="utf-8") as f:
            liste_data = json.load(f)

        known_uics = set(self.graph.nodes())
        lines: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in liste_data:
            try:
                uic = str(row.get("code_uic", ""))
                code_ligne = str(row.get("code_ligne", ""))
                pk_str = str(row.get("pk", ""))
                voyageurs = row.get("voyageurs", "N")

                if voyageurs != "O" or uic not in known_uics or not code_ligne:
                    continue

                pk_val = _parse_pk(pk_str)
                if pk_val is None:
                    continue

                x_wgs = row.get("x_wgs84")
                y_wgs = row.get("y_wgs84")

                lines[code_ligne].append({
                    "uic": uic,
                    "pk": pk_val,
                    "lat": y_wgs,
                    "lon": x_wgs,
                })
            except Exception:
                continue

        edges_added = 0
        for code_ligne, stations_on_line in lines.items():
            if len(stations_on_line) < 2:
                continue

            stations_on_line.sort(key=lambda s: s["pk"])

            deduped = [stations_on_line[0]]
            for s in stations_on_line[1:]:
                if s["uic"] != deduped[-1]["uic"]:
                    deduped.append(s)

            is_lgv = code_ligne in lgv_lines
            speed = 280.0 if is_lgv else 80.0

            for i in range(len(deduped) - 1):
                s1 = deduped[i]
                s2 = deduped[i + 1]

                dist_km = abs(s2["pk"] - s1["pk"])

                if dist_km < 0.1 or dist_km > 500:
                    pos1 = self.graph.nodes[s1["uic"]]["pos"]
                    pos2 = self.graph.nodes[s2["uic"]]["pos"]
                    dist_km = geodesic(pos1, pos2).km

                if dist_km < 0.1:
                    continue

                weight = dist_km / 3.0 if is_lgv else dist_km

                geometry = line_geometries.get(code_ligne)

                self.graph.add_edge(
                    s1["uic"],
                    s2["uic"],
                    weight=weight,
                    dist_km=dist_km,
                    speed=speed,
                    line=code_ligne,
                    type="TRAIN",
                    geometry=geometry,
                )
                edges_added += 1

        print(f"   - Topology: {edges_added} edges from {len(lines)} lines", file=sys.stderr)

    def _detect_lgv_lines(self, lignes_json: str) -> set:
        lgv_lines: set = set()
        with open(lignes_json, "r", encoding="utf-8") as f:
            lignes_data = json.load(f)

        for row in lignes_data:
            code = str(row.get("code_ligne", ""))
            name = str(row.get("lib_ligne", "")).upper()
            if "LGV" in name or "VITESSE" in name:
                lgv_lines.add(code)

        print(f"   - Detected {len(lgv_lines)} LGV lines", file=sys.stderr)
        return lgv_lines

    def _load_line_geometries(self, lignes_json: str) -> Dict[str, Any]:
        geometries: Dict[str, Any] = {}
        with open(lignes_json, "r", encoding="utf-8") as f:
            lignes_data = json.load(f)

        for row in lignes_data:
            code = str(row.get("code_ligne", ""))
            geo_shape = row.get("geo_shape", {})
            geometry = geo_shape.get("geometry")
            if code and geometry:
                coords = geometry.get("coordinates", [])
                if coords:
                    geometries[code] = [[c[1], c[0]] for c in coords]

        return geometries

    def _add_city_transfers(self) -> None:
        hubs = ["Paris", "Lyon", "Lille", "Marseille", "Bordeaux", "Nantes"]
        transfers_added = 0

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

                    if dist < 8:
                        self.graph.add_edge(
                            u1, u2,
                            weight=15,
                            dist_km=dist,
                            speed=5.0,
                            line="TRANSFERT",
                            type="WALK",
                        )
                        transfers_added += 1

        print(f"   - Added {transfers_added} transfer edges", file=sys.stderr)

    def get_path(
        self, dep_name: str, dest_name: str
    ) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
        start_uic = self._find_uic_by_name(dep_name)
        end_uic = self._find_uic_by_name(dest_name)

        if not start_uic:
            return None, f"Departure not found: {dep_name}", None
        if not end_uic:
            return None, f"Destination not found: {dest_name}", None

        try:
            full_path_uics = nx.shortest_path(
                self.graph, start_uic, end_uic, weight="weight"
            )
            simplified_names = self._simplify_path(full_path_uics)
            return simplified_names, None, full_path_uics

        except nx.NetworkXNoPath:
            return None, "No path found", None

    def _simplify_path(self, path_uics: List[str]) -> List[str]:
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

            is_hub = any(
                x in u_name
                for x in ["Paris", "Lyon", "Lille", "Bordeaux", "Marseille"]
            )

            if (prev_line and curr_line != prev_line) or (
                is_hub and "TGV" in u_name
            ):
                if final_stops[-1] != u_name:
                    final_stops.append(u_name)
            prev_line = curr_line

        if final_stops[-1] != full_names[-1]:
            final_stops.append(full_names[-1])
        return final_stops

    def _find_uic_by_name(self, search_name: str) -> Optional[str]:
        if not search_name:
            return None
        s = search_name.lower().strip()

        for uic, data in self.graph.nodes(data=True):
            if data["name"].lower() == s:
                return str(uic)

        for uic, data in self.graph.nodes(data=True):
            if s in data["name"].lower():
                return str(uic)

        return None
