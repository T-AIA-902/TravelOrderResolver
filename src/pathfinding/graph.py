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
    """
    Railway network graph with Dijkstra and A* pathfinding.

    Builds a MultiGraph from SNCF station and line data, with optimized
    weights for high-speed lines (LGV). Provides shortest path computation
    using either Dijkstra or A* algorithm, and path simplification to show
    only key stops.

    A* uses geodesic distance as heuristic, which is admissible since the
    straight-line distance is always <= actual path distance.

    Complexity:
        - Dijkstra: O((V + E) log V)
        - A*: O((V + E) log V) but explores fewer nodes due to heuristic
    """

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
        self._connect_components()
        self._add_direct_services()

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

                lines[code_ligne].append(
                    {
                        "uic": uic,
                        "pk": pk_val,
                        "lat": y_wgs,
                        "lon": x_wgs,
                    }
                )
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
                            u1,
                            u2,
                            weight=15,
                            dist_km=dist,
                            speed=5.0,
                            line="TRANSFERT",
                            type="WALK",
                        )
                        transfers_added += 1

        print(f"   - Added {transfers_added} transfer edges", file=sys.stderr)

    def _add_direct_services(self) -> None:
        """Add train service edges from tgvmax.json by reconstructing itineraries.

        Groups schedule entries by (train_no, date), reconstructs the stop
        sequence by sorting origins by departure time, then adds edges between
        consecutive stops with real travel times.  Each edge is labeled with
        its axe (e.g. SUD EST, ATLANTIQUE) so that consecutive edges on the
        same axe are not counted as transfers.
        """
        tgvmax_path = os.path.join(_DATA_DIR, "tgvmax.json")
        if not os.path.exists(tgvmax_path):
            print("   - tgvmax.json not found, skipping services", file=sys.stderr)
            return

        with open(tgvmax_path, "r", encoding="utf-8") as f:
            schedule = json.load(f)

        # Build name -> UIC mapping for tgvmax station names
        name_to_uic: Dict[str, Optional[str]] = {}
        graph_names_lower: Dict[str, str] = {}
        for uic, data in self.graph.nodes(data=True):
            graph_names_lower[data["name"].lower()] = uic

        # Create hub nodes for city-level references (e.g. "PARIS (intramuros)")
        # Hub connects to all stations in the city via zero-weight transfer edges
        hub_cities: Dict[str, str] = {}  # clean_city -> hub_uic
        intramuros_names = set()
        for entry in schedule:
            for field in ("origine", "destination"):
                name = entry.get(field, "")
                if "(intramuros)" in name.lower():
                    intramuros_names.add(name)

        for raw_name in intramuros_names:
            city = raw_name.lower().replace("(intramuros)", "").strip()
            if city in hub_cities:
                continue
            hub_uic = f"HUB_{city.upper()}"
            # Find all stations matching this city
            city_stations = []
            for uic, data in self.graph.nodes(data=True):
                sname = data["name"].lower()
                if sname.startswith(city + " ") or sname == city:
                    city_stations.append(uic)
            if not city_stations:
                continue
            # Compute centroid position
            lats = [self.graph.nodes[u]["pos"][0] for u in city_stations]
            lons = [self.graph.nodes[u]["pos"][1] for u in city_stations]
            hub_pos = (sum(lats) / len(lats), sum(lons) / len(lons))
            # Capitalize for display
            display = city.title()
            self.graph.add_node(hub_uic, name=display, pos=hub_pos)
            for st_uic in city_stations:
                self.graph.add_edge(
                    hub_uic, st_uic,
                    weight=0.25,  # ~15 min transfer within city
                    dist_km=0.5,
                    speed=5.0,
                    line="TRANSFERT",
                    type="WALK",
                )
            hub_cities[city] = hub_uic

        def _resolve(tgv_name: str) -> Optional[str]:
            if tgv_name in name_to_uic:
                return name_to_uic[tgv_name]
            clean = tgv_name.lower().strip()
            # City-level references resolve to hub node
            if "(intramuros)" in clean:
                city = clean.replace("(intramuros)", "").strip()
                if city in hub_cities:
                    name_to_uic[tgv_name] = hub_cities[city]
                    return hub_cities[city]
            clean = clean.replace("(intramuros)", "").strip()
            clean = clean.replace(" st ", " saint-").replace("st ", "saint-")
            if clean in graph_names_lower:
                name_to_uic[tgv_name] = graph_names_lower[clean]
                return name_to_uic[tgv_name]
            # Prefix match: pick the station with highest degree
            first_word = clean.split()[0] if clean else ""
            candidates = []
            for gname, guic in graph_names_lower.items():
                if gname.startswith(clean) or (
                    len(first_word) > 3 and gname.startswith(first_word)
                ):
                    candidates.append(guic)
            if candidates:
                best = max(candidates, key=lambda u: self.graph.degree(u))
                name_to_uic[tgv_name] = best
                return best
            name_to_uic[tgv_name] = None
            return None

        def _normalize_axe(axe: str) -> str:
            """Normalize axe to merge OUIGO variants with their TGV counterparts."""
            a = axe.lower().replace("ouigo_", "").replace("_", " ").strip()
            # Map common variants to canonical names
            mapping = {
                "tc": "transverse",
                "sud-est": "sud est",
                "sud est": "sud est",
                "atlantique": "atlantique",
                "est": "est",
                "nord": "nord",
            }
            return mapping.get(a, a)

        def _parse_hm(s: str) -> int:
            h, m = map(int, s.split(":"))
            return h * 60 + m

        # Group entries by (train_no, date)
        trains: Dict[tuple, list] = defaultdict(list)
        for entry in schedule:
            key = (entry.get("train_no", ""), entry.get("date", ""))
            trains[key].append(entry)

        # Reconstruct itineraries and collect consecutive-stop durations
        pair_durations: Dict[tuple, list] = defaultdict(list)  # (uicA, uicB) -> [min]
        pair_axes: Dict[tuple, str] = {}

        for (_train_no, _date), entries in trains.items():
            # Collect departure time per origin station, arrival per destination
            stops_dep: Dict[str, int] = {}  # tgv_name -> dep_minutes
            all_dests: Dict[str, int] = {}  # tgv_name -> arr_minutes
            od_times: Dict[tuple, tuple] = {}  # (orig, dest) -> (dep, arr)
            axe = ""

            for e in entries:
                orig = e.get("origine", "")
                dest = e.get("destination", "")
                axe = e.get("axe", axe)
                try:
                    dep = _parse_hm(e["heure_depart"])
                    arr = _parse_hm(e["heure_arrivee"])
                    if arr <= dep:
                        arr += 24 * 60
                except (ValueError, KeyError):
                    continue

                if orig not in stops_dep or dep < stops_dep[orig]:
                    stops_dep[orig] = dep
                if dest not in all_dests or arr > all_dests[dest]:
                    all_dests[dest] = arr
                od_times[(orig, dest)] = (dep, arr)

            # Build sorted stop sequence: origins by dep time, then terminus
            sorted_stops = sorted(stops_dep.items(), key=lambda x: x[1])
            for dest_name, arr in all_dests.items():
                if dest_name not in stops_dep:
                    sorted_stops.append((dest_name, arr))

            if len(sorted_stops) < 2:
                continue

            # Add consecutive pairs with exact travel times
            for i in range(len(sorted_stops) - 1):
                name_a = sorted_stops[i][0]
                name_b = sorted_stops[i + 1][0]
                uic_a = _resolve(name_a)
                uic_b = _resolve(name_b)
                if not uic_a or not uic_b or uic_a == uic_b:
                    continue

                # Get exact duration from OD entry if available
                if (name_a, name_b) in od_times:
                    dep, arr = od_times[(name_a, name_b)]
                    dur = arr - dep
                else:
                    # Fallback: difference between departure times
                    dep_a = sorted_stops[i][1]
                    dep_b = sorted_stops[i + 1][1]
                    dur = dep_b - dep_a

                if 1 < dur < 480:
                    key = tuple(sorted([uic_a, uic_b]))
                    pair_durations[key].append(dur)
                    pair_axes[key] = _normalize_axe(axe)

        # Add edges for all service pairs
        services_added = 0
        for (uic1, uic2), durations in pair_durations.items():
            if not durations:
                continue
            avg_hours = (sum(durations) / len(durations)) / 60.0
            pos1 = self.graph.nodes[uic1]["pos"]
            pos2 = self.graph.nodes[uic2]["pos"]
            dist_km = geodesic(pos1, pos2).km
            axe = pair_axes.get((uic1, uic2), "")
            line_label = f"TGV-{axe}" if axe else "TGV"

            self.graph.add_edge(
                uic1,
                uic2,
                weight=avg_hours,
                dist_km=round(dist_km, 1),
                speed=round(dist_km / avg_hours, 0) if avg_hours > 0 else 200,
                line=line_label,
                type="SERVICE",
                freq=len(durations),
            )
            services_added += 1

        print(
            f"   - Added {services_added} service edges from "
            f"{len(trains)} train itineraries",
            file=sys.stderr,
        )

    def _connect_components(self) -> None:
        """Connect disconnected graph components by linking nearest stations.

        Uses a spatial approach: for each isolated component, pick its
        centroid station and find the nearest station in the already-connected
        network. Much faster than comparing all pairs.
        """
        import math

        components = list(nx.connected_components(self.graph))
        if len(components) <= 1:
            return

        components.sort(key=len, reverse=True)
        connected = set(components[0])

        # Build flat list of (uic, lat, lon) for fast lookup
        def _pos(uic: str) -> Optional[Tuple[float, float]]:
            p = self.graph.nodes[uic].get("pos")
            if p and p != (0.0, 0.0):
                return p
            return None

        def _approx_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            """Fast approximate distance in km (no geodesic call)."""
            dlat = (p1[0] - p2[0]) * 111.0
            dlon = (p1[1] - p2[1]) * 111.0 * math.cos(math.radians((p1[0] + p2[0]) / 2))
            return math.sqrt(dlat * dlat + dlon * dlon)

        bridges_added = 0
        for comp in components[1:]:
            # Pick a representative station from this component (one with valid pos)
            comp_stations = [(uic, _pos(uic)) for uic in comp]
            comp_stations = [(u, p) for u, p in comp_stations if p is not None]
            if not comp_stations:
                continue

            # Find nearest connected station for each comp station, keep best
            best_pair = None
            best_dist = float("inf")

            for comp_uic, comp_pos in comp_stations:
                for conn_uic in connected:
                    conn_pos = _pos(conn_uic)
                    if conn_pos is None:
                        continue
                    dist = _approx_km(comp_pos, conn_pos)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (comp_uic, conn_uic)

            if best_pair and best_dist < 100:
                u, v = best_pair
                speed = 60.0
                weight = best_dist / speed
                self.graph.add_edge(
                    u, v,
                    weight=weight,
                    dist_km=round(best_dist, 1),
                    speed=speed,
                    line="CONNEXION",
                    type="TRAIN",
                )
                bridges_added += 1
                connected |= comp

        print(f"   - Connected {bridges_added} isolated components (< 100km)", file=sys.stderr)

    def _heuristic(self, node_uic: str, goal_uic: str) -> float:
        """
        A* heuristic: geodesic distance to goal.

        This heuristic is admissible because the straight-line distance
        is always less than or equal to the actual path distance.

        Args:
            node_uic: Current node UIC
            goal_uic: Goal node UIC

        Returns:
            Estimated distance to goal in km
        """
        pos1 = self.graph.nodes[node_uic]["pos"]
        pos2 = self.graph.nodes[goal_uic]["pos"]
        return float(geodesic(pos1, pos2).km)

    def get_path(
        self,
        dep_name: str,
        dest_name: str,
        intermediates: Optional[List[str]] = None,
        algorithm: str = "astar",
    ) -> Tuple[Optional[List[str]], Optional[str], Optional[List[str]]]:
        """
        Find shortest path between two stations, optionally via intermediates.

        Args:
            dep_name: Departure station name
            dest_name: Destination station name
            intermediates: Optional list of intermediate station names
            algorithm: Pathfinding algorithm ("dijkstra" or "astar")

        Returns:
            Tuple of (simplified_path, error_message, full_uic_path)
        """
        # Build waypoints: departure -> intermediates -> destination
        waypoints = [dep_name]
        if intermediates:
            waypoints.extend(intermediates)
        waypoints.append(dest_name)

        # Find UICs for all waypoints
        waypoint_uics = []
        for wp in waypoints:
            uic = self._find_uic_by_name(wp)
            if not uic:
                return None, f"Station not found: {wp}", None
            waypoint_uics.append(uic)

        # For departure and destination, try alternative stations in the same city
        # and pick the combination with the shortest total path
        dep_alternatives = self._find_city_alternatives(waypoints[0])
        dest_alternatives = self._find_city_alternatives(waypoints[-1])

        best_path: Optional[List[str]] = None
        best_weight = float("inf")
        best_waypoints: List[str] = waypoint_uics

        dep_candidates = dep_alternatives if dep_alternatives else [waypoint_uics[0]]
        dest_candidates = dest_alternatives if dest_alternatives else [waypoint_uics[-1]]

        for dep_uic in dep_candidates:
            for dest_uic in dest_candidates:
                trial_wps = [dep_uic] + waypoint_uics[1:-1] + [dest_uic]
                path = self._chain_path(trial_wps, algorithm)
                if path is not None:
                    weight = sum(
                        self.graph[path[i]][path[i + 1]][0].get("weight", 1)
                        for i in range(len(path) - 1)
                    )
                    if weight < best_weight:
                        best_weight = weight
                        best_path = path
                        best_waypoints = trial_wps

        if best_path is None:
            return None, "No path found", None

        simplified_names = self._simplify_path(best_path, best_waypoints)
        return simplified_names, None, best_path

    def _find_city_alternatives(self, station_name: str) -> List[str]:
        """Find all major stations in the same city (degree >= 4), including hubs."""
        city = station_name.lower().split()[0]  # "Paris Montparnasse" -> "paris"
        alternatives = []
        # Check for hub node first
        hub_uic = f"HUB_{city.upper()}"
        if hub_uic in self.graph:
            alternatives.append(hub_uic)
        for uic, data in self.graph.nodes(data=True):
            if str(uic).startswith("HUB_"):
                continue
            if data["name"].lower().startswith(city + " ") or data["name"].lower() == city:
                if self.graph.degree(uic) >= 4:
                    alternatives.append(str(uic))
        return alternatives if len(alternatives) > 1 else []

    def _chain_path(self, waypoint_uics: List[str], algorithm: str) -> Optional[List[str]]:
        """Chain paths between consecutive waypoints. Returns None if any segment fails."""
        full_path: List[str] = []
        try:
            for i in range(len(waypoint_uics) - 1):
                start_uic = waypoint_uics[i]
                end_uic = waypoint_uics[i + 1]
                if algorithm == "astar":
                    segment = nx.astar_path(
                        self.graph, start_uic, end_uic,
                        heuristic=lambda u, v: self._heuristic(u, end_uic),
                        weight="weight",
                    )
                else:
                    segment = nx.shortest_path(self.graph, start_uic, end_uic, weight="weight")
                if full_path and segment:
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
            return full_path
        except nx.NetworkXNoPath:
            return None

    def _simplify_path(
        self, path_uics: List[str], waypoint_uics: Optional[List[str]] = None
    ) -> List[str]:
        """
        Simplify path to show only key stops (hubs, line changes, and waypoints).

        Skips hub nodes (HUB_*) from the simplified output since they are
        virtual routing nodes, not real stations.

        Args:
            path_uics: Full list of station UICs
            waypoint_uics: Optional list of waypoint UICs that must be included

        Returns:
            Simplified list of station names
        """
        if not path_uics:
            return []

        waypoints_set = set(waypoint_uics) if waypoint_uics else set()

        display_uics = list(path_uics)
        if not display_uics:
            return []

        full_names = [self.graph.nodes[u]["name"] for u in display_uics]
        final_stops = [full_names[0]]
        prev_line: Optional[str] = None

        for i in range(len(display_uics) - 1):
            u, v = display_uics[i], display_uics[i + 1]
            edge_data = self.graph.get_edge_data(u, v)
            if not edge_data:
                # Nodes may not be directly connected after hub removal
                prev_line = None
                continue
            curr_line = edge_data[0].get("line", "UNKNOWN")
            curr_type = edge_data[0].get("type", "TRAIN")
            u_name = self.graph.nodes[u]["name"]

            is_hub = any(
                x in u_name
                for x in [
                    "Paris", "Lyon", "Lille", "Bordeaux", "Marseille",
                    "Toulouse", "Nantes", "Rennes", "Strasbourg",
                ]
            )
            is_waypoint = u in waypoints_set
            is_line_change = prev_line is not None and curr_line != prev_line
            is_service_stop = curr_type == "SERVICE" or (
                prev_line is not None
                and prev_line.startswith("TGV")
            )

            if is_line_change or is_hub or is_waypoint or is_service_stop:
                if final_stops[-1] != u_name:
                    final_stops.append(u_name)
            prev_line = curr_line

        if final_stops[-1] != full_names[-1]:
            final_stops.append(full_names[-1])
        return final_stops

    def _find_uic_by_name(self, search_name: str) -> Optional[str]:
        """
        Find station UIC by name (exact match, then prefix, then contains).

        Priority:
        1. Exact match: "Paris" == "Paris"
        2. Prefix match: "Paris" matches "Paris Gare de Lyon"
        3. Contains match: "Paris" matches "Cormeilles-en-Parisis" (fallback)
        """
        if not search_name:
            return None
        s = search_name.lower().strip()

        for uic, data in self.graph.nodes(data=True):
            if data["name"].lower() == s:
                return str(uic)

        # Try prefix match (station name starts with search term)
        for uic, data in self.graph.nodes(data=True):
            name_lower = data["name"].lower()
            if name_lower.startswith(s) or name_lower.startswith(s + " "):
                return str(uic)

        # Fall back to contains match
        for uic, data in self.graph.nodes(data=True):
            if s in data["name"].lower():
                return str(uic)

        return None
