import pandas as pd
import networkx as nx
import json
import sys
from shapely.geometry import Point, shape
from shapely.ops import substring
from geopy.distance import geodesic

class TrainGraph:
    def __init__(self, gares_csv, lignes_csv):
        print("🗺️  Construction du Graphe Ferroviaire OPTIMISÉ (Géométrie HD)...", file=sys.stderr)
        self.graph = nx.MultiGraph()
        self.stations = []

        self.load_stations(gares_csv)
        self.map_lines(lignes_csv)
        self.add_city_transfers()
        self.ensure_connectivity()
        
        print(f"✅ Graphe terminé : {self.graph.number_of_nodes()} gares connectées.", file=sys.stderr)

    def load_stations(self, csv_file):
        print(f"   - Chargement des gares...", file=sys.stderr)
        try:
            df = pd.read_csv(csv_file, sep=";", on_bad_lines='skip', encoding='utf-8')
        except:
            df = pd.read_csv(csv_file, sep=";", encoding='latin-1')

        for _, row in df.iterrows():
            if 'Position géographique' in row and pd.notna(row['Position géographique']):
                try:
                    lat, lon = map(float, row['Position géographique'].split(','))
                    name = str(row['Nom'])
                    uic = str(row['Code(s) UIC'])

                    station_data = {
                        "uic": uic,
                        "name": name,
                        "point": Point(lon, lat),
                        "coords": (lat, lon)
                    }
                    self.stations.append(station_data)
                    self.graph.add_node(uic, name=name, pos=(lat, lon))
                except:
                    continue

    def map_lines(self, lignes_csv):
        print("   - Découpage des géométries de lignes (LGV prises en compte)...", file=sys.stderr)
        try:
            df_lignes = pd.read_csv(lignes_csv, sep=";", on_bad_lines='skip', encoding='utf-8')
        except:
            df_lignes = pd.read_csv(lignes_csv, sep=";", encoding='latin-1')

        TOLERANCE_DEG = 0.002

        for _, row in df_lignes.iterrows():
            try:
                geo_shape = json.loads(row['Geo Shape'])
                line_obj = shape(geo_shape)
                line_code = str(row['CODE_LIGNE'])
                line_name = str(row.get('LIB_LIGNE', '')).upper()

                is_lgv = "LGV" in line_name or "VITESSE" in line_name

                minx, miny, maxx, maxy = line_obj.bounds
                possible_stations = [s for s in self.stations if minx - TOLERANCE_DEG <= s['point'].x <= maxx + TOLERANCE_DEG and miny - TOLERANCE_DEG <= s['point'].y <= maxy + TOLERANCE_DEG]

                stations_on_line = []
                for s in possible_stations:
                    dist_proj = line_obj.project(s['point'])
                    if line_obj.distance(s['point']) < TOLERANCE_DEG:
                        stations_on_line.append((s, dist_proj))

                stations_on_line.sort(key=lambda x: x[1])

                for i in range(len(stations_on_line) - 1):
                    s1, dist1 = stations_on_line[i]
                    s2, dist2 = stations_on_line[i+1]

                    if dist1 != dist2:
                        segment = substring(line_obj, dist1, dist2)
                        curve_coords = [[y, x] for x, y in segment.coords]
                    else:
                        curve_coords = [s1['coords'], s2['coords']]

                    dist_km = geodesic(s1['coords'], s2['coords']).km
                    weight = dist_km / 3.0 if is_lgv else dist_km

                    self.graph.add_edge(
                        s1['uic'],
                        s2['uic'],
                        weight=weight,
                        line=line_code,
                        type="TRAIN",
                        geometry=curve_coords
                    )
                    
            except Exception:
                continue

    def add_city_transfers(self):
        """Relie les gares d'une même grande ville"""
        hubs = ["Paris", "Lyon", "Lille", "Marseille", "Bordeaux", "Nantes"]
        for hub in hubs:
            candidates = []
            for uic, data in self.graph.nodes(data=True):
                if hub.lower() in data['name'].lower():
                    candidates.append(uic)
            
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    u1, u2 = candidates[i], candidates[j]
                    d1 = self.graph.nodes[u1]['pos']
                    d2 = self.graph.nodes[u2]['pos']
                    dist = geodesic(d1, d2).km

                    if dist < 8:
                        self.graph.add_edge(u1, u2, weight=15, line="TRANSFERT", type="WALK")

    def ensure_connectivity(self):
        orphans = [n for n in self.graph.nodes if self.graph.degree(n) == 0]
        pass

    def get_path(self, dep_name, dest_name):
        start_uic = self._find_uic_by_name(dep_name)
        end_uic = self._find_uic_by_name(dest_name)

        if not start_uic: return None, f"Départ introuvable: {dep_name}", None
        if not end_uic: return None, f"Arrivée introuvable: {dest_name}", None

        try:
            full_path_uics = nx.shortest_path(self.graph, start_uic, end_uic, weight="weight")
            simplified_names = self._simplify_path(full_path_uics)

            return simplified_names, None, full_path_uics
        
        except nx.NetworkXNoPath:
            return None, "Aucun chemin trouvé", None

    def _simplify_path(self, path_uics):
        if not path_uics: return []
        full_names = [self.graph.nodes[u]['name'] for u in path_uics]
        final_stops = [full_names[0]]
        prev_line = None

        for i in range(len(path_uics) - 1):
            u, v = path_uics[i], path_uics[i+1]
            edge_data = self.graph.get_edge_data(u, v)[0]
            curr_line = edge_data.get('line', 'UNKNOWN')
            u_name = self.graph.nodes[u]['name']

            is_hub = any(x in u_name for x in ["Paris", "Lyon", "Lille", "Bordeaux", "Marseille"])
            
            if (prev_line and curr_line != prev_line) or (is_hub and "TGV" in u_name):
                if final_stops[-1] != u_name:
                    final_stops.append(u_name)
            prev_line = curr_line

        if final_stops[-1] != full_names[-1]:
            final_stops.append(full_names[-1])
        return final_stops

    def _find_uic_by_name(self, search_name):
        if not search_name: return None
        s = search_name.lower().strip()
        for uic, data in self.graph.nodes(data=True):
            if data['name'].lower() == s: return uic
        for uic, data in self.graph.nodes(data=True):
            if s in data['name'].lower(): return uic
        return None