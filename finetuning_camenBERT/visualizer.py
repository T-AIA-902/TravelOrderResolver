import folium
import webbrowser
import os
import sys

class MapVisualizer:
    def __init__(self, graph_engine):
        self.graph_engine = graph_engine
        self.map = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles="cartodbpositron")

    def draw_path(self, full_path_uics, filename="trajet.html"):
        print(f"   🎨 Génération de la carte interactive...", file=sys.stderr)

        all_coords = []

        for i in range(len(full_path_uics) - 1):
            u = full_path_uics[i]
            v = full_path_uics[i+1]

            edge_data = self.graph_engine.graph.get_edge_data(u, v)[0]

            if 'geometry' in edge_data:
                segment_coords = edge_data['geometry']
                color = "red"
                weight = 4
                dash_array = None
            else:
                p1 = self.graph_engine.graph.nodes[u]['pos']
                p2 = self.graph_engine.graph.nodes[v]['pos']
                segment_coords = [p1, p2]
                color = "blue"
                weight = 3
                dash_array = "5, 5"
            
            folium.PolyLine(
                segment_coords,
                color=color,
                weight=weight,
                opacity=0.8,
                dash_array=dash_array
            ).add_to(self.map)

            all_coords.extend(segment_coords)

        start_uic, end_uic = full_path_uics[0], full_path_uics[-1]
        for uic, label, col in [(start_uic, "Départ", "green"), (end_uic, "Arrivée", "black")]:
            pos = self.graph_engine.graph.nodes[uic]['pos']
            name = self.graph_engine.graph.nodes[uic]['name']
            folium.Marker(
                location=pos, 
                popup=f"<b>{label}</b><br>{name}",
                icon=folium.Icon(color=col, icon="train", prefix="fa")
            ).add_to(self.map)

        if all_coords:
            self.map.fit_bounds(all_coords)

        try:
            self.map.save(filename)
            abs_path = os.path.abspath(filename)
            file_url = f"file://{abs_path}"
            
            print(f"   ✅ Carte sauvegardée : {abs_path}", file=sys.stderr)
            webbrowser.open(file_url)
        except Exception as e:
            print(f"   ❌ Erreur d'ouverture carte : {e}", file=sys.stderr)