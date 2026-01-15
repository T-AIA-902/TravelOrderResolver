"""
Interactive map visualization for railway routes.

Uses Folium to generate HTML maps showing train routes with
markers for departure and arrival stations.
"""

import os
import sys
import webbrowser
from typing import TYPE_CHECKING, List, Optional

import folium

if TYPE_CHECKING:
    from src.pathfinding.graph import TrainGraph


class MapVisualizer:
    """
    Generate interactive HTML maps of railway routes.

    Uses Folium to create maps with:
    - Red lines for train segments
    - Blue dashed lines for walking transfers
    - Green marker for departure
    - Black marker for arrival
    """

    def __init__(self, graph_engine: "TrainGraph") -> None:
        """
        Initialize the map visualizer.

        Args:
            graph_engine: TrainGraph instance for accessing station data
        """
        self.graph_engine = graph_engine
        self.map: Optional[folium.Map] = None

    def _create_map(self) -> folium.Map:
        """Create a new Folium map centered on France."""
        return folium.Map(
            location=[46.603354, 1.888334],
            zoom_start=6,
            tiles="cartodbpositron",
        )

    def draw_path(
        self,
        full_path_uics: List[str],
        filename: str = "trajet.html",
        open_browser: bool = True,
    ) -> str:
        """
        Draw a route on the map and save to HTML file.

        Args:
            full_path_uics: List of station UICs representing the route
            filename: Output HTML file name
            open_browser: Whether to open the map in browser

        Returns:
            Absolute path to the saved HTML file
        """
        print("   Generating interactive map...", file=sys.stderr)

        self.map = self._create_map()
        all_coords: List[List[float]] = []

        # Draw route segments
        for i in range(len(full_path_uics) - 1):
            u = full_path_uics[i]
            v = full_path_uics[i + 1]

            edge_data = self.graph_engine.graph.get_edge_data(u, v)[0]

            if "geometry" in edge_data:
                segment_coords = edge_data["geometry"]
                color = "red"
                weight = 4
                dash_array = None
            else:
                # Walking transfer (no geometry)
                p1 = self.graph_engine.graph.nodes[u]["pos"]
                p2 = self.graph_engine.graph.nodes[v]["pos"]
                segment_coords = [list(p1), list(p2)]
                color = "blue"
                weight = 3
                dash_array = "5, 5"

            folium.PolyLine(
                segment_coords,
                color=color,
                weight=weight,
                opacity=0.8,
                dash_array=dash_array,
            ).add_to(self.map)

            all_coords.extend(segment_coords)

        # Add markers for start and end stations
        start_uic, end_uic = full_path_uics[0], full_path_uics[-1]
        markers = [
            (start_uic, "Departure", "green"),
            (end_uic, "Arrival", "black"),
        ]

        for uic, label, col in markers:
            pos = self.graph_engine.graph.nodes[uic]["pos"]
            name = self.graph_engine.graph.nodes[uic]["name"]
            folium.Marker(
                location=pos,
                popup=f"<b>{label}</b><br>{name}",
                icon=folium.Icon(color=col, icon="train", prefix="fa"),
            ).add_to(self.map)

        # Fit map bounds to show entire route
        if all_coords:
            self.map.fit_bounds(all_coords)

        # Save and optionally open
        try:
            self.map.save(filename)
            abs_path = os.path.abspath(filename)

            print(f"   Map saved: {abs_path}", file=sys.stderr)

            if open_browser:
                file_url = f"file://{abs_path}"
                webbrowser.open(file_url)

            return abs_path

        except Exception as e:
            print(f"   Map error: {e}", file=sys.stderr)
            raise
