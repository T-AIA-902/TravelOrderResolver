import os
import sys
import tempfile
import webbrowser
from typing import TYPE_CHECKING, List, Optional, Tuple

import folium
from folium import FeatureGroup, LayerControl

if TYPE_CHECKING:
    from src.pathfinding.graph import TrainGraph
    from src.pathfinding.route_optimizer import RouteResult

_PARETO_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]


class MapVisualizer:
    def __init__(self, graph_engine: "TrainGraph") -> None:
        self.graph_engine = graph_engine

    def _create_map(self) -> folium.Map:
        return folium.Map(
            location=[46.603354, 1.888334],
            zoom_start=6,
            tiles="cartodbpositron",
        )

    def _get_node_pos(self, uic: str) -> Tuple[float, float]:
        return tuple(self.graph_engine.graph.nodes[uic]["pos"])  # type: ignore[return-value]

    def _get_node_name(self, uic: str) -> str:
        return str(self.graph_engine.graph.nodes[uic]["name"])

    def _draw_single_path(
        self,
        fmap: folium.Map,
        path_uics: List[str],
        color: str = "#e6194b",
        weight: int = 4,
        group: Optional[FeatureGroup] = None,
    ) -> List[List[float]]:
        target = group if group else fmap
        all_coords: List[List[float]] = []
        graph = self.graph_engine.graph

        for i in range(len(path_uics) - 1):
            u, v = path_uics[i], path_uics[i + 1]
            pos_u = self._get_node_pos(u)
            pos_v = self._get_node_pos(v)

            edge_data = graph.get_edge_data(u, v)
            if edge_data is None:
                continue
            edata = edge_data[0]

            seg_type = edata.get("type", "TRAIN")
            line_code = edata.get("line", "?")
            dist_km = edata.get("dist_km", 0)

            segment_coords = [list(pos_u), list(pos_v)]

            if seg_type == "WALK":
                style = {
                    "color": "#2196F3",
                    "weight": 3,
                    "dash_array": "8, 6",
                    "opacity": 0.7,
                }
                popup_text = f"Correspondance<br>{dist_km:.1f} km"
            else:
                style = {
                    "color": color,
                    "weight": weight,
                    "dash_array": None,
                    "opacity": 0.8,
                }
                popup_text = f"Ligne {line_code}<br>{dist_km:.1f} km"

            folium.PolyLine(
                segment_coords,
                color=style["color"],
                weight=style["weight"],
                opacity=style["opacity"],
                dash_array=style["dash_array"],
                popup=folium.Popup(popup_text, max_width=200),
            ).add_to(target)

            all_coords.extend(segment_coords)

        return all_coords

    def _add_marker(
        self,
        fmap: folium.Map,
        uic: str,
        label: str,
        color: str,
        icon: str = "train",
    ) -> None:
        pos = self._get_node_pos(uic)
        name = self._get_node_name(uic)
        folium.Marker(
            location=pos,
            popup=folium.Popup(
                f"<b>{label}</b><br>{name}<br><small>UIC: {uic}</small>",
                max_width=250,
            ),
            icon=folium.Icon(color=color, icon=icon, prefix="fa"),
        ).add_to(fmap)

    def _add_legend(self, fmap: folium.Map, entries: List[dict]) -> None:
        legend_items = ""
        for entry in entries:
            c = entry["color"]
            label = entry["label"]
            legend_items += (
                f'<li style="margin:4px 0">'
                f'<span style="background:{c};width:20px;height:3px;'
                f'display:inline-block;margin-right:6px;vertical-align:middle"></span>'
                f"{label}</li>"
            )

        legend_html = f"""
        <div style="
            position: fixed; bottom: 30px; left: 30px; z-index: 1000;
            background: white; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 12px;
            font-family: Arial, sans-serif; max-width: 320px;
        ">
            <b style="font-size:13px">Itineraires</b>
            <ul style="list-style:none;padding:4px 0;margin:0">{legend_items}</ul>
            <div style="margin-top:6px;color:#666">
                <span style="background:#2196F3;width:20px;height:3px;
                display:inline-block;margin-right:6px;vertical-align:middle;
                border-bottom:2px dashed #2196F3"></span>Correspondance
            </div>
        </div>
        """
        fmap.get_root().html.add_child(folium.Element(legend_html))

    def draw_path(
        self,
        full_path_uics: List[str],
        filename: Optional[str] = None,
        open_browser: bool = True,
    ) -> str:
        print("   Generating interactive map...", file=sys.stderr)

        fmap = self._create_map()
        all_coords = self._draw_single_path(fmap, full_path_uics)

        if len(full_path_uics) >= 2:
            self._add_marker(fmap, full_path_uics[0], "Depart", "green")
            self._add_marker(fmap, full_path_uics[-1], "Arrivee", "red")

        if all_coords:
            fmap.fit_bounds(all_coords)

        return self._save_map(fmap, filename, open_browser)

    def draw_route(
        self,
        route_result: "RouteResult",
        filename: Optional[str] = None,
        open_browser: bool = True,
    ) -> str:
        if not route_result.success:
            raise ValueError(f"Cannot visualize failed route: {route_result.error}")

        if route_result.pareto_paths and len(route_result.pareto_paths) > 1:
            return self._draw_pareto_paths(route_result, filename, open_browser)

        if route_result.full_uic_path:
            return self.draw_path(route_result.full_uic_path, filename, open_browser)

        raise ValueError("RouteResult has no path data to visualize")

    def _draw_pareto_paths(
        self,
        route_result: "RouteResult",
        filename: Optional[str],
        open_browser: bool,
    ) -> str:
        print("   Generating MOA* Pareto map...", file=sys.stderr)

        fmap = self._create_map()
        all_coords: List[List[float]] = []
        legend_entries: List[dict] = []
        pareto = route_result.pareto_paths or []

        for idx, path_result in enumerate(pareto):
            color = _PARETO_COLORS[idx % len(_PARETO_COLORS)]
            cost = path_result.cost

            label = (
                f"Chemin {idx + 1}: {cost.time:.1f}h, "
                f"{cost.distance:.0f}km, {cost.transfers} corresp."
            )

            group = FeatureGroup(name=label, show=(idx == 0))
            coords = self._draw_single_path(
                fmap,
                path_result.uic_path,
                color=color,
                weight=4 if idx == 0 else 3,
                group=group,
            )
            group.add_to(fmap)

            all_coords.extend(coords)
            legend_entries.append({"color": color, "label": label})

        if pareto:
            first_path = pareto[0].uic_path
            if len(first_path) >= 2:
                self._add_marker(fmap, first_path[0], "Depart", "green")
                self._add_marker(fmap, first_path[-1], "Arrivee", "red")

        LayerControl(collapsed=False).add_to(fmap)
        self._add_legend(fmap, legend_entries)

        if all_coords:
            fmap.fit_bounds(all_coords)

        return self._save_map(fmap, filename, open_browser)

    def _save_map(self, fmap: folium.Map, filename: Optional[str], open_browser: bool) -> str:
        if filename:
            fmap.save(filename)
            abs_path = os.path.abspath(filename)
        else:
            tmp = tempfile.NamedTemporaryFile(suffix=".html", prefix="carte_", delete=False)
            fmap.save(tmp.name)
            abs_path = tmp.name
            tmp.close()

        print(f"   Map saved: {abs_path}", file=sys.stderr)

        if open_browser:
            webbrowser.open(f"file://{abs_path}")

        return abs_path
