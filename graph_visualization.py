import os
from typing import Dict, Any, List
from pyvis.network import Network
from dataclasses import dataclass
from graph_storage import GraphStorage


@dataclass
class VisualizationConfig:
    """Visualization configuration"""

    height: str = "100vh"
    width: str = "100%"
    bgcolor: str = "#f8f9fa"
    font_color: str = "#2c3e50"
    node_color: str = "#3498db"
    edge_color: str = "#e74c3c"
    alias_color: str = "#9b59b6"
    default_node_size: int = 30
    min_node_size: int = 20
    max_node_size: int = 60


class GraphVisualization:
    """GraphVisualization manager"""

    def __init__(self, storage: GraphStorage):
        """Initialize the visualization manager"""
        self.storage: GraphStorage = storage
        self.config = VisualizationConfig()
        self.base_path = storage.base_path

    def visualize(self) -> None:
        """Base visualization"""
        try:
            net = self._create_network()
            self._add_base_nodes_and_edges(net)
            self._save_visualization(net, "graph.html")
        except Exception as e:
            print(f"[ERROR] Failed to create the base visualization: {str(e)}")

    def visualize_communities(self) -> None:
        """Visualize the communities"""
        try:
            if not hasattr(self.storage, "communities") or not self.storage.communities:
                print("Community data not found.")
                return

            net = self._create_network()
            colors = self._generate_colors(len(self.storage.communities))
            node_info = self._create_community_mapping(colors)
            self._add_community_nodes_and_edges(net, node_info)
            self._save_visualization(net, "communities.html")
        except Exception as e:
            print(f"[ERROR] Failed to create the community visualization: {str(e)}")

    def _create_network(self) -> Network:
        """Create the network instance and set up the physics parameters"""
        net = Network(
            height=self.config.height,
            width=self.config.width,
            bgcolor=self.config.bgcolor,
            font_color=self.config.font_color,
            directed=True,
        )

        # Add complete physics and interaction configuration
        net.set_options(
            """{
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -200,
                    "centralGravity": 0.03,
                    "springLength": 250,
                    "springConstant": 0.05,
                    "damping": 0.97,
                    "avoidOverlap": 1
                },
                "stabilization": {
                    "enabled": true,
                    "iterations": 2000,
                    "updateInterval": 50
                }
            },
            "layout": {
                "improvedLayout": true,
                "hierarchical": {
                    "enabled": false
                }
            },
            "interaction": {
                "hover": true,
                "hoverConnectedEdges": true,
                "selectable": true,
                "selectConnectedEdges": true,
                "multiselect": true,
                "dragNodes": true,
                "dragView": true,
                "zoomView": true,
                "navigationButtons": true,
                "hideEdgesOnDrag": false,
                "hideEdgesOnZoom": false,
                "keyboard": {
                    "enabled": true,
                    "speed": {"x": 10, "y": 10, "zoom": 0.1}
                }
            },
            "edges": {
                "smooth": {
                    "type": "continuous",
                    "forceDirection": "none",
                    "roundness": 0.5
                },
                "color": {
                    "inherit": false,
                    "color": "#e74c3c",
                    "highlight": "#f1c40f",
                    "hover": "#f39c12"
                },
                "width": 2,
                "selectionWidth": 3,
                "hoverWidth": 2,
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                }
            },
            "nodes": {
                "shape": "dot",
                "font": {
                    "size": 14,
                    "face": "Arial",
                    "strokeWidth": 2,
                    "strokeColor": "#ffffff"
                },
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "color": {
                    "border": "#2c3e50",
                    "background": "#3498db",
                    "highlight": {
                        "border": "#2c3e50",
                        "background": "#e74c3c"
                    },
                    "hover": {
                        "border": "#2c3e50",
                        "background": "#e74c3c"
                    }
                }
            }
        }"""
        )

        return net

    def _calculate_node_weights(self) -> Dict[str, int]:
        """Calculate node weights (sum of in and out degrees)"""
        weights = {}
        for node in self.storage.graph.nodes():
            out_degree = len(list(self.storage.graph.successors(node)))
            in_degree = len(list(self.storage.graph.predecessors(node)))
            weights[node] = out_degree + in_degree
        return weights

    def _add_base_nodes_and_edges(self, net: Network) -> None:
        """Add base nodes, alias nodes, and edges"""
        try:
            # Calculate node weights and normalize
            weights = self._calculate_node_weights()
            max_weight = max(weights.values()) if weights else 1
            size_range = self.config.max_node_size - self.config.min_node_size

            # Add main entity nodes
            for node in self.storage.graph.nodes():
                weight = weights.get(node, 0)
                size = self.config.min_node_size + (weight / max_weight) * size_range
                net.add_node(
                    str(node),
                    label=str(node),
                    title=f"Entity: {node}\nNumber of relationships: {weight}",
                    color=self.config.node_color,
                    size=size,
                )

            # Add alias nodes and relationships
            for main_id, aliases in self.storage.entity_aliases.items():
                real_aliases = [alias for alias in aliases if alias != main_id]
                for alias in real_aliases:
                    net.add_node(
                        str(alias),
                        label=str(alias),
                        title=f"Alias: {alias}\nMain entity: {main_id}",
                        color=self.config.alias_color,
                        size=self.config.min_node_size * 0.8,
                    )
                    net.add_edge(
                        str(alias),
                        str(main_id),
                        color=self.config.edge_color,
                        dashes=True,
                        title="Alias",
                    )

            # Add relationship edges
            for source, target, data in self.storage.graph.edges(data=True):
                if source != target:
                    net.add_edge(
                        str(source),
                        str(target),
                        title=str(data.get("type", "relationship")),
                        color=self.config.edge_color,
                    )
        except Exception as e:
            print(f"[ERROR] Failed to add nodes and edges: {str(e)}")
            raise

    def _generate_colors(self, n: int) -> List[str]:
        """Generate community colors"""
        colors = []
        for i in range(n):
            hue = i * (360 / n)
            s, l = 70, 45  # saturation and lightness
            r, g, b = self._hsl_to_rgb(hue / 360, s / 100, l / 100)
            colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
        return colors

    def _hsl_to_rgb(self, h: float, s: float, l: float) -> tuple:
        """HSL color to RGB"""

        def hue_to_rgb(p: float, q: float, t: float) -> float:
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1 / 6:
                return p + (q - p) * 6 * t
            if t < 1 / 2:
                return q
            if t < 2 / 3:
                return p + (q - p) * (2 / 3 - t) * 6
            return p

        if s == 0:
            return (l, l, l)

        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q

        return (
            hue_to_rgb(p, q, h + 1 / 3),
            hue_to_rgb(p, q, h),
            hue_to_rgb(p, q, h - 1 / 3),
        )

    def _create_community_mapping(self, colors: List[str]) -> Dict:
        """Create a mapping from nodes to communities"""
        mapping = {}
        try:
            weights = self._calculate_node_weights()
            max_weight = max(weights.values()) if weights else 1
            size_range = self.config.max_node_size - self.config.min_node_size

            for comm_id, (color, comm_data) in zip(
                self.storage.communities.keys(),
                zip(colors, self.storage.communities.values()),
            ):
                for node in comm_data["members"]:
                    weight = weights.get(node, 0)
                    size = (
                        self.config.min_node_size + (weight / max_weight) * size_range
                    )
                    mapping[str(node)] = {
                        "community": comm_id,
                        "color": color,
                        "size": size,
                    }
        except Exception as e:
            print(f"[ERROR] Failed to create a community mapping: {str(e)}")
        return mapping

    def _add_community_nodes_and_edges(self, net: Network, node_info: Dict) -> None:
        """Add community nodes and edges"""
        try:
            # Add nodes
            for node in self.storage.graph.nodes():
                info = node_info.get(
                    str(node), {"color": "#d3d3d3", "size": self.config.min_node_size}
                )
                net.add_node(
                    str(node),
                    label=str(node),
                    title=f"Entity: {node}\nCommunity: {info.get('community', 'Uncategorized')}",
                    color=info["color"],
                    size=info["size"],
                )

            # Add edges
            for source, target, data in self.storage.graph.edges(data=True):
                if source != target:
                    source, target = str(source), str(target)
                    if source in node_info and target in node_info:
                        # All edges within the same community use the community color
                        if node_info[source].get("community") == node_info[target].get(
                            "community"
                        ):
                            color = node_info[source]["color"]
                        else:
                            color = "#999999"  # Use gray for inter-community edges
                    else:
                        color = "#999999"
                    net.add_edge(source, target, color=color)
        except Exception as e:
            print(f"[ERROR] Failed to create community edges and edges: {str(e)}")
            raise

    def _save_visualization(self, net: Network, filename: str) -> None:
        """Save visualization to an HTML file"""
        try:
            path = os.path.join(self.base_path, filename)
            net.save_graph(path)
            print(f"Visualization results have been saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save visualization: {str(e)}")
            raise
