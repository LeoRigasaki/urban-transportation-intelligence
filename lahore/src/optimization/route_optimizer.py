"""
Route Optimizer Module for Lahore Traffic Intelligence System.
Implements Dijkstra and A* algorithms for shortest path finding.
"""
import networkx as nx
import pickle
import logging
import math
from typing import Tuple, List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in meters).
    Used as heuristic for A* algorithm.
    """
    R = 6371000  # Earth's radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


class RouteOptimizer:
    """
    A unified route optimizer supporting multiple algorithms.
    Loads a NetworkX graph and provides methods for pathfinding.
    """

    def __init__(self, graph_path: str = "lahore/models/trained/lahore_graph.pickle"):
        """
        Initialize the optimizer by loading the graph.

        Args:
            graph_path: Path to the pickled NetworkX graph file.
        """
        self.graph_path = graph_path
        self.graph: Optional[nx.MultiDiGraph] = None
        self._load_graph()

    def _load_graph(self) -> None:
        """Load the graph from the pickle file."""
        try:
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except FileNotFoundError:
            logger.error(f"❌ Graph file not found at {self.graph_path}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load graph: {e}")
            raise

    def _get_edge_weight(self, u: int, v: int, data: Dict[str, Any], weight_attr: str) -> float:
        """
        Get edge weight, handling missing attributes gracefully.
        Defaults to 1.0 if attribute is missing.
        """
        return data.get(weight_attr, 1.0) if data.get(weight_attr) is not None else 1.0

    def get_shortest_path_dijkstra(
        self,
        source: int,
        target: int,
        weight: str = "length"
    ) -> Tuple[Optional[List[int]], float]:
        """
        Find the shortest path using Dijkstra's algorithm.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge attribute to use as weight (e.g., 'length', 'travel_time').

        Returns:
            Tuple of (path as list of node IDs, total path cost). Returns (None, inf) if no path.
        """
        try:
            path = nx.dijkstra_path(self.graph, source, target, weight=weight)
            cost = nx.dijkstra_path_length(self.graph, source, target, weight=weight)
            logger.info(f"Dijkstra: Found path with {len(path)} nodes, cost: {cost:.2f}")
            return path, cost
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {source} and {target}")
            return None, float('inf')
        except nx.NodeNotFound as e:
            logger.error(f"Node not found: {e}")
            return None, float('inf')

    def get_shortest_path_astar(
        self,
        source: int,
        target: int,
        weight: str = "length"
    ) -> Tuple[Optional[List[int]], float]:
        """
        Find the shortest path using A* algorithm with haversine heuristic.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge attribute to use as weight.

        Returns:
            Tuple of (path as list of node IDs, total path cost). Returns (None, inf) if no path.
        """

        def heuristic(u: int, v: int) -> float:
            """Haversine distance heuristic for A*."""
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            return haversine_distance(u_data['x'], u_data['y'], v_data['x'], v_data['y'])

        try:
            path = nx.astar_path(self.graph, source, target, heuristic=heuristic, weight=weight)
            # Calculate actual path cost
            cost = sum(
                self._get_edge_weight(path[i], path[i + 1], self.graph[path[i]][path[i + 1]][0], weight)
                for i in range(len(path) - 1)
            )
            logger.info(f"A*: Found path with {len(path)} nodes, cost: {cost:.2f}")
            return path, cost
        except nx.NetworkXNoPath:
            logger.warning(f"No path found between {source} and {target}")
            return None, float('inf')
        except nx.NodeNotFound as e:
            logger.error(f"Node not found: {e}")
            return None, float('inf')

    def update_edge_weights(self, congestion_data: Dict[Tuple[int, int], float]) -> None:
        """
        Update edge weights based on congestion data.
        Creates a new 'travel_time' attribute based on length and congestion factor.

        Args:
            congestion_data: Dictionary mapping (u, v) edge tuples to congestion multipliers.
                             A multiplier of 1.0 means free flow, >1.0 means congested.
        """
        updated_count = 0
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            base_length = data.get('length', 100)  # Default length if missing
            congestion_factor = congestion_data.get((u, v), 1.0)
            # Travel time = base_length * congestion_factor (simplified model)
            self.graph[u][v][key]['travel_time'] = base_length * congestion_factor
            updated_count += 1

        logger.info(f"✅ Updated travel_time for {updated_count} edges based on congestion data.")

    def get_random_node_pair(self) -> Tuple[int, int]:
        """Get a random pair of nodes for testing."""
        import random
        nodes = list(self.graph.nodes())
        source = random.choice(nodes)
        target = random.choice(nodes)
        while target == source:
            target = random.choice(nodes)
        return source, target


if __name__ == "__main__":
    # Quick test
    optimizer = RouteOptimizer()
    source, target = optimizer.get_random_node_pair()
    logger.info(f"Testing route from {source} to {target}")

    path_dij, cost_dij = optimizer.get_shortest_path_dijkstra(source, target)
    path_astar, cost_astar = optimizer.get_shortest_path_astar(source, target)

    if path_dij:
        logger.info(f"✅ Dijkstra path length: {len(path_dij)}, cost: {cost_dij:.2f}")
    if path_astar:
        logger.info(f"✅ A* path length: {len(path_astar)}, cost: {cost_astar:.2f}")
