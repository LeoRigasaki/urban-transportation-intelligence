"""
Congestion-Aware Router for Lahore Traffic Intelligence System.
Integrates traffic prediction data with routing algorithms.
"""
import logging
import random
from typing import Dict, Tuple, List, Optional
import pickle
import networkx as nx

from lahore.src.optimization.route_optimizer import RouteOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CongestionRouter:
    """
    A congestion-aware routing engine that dynamically adjusts paths
    based on real-time or predicted traffic conditions.
    """

    def __init__(self, graph_path: str = "lahore/models/trained/lahore_graph.pickle"):
        """
        Initialize the congestion-aware router.

        Args:
            graph_path: Path to the pickled NetworkX graph.
        """
        self.optimizer = RouteOptimizer(graph_path)
        self.congestion_data: Dict[Tuple[int, int], float] = {}

    def simulate_congestion(self, congestion_percentage: float = 0.1, severity_range: Tuple[float, float] = (1.5, 3.0)) -> None:
        """
        Simulate congestion on a percentage of edges.
        Used for testing when real traffic data is unavailable.

        Args:
            congestion_percentage: Fraction of edges to mark as congested (0-1).
            severity_range: Range of congestion multipliers (min, max).
        """
        edges = list(self.optimizer.graph.edges())
        num_congested = int(len(edges) * congestion_percentage)
        congested_edges = random.sample(edges, min(num_congested, len(edges)))

        self.congestion_data = {}
        for u, v in congested_edges:
            severity = random.uniform(*severity_range)
            self.congestion_data[(u, v)] = severity

        # Update graph weights
        self.optimizer.update_edge_weights(self.congestion_data)
        logger.info(f"✅ Simulated congestion on {len(self.congestion_data)} edges (severity: {severity_range})")

    def load_congestion_from_redis(self, redis_host: str = "localhost", redis_port: int = 6379) -> None:
        """
        Load real-time congestion data from Redis cache.
        Expected Redis format: key = "congestion:{u}:{v}", value = float multiplier.
        """
        try:
            import redis
            r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

            self.congestion_data = {}
            keys = r.keys("congestion:*")
            for key in keys:
                parts = key.split(":")
                if len(parts) == 3:
                    u, v = int(parts[1]), int(parts[2])
                    self.congestion_data[(u, v)] = float(r.get(key))

            self.optimizer.update_edge_weights(self.congestion_data)
            logger.info(f"✅ Loaded {len(self.congestion_data)} congestion entries from Redis")

        except ImportError:
            logger.warning("Redis not available, using simulated congestion")
            self.simulate_congestion()
        except Exception as e:
            logger.error(f"Failed to load from Redis: {e}")
            self.simulate_congestion()

    def get_optimal_route(
        self,
        source: int,
        target: int,
        algorithm: str = "astar",
        use_congestion: bool = True
    ) -> Tuple[Optional[List[int]], float, Dict]:
        """
        Get the optimal route considering current traffic conditions.

        Args:
            source: Source node ID.
            target: Target node ID.
            algorithm: Routing algorithm ('dijkstra' or 'astar').
            use_congestion: If True, use travel_time (with congestion); else use length.

        Returns:
            Tuple of (path, cost, metadata)
        """
        weight = 'travel_time' if use_congestion else 'length'

        if algorithm == 'dijkstra':
            path, cost = self.optimizer.get_shortest_path_dijkstra(source, target, weight=weight)
        else:
            path, cost = self.optimizer.get_shortest_path_astar(source, target, weight=weight)

        metadata = {
            'algorithm': algorithm,
            'weight_used': weight,
            'congestion_aware': use_congestion,
            'path_nodes': len(path) if path else 0
        }

        return path, cost, metadata

    def compare_routes(
        self,
        source: int,
        target: int
    ) -> Dict:
        """
        Compare routes with and without congestion awareness.

        Returns:
            Dictionary with comparison results.
        """
        # Static route (no congestion consideration)
        static_path, static_cost = self.optimizer.get_shortest_path_dijkstra(source, target, weight='length')

        # Dynamic route (with congestion)
        dynamic_path, dynamic_cost = self.optimizer.get_shortest_path_astar(source, target, weight='travel_time')

        # Calculate actual travel time for both routes
        static_travel_time = 0
        if static_path:
            for i in range(len(static_path) - 1):
                edge_data = self.optimizer.graph[static_path[i]][static_path[i + 1]][0]
                static_travel_time += edge_data.get('travel_time', edge_data.get('length', 100))

        result = {
            'static_route': {
                'path_length': len(static_path) if static_path else 0,
                'distance': static_cost,
                'actual_travel_time': static_travel_time
            },
            'congestion_aware_route': {
                'path_length': len(dynamic_path) if dynamic_path else 0,
                'travel_time': dynamic_cost
            },
            'paths_different': static_path != dynamic_path if (static_path and dynamic_path) else False,
            'time_saved': static_travel_time - dynamic_cost if (static_path and dynamic_path) else 0
        }

        return result


if __name__ == "__main__":
    router = CongestionRouter()

    # Simulate some congestion
    router.simulate_congestion(congestion_percentage=0.15, severity_range=(2.0, 4.0))

    # Get random node pair
    source, target = router.optimizer.get_random_node_pair()
    logger.info(f"Testing congestion-aware routing: {source} -> {target}")

    # Compare routes
    comparison = router.compare_routes(source, target)

    logger.info("=" * 50)
    logger.info("Route Comparison Results:")
    logger.info(f"  Static Route: {comparison['static_route']}")
    logger.info(f"  Congestion-Aware Route: {comparison['congestion_aware_route']}")
    logger.info(f"  Routes Different: {comparison['paths_different']}")
    logger.info(f"  Estimated Time Saved: {comparison['time_saved']:.2f}")
