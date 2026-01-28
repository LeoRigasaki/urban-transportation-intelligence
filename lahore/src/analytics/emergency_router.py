"""
Emergency Route Planning for Lahore.
Prioritizes major roads and boulevards for emergency vehicles (Ambulances, Fire Trucks).
"""
import networkx as nx
import logging
from typing import List, Optional, Tuple, Dict, Any
from lahore.src.optimization.route_optimizer import RouteOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmergencyRouter(RouteOptimizer):
    def __init__(self, graph_path: str = "lahore/models/trained/lahore_graph.pickle"):
        super().__init__(graph_path)
        # Highway types priority for emergency vehicles
        # Lower values = higher priority/easier to navigate
        self.road_priority = {
            'motorway': 0.5,
            'trunk': 0.6,
            'primary': 0.8,
            'secondary': 1.0,
            'tertiary': 2.0,
            'residential': 5.0,
            'living_street': 10.0,
            'pedestrian': 50.0,
            'service': 5.0
        }

    def _calculate_emergency_weight(self, u: int, v: int, data: Dict[str, Any]) -> float:
        """
        Custom weight function for emergency vehicles.
        Favors major roads and penalizes small narrow streets.
        """
        # If it's a MultiDiGraph data from certain nx functions might be a dict-of-dicts
        # but usually it's just the edge data dict for the specific edge being evaluated.
        if 0 in data and isinstance(data[0], dict):
            data = data[0] # Default to first key if it's a multi-edge dict
        
        length = data.get('length', 1.0)
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list): highway = highway[0]
        
        # Get priority multiplier
        priority = self.road_priority.get(highway, 3.0)
        
        # Consider congestion but less than a normal vehicle (emergency can bypass some traffic)
        # However, gridlocked roads are still slow
        congestion = data.get('congestion_level', 1.0)
        effective_congestion = 1.0 + (congestion - 1.0) * 0.7 
        
        # Lane count bonus (more lanes = easier to move through traffic)
        lanes = data.get('lanes', 1)
        if isinstance(lanes, list): lanes = int(lanes[0])
        else: lanes = int(lanes) if str(lanes).isdigit() else 1
        
        lane_bonus = 1.0 / (1.0 + (lanes - 1) * 0.2)
        
        return length * priority * effective_congestion * lane_bonus

    def get_emergency_route(self, source: int, target: int) -> Tuple[Optional[List[int]], float]:
        """
        Find the optimal route for an emergency vehicle.
        """
        logger.info(f"🚑 Calculating emergency route from {source} to {target}...")
        
        try:
            path = nx.shortest_path(
                self.graph, 
                source=source, 
                target=target, 
                weight=self._calculate_emergency_weight
            )
            
            # Calculate actual distance
            total_dist = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                data = self.graph[u][v][0] # Simple path for demo
                total_dist += data.get('length', 0)
                
            return path, total_dist
        except nx.NetworkXNoPath:
            logger.error("No path found for emergency vehicle!")
            return None, 0.0

    def compare_with_standard(self, source: int, target: int) -> Dict[str, Any]:
        """Compare emergency route with a standard shortest-distance route."""
        # Standard route
        std_path, std_dist = self.get_shortest_path_dijkstra(source, target, weight='length')
        
        # Emergency route
        emg_path, emg_dist = self.get_emergency_route(source, target)
        
        return {
            "standard": {"path": std_path, "distance": std_dist},
            "emergency": {"path": emg_path, "distance": emg_dist},
            "paths_different": std_path != emg_path
        }
