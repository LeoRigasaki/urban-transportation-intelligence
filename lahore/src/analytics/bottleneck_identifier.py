"""
Bottleneck Identification for Road Networks.
Identifies critical points and persistent congestion hotspots.
"""
import numpy as np
import pandas as pd
import networkx as nx
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BottleneckIdentifier:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def identify_structural_bottlenecks(self, k: int = 100) -> List[Tuple[int, int, Dict]]:
        """
        Estimate edge betweenness centrality to find critical links.
        We use a subset (k nodes) for large graphs to keep it computationally feasible.
        """
        logger.info(f"Calculating edge betweenness centrality using k={k} nodes...")
        # For huge graphs, we approximate
        centrality = nx.edge_betweenness_centrality(self.graph, k=k, weight='length')
        
        # Sort and return top edges
        sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_centrality

    def identify_congestion_hotspots(self, historical_traffic: pd.DataFrame) -> pd.DataFrame:
        """
        Find edges that are consistently congested.
        Expects historical_traffic with columns: ['u', 'v', 'key', 'congestion_level', 'timestamp']
        """
        logger.info("Identifying persistent congestion hotspots...")
        
        # Group by edge and calculate mean congestion
        hotspots = historical_traffic.groupby(['u', 'v', 'key'])['congestion_level'].agg(['mean', 'std', 'count']).reset_index()
        
        # Persistent bottleneck: High mean congestion, low variance (constantly bad)
        # Volatile bottleneck: High variance (unpredictable)
        hotspots['persistence_score'] = hotspots['mean'] / (hotspots['std'] + 0.1)
        
        return hotspots.sort_values(by='mean', ascending=False)

    def calculate_v_c_ratio(self, current_traffic: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume-to-Capacity ratio if capacity is known.
        If capacity isn't in graph, we estimate based on road type.
        """
        # Road type capacities (typical values)
        capacities = {
            'motorway': 2000,
            'trunk': 1500,
            'primary': 1000,
            'secondary': 800,
            'tertiary': 500,
            'residential': 200
        }
        
        logger.info("Calculating V/C ratios...")
        results = []
        for _, row in current_traffic.iterrows():
            u, v, key = int(row['u']), int(row['v']), row.get('key', 0)
            if self.graph.has_edge(u, v, key):
                data = self.graph[u][v][key]
                road_type = data.get('highway', 'unclassified')
                if isinstance(road_type, list): road_type = road_type[0]
                
                capacity = capacities.get(road_type, 300)
                vc_ratio = row['volume'] / capacity
                
                results.append({
                    'u': u, 'v': v, 'key': key,
                    'vc_ratio': vc_ratio,
                    'is_bottleneck': vc_ratio > 0.85
                })
        
        return pd.DataFrame(results)

    def get_top_critical_points(self, traffic_data: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Combined score of structural importance and current congestion.
        """
        # Get persistence
        hotspots = self.identify_congestion_hotspots(traffic_data)
        
        # Add road names for better reporting
        def get_name(u, v, key):
            if self.graph.has_edge(u, v, key):
                name = self.graph[u][v][key].get('name', 'Unnamed Road')
                return name if name else "Unnamed Road"
            return "N/A"
            
        hotspots['road_name'] = hotspots.apply(lambda r: get_name(r['u'], r['v'], r['key']), axis=1)
        
        return hotspots.head(top_n)
