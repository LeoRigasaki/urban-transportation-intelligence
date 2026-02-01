"""
Streaming Feature Extractor for Real-Time Traffic Analytics.
Provides utilities for computing features from windowed traffic data.
"""
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import deque
from dataclasses import dataclass, field
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EdgeState:
    """Maintains sliding window state for a single road segment."""
    u: int
    v: int
    key: int
    speeds: deque = field(default_factory=lambda: deque(maxlen=100))
    volumes: deque = field(default_factory=lambda: deque(maxlen=100))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: float = 0.0

class StreamingFeatureExtractor:
    """
    Extracts real-time features from streaming traffic data.
    Maintains sliding window state for each road segment.
    """
    
    def __init__(self, window_size: int = 100, max_age_seconds: int = 600):
        """
        Args:
            window_size: Maximum number of updates to keep per edge.
            max_age_seconds: Evict edges not updated within this time.
        """
        self.window_size = window_size
        self.max_age_seconds = max_age_seconds
        self.edge_states: Dict[str, EdgeState] = {}
        
    def _get_edge_key(self, u: int, v: int, key: int) -> str:
        return f"{u}_{v}_{key}"
    
    def update(self, u: int, v: int, key: int, speed: float, volume: int, timestamp: float) -> Dict[str, Any]:
        """
        Update state with a new traffic reading and return computed features.
        """
        edge_key = self._get_edge_key(u, v, key)
        
        if edge_key not in self.edge_states:
            self.edge_states[edge_key] = EdgeState(u=u, v=v, key=key)
        
        state = self.edge_states[edge_key]
        state.speeds.append(speed)
        state.volumes.append(volume)
        state.timestamps.append(timestamp)
        state.last_update = time.time()
        
        return self.compute_features(edge_key)
    
    def compute_features(self, edge_key: str) -> Dict[str, Any]:
        """
        Compute features for a specific edge.
        """
        if edge_key not in self.edge_states:
            return {}
        
        state = self.edge_states[edge_key]
        speeds = list(state.speeds)
        volumes = list(state.volumes)
        
        if len(speeds) == 0:
            return {}
        
        # Basic statistics
        avg_speed = np.mean(speeds)
        std_speed = np.std(speeds) if len(speeds) > 1 else 0.0
        min_speed = np.min(speeds)
        max_speed = np.max(speeds)
        
        # Volume metrics
        total_volume = sum(volumes)
        avg_volume = np.mean(volumes)
        
        # Congestion index (0 = free flow, 1 = gridlock)
        congestion_index = max(0.0, min(1.0, 1.0 - (avg_speed / 60.0)))
        
        # Speed trend (positive = improving, negative = worsening)
        if len(speeds) >= 5:
            recent = np.mean(speeds[-5:])
            older = np.mean(speeds[:-5]) if len(speeds) > 5 else recent
            speed_trend = (recent - older) / (older + 0.1)
        else:
            speed_trend = 0.0
        
        # Volatility (coefficient of variation)
        volatility = std_speed / (avg_speed + 0.1)
        
        return {
            'u': state.u,
            'v': state.v,
            'key': state.key,
            'avg_speed': round(avg_speed, 2),
            'std_speed': round(std_speed, 2),
            'min_speed': round(min_speed, 2),
            'max_speed': round(max_speed, 2),
            'total_volume': total_volume,
            'avg_volume': round(avg_volume, 2),
            'congestion_index': round(congestion_index, 3),
            'speed_trend': round(speed_trend, 3),
            'volatility': round(volatility, 3),
            'sample_count': len(speeds),
            'last_update': state.last_update
        }
    
    def get_all_features(self) -> List[Dict[str, Any]]:
        """
        Get features for all tracked edges.
        """
        features = []
        for edge_key in self.edge_states:
            f = self.compute_features(edge_key)
            if f:
                features.append(f)
        return features
    
    def evict_stale(self) -> int:
        """
        Remove edges that haven't been updated recently.
        Returns the number of evicted edges.
        """
        current_time = time.time()
        stale_keys = [
            k for k, v in self.edge_states.items()
            if (current_time - v.last_update) > self.max_age_seconds
        ]
        for k in stale_keys:
            del self.edge_states[k]
        
        if stale_keys:
            logger.info(f"🗑️ Evicted {len(stale_keys)} stale edge states")
        
        return len(stale_keys)
    
    def get_congested_edges(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Get edges with congestion index above threshold.
        """
        all_features = self.get_all_features()
        return [f for f in all_features if f.get('congestion_index', 0) > threshold]
