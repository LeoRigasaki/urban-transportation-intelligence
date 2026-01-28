"""
Anomaly Detection for Traffic Patterns.
Uses statistical methods and Isolation Forest to detect unusual traffic events.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: The amount of contamination of the data set, 
                           i.e. the proportion of outliers in the data set.
        """
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_fitted = False

    def detect_outliers_ml(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Use Isolation Forest to detect anomalies based on multiple features.
        
        Args:
            data: DataFrame containing traffic data.
            features: List of column names to use for detection (e.g., ['speed', 'volume']).
            
        Returns:
            Boolean array where True indicates an anomaly.
        """
        if data.empty:
            return np.array([])
            
        X = data[features].fillna(0)
        self.model.fit(X)
        self.is_fitted = True
        
        # IsolationForest returns -1 for outliers and 1 for inliers
        predictions = self.model.predict(X)
        return predictions == -1

    def detect_outliers_statistical(self, data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """
        Use Z-score to detect anomalies in a single dimension (e.g., speed drop).
        
        Args:
            data: Series of traffic metrics.
            threshold: Z-score threshold for identifying anomalies.
            
        Returns:
            Boolean array where True indicates an anomaly.
        """
        if data.empty or data.std() == 0:
            return np.zeros(len(data), dtype=bool)
            
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold

    def mark_incidents(self, graph, anomaly_results: pd.DataFrame):
        """
        Update the networkx graph with incident flags where anomalies are detected.
        
        Args:
            graph: The road network graph.
            anomaly_results: DataFrame with 'edge_id' (or 'u', 'v', 'key') and 'is_anomaly'.
        """
        logger.info(f"Marking {anomaly_results['is_anomaly'].sum()} incidents on the graph.")
        
        for _, row in anomaly_results[anomaly_results['is_anomaly']].iterrows():
            u, v = int(row['u']), int(row['v'])
            key = row.get('key', 0)
            
            if graph.has_edge(u, v, key):
                graph[u][v][key]['incident'] = True
                graph[u][v][key]['anomaly_score'] = row.get('anomaly_score', 1.0)

    def analyze_incidents(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive anomaly analysis.
        """
        logger.info("Running comprehensive anomaly analysis...")
        
        # Machine Learning based detection
        features = ['speed', 'volume', 'congestion_level']
        available_features = [f for f in features if f in data.columns]
        
        if len(available_features) > 1:
            data['is_anomaly_ml'] = self.detect_outliers_ml(data, available_features)
        else:
            data['is_anomaly_ml'] = False
            
        # Statistical detection for speed drops (usually more indicative of accidents)
        if 'speed' in data.columns:
            data['is_anomaly_stat'] = self.detect_outliers_statistical(data['speed'])
        else:
            data['is_anomaly_stat'] = False
            
        # Combine (priority to statistical as it's often more interpretable for traffic)
        data['is_anomaly'] = data['is_anomaly_ml'] | data['is_anomaly_stat']
        
        return data
