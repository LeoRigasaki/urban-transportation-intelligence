"""
Verification Script for Day 6: Advanced Analytics.
Validates Anomaly Detection, Bottleneck Identification, and Emergency Routing.
"""
import pandas as pd
import numpy as np
import logging
import os
import random
from datetime import datetime, timedelta
from lahore.src.analytics.anomaly_detector import AnomalyDetector
from lahore.src.analytics.bottleneck_identifier import BottleneckIdentifier
from lahore.src.analytics.trend_analyzer import TrendAnalyzer
from lahore.src.analytics.emergency_router import EmergencyRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_simulated_traffic_data(graph, hours=24) -> pd.DataFrame:
    """Generate 24 hours of simulated traffic data for all edges."""
    logger.info(f"Generating simulated traffic data for {hours} hours...")
    
    edges = list(graph.edges(keys=True, data=True))
    # Select a subset if graph is too large, but for demo we'll take enough
    sample_size = min(len(edges), 5000)
    sampled_edges = random.sample(edges, sample_size)
    
    data_list = []
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    for h in range(hours):
        current_time = base_time + timedelta(hours=h)
        # Hour of day influence
        hour_factor = 1.0 + 0.5 * np.sin((h - 8) * np.pi / 12) # Peak around 8 AM / 8 PM
        
        for u, v, k, data in sampled_edges:
            length = data.get('length', 1.0)
            base_speed = 50.0  # km/h
            
            # Simple simulation of speed/volume
            vol = int(np.random.poisson(100 * hour_factor))
            speed = max(5, base_speed * (1.1 - 0.5 * (vol / 200))) 
            congestion = 1.0 + (vol / 150)
            
            data_list.append({
                'timestamp': current_time,
                'u': u, 'v': v, 'key': k,
                'speed': speed,
                'volume': vol,
                'congestion_level': congestion
            })
            
    return pd.DataFrame(data_list)

def run_verification():
    logger.info("="*60)
    logger.info("🚀 Starting Advanced Analytics Verification")
    logger.info("="*60)
    
    # Initialize components
    router = EmergencyRouter() # Loads graph
    graph = router.graph
    
    # 1. Trend Analysis
    logger.info("\n📊 Step 1: Trend Analysis")
    traffic_data = generate_simulated_traffic_data(graph)
    analyzer = TrendAnalyzer(traffic_data)
    peaks = analyzer.identify_peak_hours()
    logger.info(f"✅ Identified Peaks: {peaks}")
    analyzer.plot_patterns()
    
    # 2. Anomaly Detection
    logger.info("\n🚨 Step 2: Anomaly Detection")
    # Inject an anomaly: Edge 100 has a massive speed drop suddenly
    target_idx = 50 
    u_anom = traffic_data.iloc[target_idx]['u']
    v_anom = traffic_data.iloc[target_idx]['v']
    
    # Modify some rows for this edge
    traffic_data.loc[(traffic_data['u'] == u_anom) & (traffic_data['v'] == v_anom), 'speed'] = 2.0
    traffic_data.loc[(traffic_data['u'] == u_anom) & (traffic_data['v'] == v_anom), 'volume'] = 500
    
    detector = AnomalyDetector(contamination=0.01)
    results = detector.analyze_incidents(traffic_data)
    
    detected_anoms = results[results['is_anomaly']]
    logger.info(f"✅ Detected {len(detected_anoms)} total anomalies.")
    
    # Check if our injected one was caught
    found_injected = any((detected_anoms['u'] == u_anom) & (detected_anoms['v'] == v_anom))
    logger.info(f"🔎 Injected anomaly at ({u_anom}, {v_anom}) caught: {found_injected}")
    
    # 3. Bottleneck Identification
    logger.info("\n🚧 Step 3: Bottleneck Identification")
    identifier = BottleneckIdentifier(graph)
    hotspots = identifier.get_top_critical_points(traffic_data, top_n=5)
    logger.info("✅ Top 5 Bottlenecks Identified:")
    for i, row in hotspots.iterrows():
        logger.info(f"  - {row['road_name']}: Mean Congestion {row['mean']:.2f}")

    # 4. Emergency Routing
    logger.info("\n🚑 Step 4: Emergency Routing")
    # Choose two nodes that might have a choice between narrow streets and main road
    source, target = router.get_random_node_pair()
    comparison = router.compare_with_standard(source, target)
    
    logger.info(f"✅ Emergency Route comparison complete.")
    logger.info(f"  Standard Dist: {comparison['standard']['distance']:.2f}m")
    logger.info(f"  Emergency Dist: {comparison['emergency']['distance']:.2f}m")
    logger.info(f"  Paths Different: {comparison['paths_different']}")

    logger.info("\n" + "="*60)
    logger.info("✅ Advanced Analytics Verification Complete!")
    logger.info("="*60)

if __name__ == "__main__":
    run_verification()
