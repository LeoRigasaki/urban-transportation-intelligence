"""
Stream Processor for Real-Time Traffic Analytics.
Uses Faust for Kafka stream processing with windowed aggregations.
"""
import faust
import logging
from datetime import timedelta
from typing import Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Faust App Configuration
app = faust.App(
    'lahore-traffic-stream',
    broker='kafka://localhost:9092',
    value_serializer='json',
)

# Input topic: raw traffic updates
class TrafficUpdate(faust.Record):
    u: int
    v: int
    key: int
    timestamp: int
    speed: float
    volume: int

# Output topic: aggregated features
class TrafficFeatures(faust.Record):
    u: int
    v: int
    key: int
    window_start: int
    window_end: int
    avg_speed: float
    min_speed: float
    max_speed: float
    std_speed: float
    total_volume: int
    update_count: int
    congestion_index: float

# Topics
traffic_updates_topic = app.topic('lahore_traffic_updates', value_type=TrafficUpdate)
traffic_features_topic = app.topic('lahore_traffic_features', value_type=TrafficFeatures)

# Table for windowed aggregation (5-minute tumbling window)
traffic_table = app.Table(
    'traffic_aggregates',
    default=lambda: {
        'speeds': [],
        'volumes': [],
        'count': 0,
        'window_start': 0
    },
    partitions=8,
).tumbling(timedelta(minutes=5), expires=timedelta(hours=1))

@app.agent(traffic_updates_topic)
async def process_traffic_updates(updates):
    """
    Process incoming traffic updates and aggregate into 5-minute windows.
    """
    async for update in updates:
        edge_key = f"{update.u}_{update.v}_{update.key}"
        
        # Get current window state
        current = traffic_table[edge_key].value()
        
        # Initialize window start if first update
        if current['count'] == 0:
            current['window_start'] = update.timestamp
        
        # Accumulate values
        current['speeds'].append(update.speed)
        current['volumes'].append(update.volume)
        current['count'] += 1
        
        # Update table
        traffic_table[edge_key] = current
        
        # Log progress periodically
        if current['count'] % 100 == 0:
            logger.info(f"Processed {current['count']} updates for edge {edge_key}")

@app.timer(interval=60.0)  # Emit features every 60 seconds
async def emit_features():
    """
    Periodically emit aggregated features to the output topic.
    """
    import numpy as np
    
    current_time = int(time.time())
    emitted_count = 0
    
    for key, window in traffic_table.items():
        data = window.value()
        if data['count'] == 0:
            continue
        
        speeds = data['speeds']
        volumes = data['volumes']
        
        # Calculate statistics
        avg_speed = np.mean(speeds)
        std_speed = np.std(speeds) if len(speeds) > 1 else 0.0
        min_speed = np.min(speeds)
        max_speed = np.max(speeds)
        total_volume = sum(volumes)
        
        # Congestion index: lower speed = higher congestion
        # Normalized between 0 (free flow) and 1 (gridlock)
        # Assuming max free-flow speed of 60 km/h
        congestion_index = max(0.0, min(1.0, 1.0 - (avg_speed / 60.0)))
        
        # Parse edge key
        parts = key.split('_')
        u, v, k = int(parts[0]), int(parts[1]), int(parts[2])
        
        features = TrafficFeatures(
            u=u,
            v=v,
            key=k,
            window_start=data['window_start'],
            window_end=current_time,
            avg_speed=round(avg_speed, 2),
            min_speed=round(min_speed, 2),
            max_speed=round(max_speed, 2),
            std_speed=round(std_speed, 2),
            total_volume=total_volume,
            update_count=data['count'],
            congestion_index=round(congestion_index, 3)
        )
        
        await traffic_features_topic.send(value=features)
        emitted_count += 1
    
    if emitted_count > 0:
        logger.info(f"📤 Emitted {emitted_count} feature records to '{traffic_features_topic.get_topic_name()}'")

if __name__ == '__main__':
    app.main()
