import time
import json
import random
import logging
import numpy as np
from kafka import KafkaProducer
from sqlalchemy import create_engine
import pandas as pd
from shared.config.database.config import KAFKA_BOOTSTRAP_SERVERS, DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficSimulator:
    def __init__(self, topic='lahore_traffic_updates'):
        self.topic = topic
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all'
            )
            logger.info(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            raise

        # Load edges to simulate traffic on
        self.engine = create_engine(DATABASE_URL)
        self.edges_df = pd.read_sql("SELECT u, v, key, length, maxspeed FROM lahore_edges", self.engine)
        # Convert maxspeed to numeric, default to 40 if missing/invalid
        self.edges_df['maxspeed'] = pd.to_numeric(self.edges_df['maxspeed'], errors='coerce').fillna(40)
        logger.info(f"Loaded {len(self.edges_df)} road segments for simulation.")

    def get_traffic_factor(self):
        """
        Simulates time-of-day traffic variation using a sine wave.
        Returns a factor between 0.2 (congested) and 1.0 (free flow).
        """
        current_hour = time.localtime().tm_hour
        # Peak hours: 8-9am and 5-7pm
        # Basic sine wave centered around 12pm
        factor = 0.6 + 0.3 * np.sin((current_hour - 12) * np.pi / 12)
        # Add rush hour dips
        if 8 <= current_hour <= 9 or 17 <= current_hour <= 19:
            factor -= random.uniform(0.2, 0.4)
        return max(0.1, min(1.0, factor))

    def run(self, interval=5, duration=None):
        """
        Runs the simulation.
        :param interval: Seconds between updates per batch
        :param duration: Total runtime in seconds (None for infinity)
        """
        logger.info(f"🚀 Starting traffic simulation on topic '{self.topic}'...")
        start_time = time.time()
        
        try:
            while True:
                if duration and (time.time() - start_time) > duration:
                    break
                
                batch_time = time.time()
                traffic_factor = self.get_traffic_factor()
                
                # Sample 1000 edges per update to avoid overwhelming if testing
                # In full system, we might update all or use partitions
                sample_edges = self.edges_df.sample(min(1000, len(self.edges_df)))
                
                updates = []
                for _, edge in sample_edges.iterrows():
                    # Simulate speed: base speed * factor * noise
                    base_speed = edge['maxspeed']
                    current_speed = base_speed * traffic_factor * random.uniform(0.8, 1.2)
                    current_speed = min(current_speed, base_speed * 1.1) # Max 10% overspeed
                    
                    # Simulate volume
                    volume = int(random.uniform(10, 100) * (1 - traffic_factor + 0.5))
                    
                    update = {
                        'u': int(edge['u']),
                        'v': int(edge['v']),
                        'key': int(edge['key']),
                        'timestamp': int(batch_time),
                        'speed': round(float(current_speed), 2),
                        'volume': volume
                    }
                    self.producer.send(self.topic, value=update)
                    updates.append(update)
                
                self.producer.flush()
                logger.info(f"Sent batch of {len(updates)} updates. Traffic Factor: {traffic_factor:.2f}")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping simulator...")
        finally:
            self.producer.close()

if __name__ == "__main__":
    simulator = TrafficSimulator()
    # Run for 60 seconds as a test pulse
    simulator.run(interval=2, duration=60)
