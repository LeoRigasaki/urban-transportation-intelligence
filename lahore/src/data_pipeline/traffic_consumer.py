import json
import logging
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text
import pandas as pd
from shared.config.database.config import KAFKA_BOOTSTRAP_SERVERS, DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficConsumer:
    def __init__(self, topic='lahore_traffic_updates'):
        self.topic = topic
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='lahore_traffic_consumer_group',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info(f"✅ Connected to Kafka for consuming topic '{self.topic}'")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka Consumer: {e}")
            raise

        self.engine = create_engine(DATABASE_URL)
        self._init_db()

    def _init_db(self):
        """Creates the historical traffic table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS lahore_traffic_history (
            id SERIAL PRIMARY KEY,
            u BIGINT,
            v BIGINT,
            key INT,
            timestamp BIGINT,
            speed FLOAT,
            volume INT
        );
        CREATE INDEX IF NOT EXISTS idx_traffic_uv ON lahore_traffic_history (u, v);
        CREATE INDEX IF NOT EXISTS idx_traffic_ts ON lahore_traffic_history (timestamp);
        """
        with self.engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()
        logger.info("Initialized historical traffic table.")

    def run(self, max_messages=None):
        """
        Consumes messages and persists them.
        :param max_messages: Stop after this many messages (None for infinity)
        """
        logger.info(f"👂 Listening for traffic updates on '{self.topic}'...")
        msg_count = 0
        batch = []
        batch_size = 100
        
        try:
            for message in self.consumer:
                data = message.value
                batch.append(data)
                msg_count += 1
                
                if len(batch) >= batch_size:
                    self._persist_batch(batch)
                    batch = []
                    logger.info(f"Persisted {msg_count} messages so far...")
                
                if max_messages and msg_count >= max_messages:
                    if batch:
                        self._persist_batch(batch)
                    logger.info(f"Reached max_messages: {max_messages}")
                    break
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.consumer.close()

    def _persist_batch(self, batch):
        df = pd.DataFrame(batch)
        df.to_sql("lahore_traffic_history", self.engine, if_exists="append", index=False)

if __name__ == "__main__":
    consumer = TrafficConsumer()
    # Run until manual stop or for a few batches in this context
    consumer.run(max_messages=1000)
