import sys
import logging
from sqlalchemy import create_engine, text
import redis
from kafka import KafkaProducer
from shared.config.database.config import DATABASE_URL, REDIS_HOST, REDIS_PORT, KAFKA_BOOTSTRAP_SERVERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_postgres():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ PostgreSQL connection successful!")
        return True
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        return False

def verify_redis():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        logger.info("✅ Redis connection successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

def verify_kafka():
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            request_timeout_ms=5000,
            retry_backoff_ms=500
        )
        if producer.bootstrap_connected():
            logger.info("✅ Kafka connection successful!")
            return True
        else:
            logger.error("❌ Kafka bootstrap not connected")
            return False
    except Exception as e:
        logger.error(f"❌ Kafka connection failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting infrastructure verification...")
    results = [
        verify_postgres(),
        verify_redis(),
        verify_kafka()
    ]
    
    if all(results):
        logger.info("🚀 All infrastructure components are online and reachable!")
        sys.exit(0)
    else:
        logger.error("⚠️ Some infrastructure components failed verification.")
        sys.exit(1)
