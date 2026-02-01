import redis
import json
import logging
from typing import Optional, Any
from shared.config.database.config import REDIS_HOST, REDIS_PORT

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages Redis caching for API performance optimization.
    """
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT):
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.client = None

    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set(self, key: str, value: Any, expire: int = 300):
        if not self.client:
            return
        self.client.set(key, json.dumps(value), ex=expire)

    def get_latest_speed(self, edge_id: str) -> Optional[float]:
        """Get the latest predicted speed for an edge."""
        return self.get(f"speed:{edge_id}")

    def cache_route(self, start: int, end: int, aware: bool, route: dict):
        """Cache optimized routes for 5 minutes."""
        key = f"route:{start}:{end}:{aware}"
        self.set(key, route, expire=300)

cache_manager = CacheManager()
