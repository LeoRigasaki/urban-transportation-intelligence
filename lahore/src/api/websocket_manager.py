from fastapi import WebSocket
import asyncio
import json
import logging
from typing import List
from kafka import KafkaConsumer
from shared.config.database.config import KAFKA_BOOTSTRAP_SERVERS

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket client connections.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                # We'll handle stale connections during cleanup or next broadcast

manager = ConnectionManager()

async def kafka_to_websocket(topic: str = "lahore_traffic_predictions"):
    """
    Continuously consumes from Kafka and broadcasts to all WebSocket clients.
    """
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
        logger.info(f"🚀 WebSocket Kafka Consumer started on topic {topic}")
        
        # This needs to be run as a background task
        while True:
            # We use non-blocking approach for FastAPI loop
            msg_pack = consumer.poll(timeout_ms=100)
            for tp, messages in msg_pack.items():
                for message in messages:
                    await manager.broadcast(json.dumps(message.value))
            
            await asyncio.sleep(0.01) # Yield control
            
    except Exception as e:
        logger.error(f"Kafka to WebSocket bridge failed: {e}")
