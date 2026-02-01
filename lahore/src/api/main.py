import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from lahore.src.api.schemas import (
    PredictionRequest, PredictionResponse, 
    RoutingRequest, RoutingResponse, RouteStep, SystemStats
)
from lahore.src.api.websocket_manager import manager, kafka_to_websocket
from lahore.src.api.cache_manager import cache_manager
from lahore.src.optimization.congestion_router import CongestionRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Singletons
router_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Heavy Graph and Start Kafka Bridge
    global router_engine
    logger.info("📡 Starting API and background tasks...")
    
    try:
        router_engine = CongestionRouter()
        # Initial load from Redis if available
        router_engine.load_congestion_from_redis()
    except Exception as e:
        logger.error(f"Failed to initialize router engine: {e}")

    # Start Kafka-to-WebSocket bridge in background
    kafka_task = asyncio.create_task(kafka_to_websocket())
    
    yield
    
    # Shutdown: Clean up
    logger.info("🛑 Shutting down API...")
    kafka_task.cancel()

app = FastAPI(
    title="Lahore Smart Traffic AI",
    description="Real-time traffic prediction and congestion-aware routing API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "healthy", "components": {"redis": cache_manager.client is not None, "graph": router_engine is not None}}

@app.post("/route", response_model=RoutingResponse)
async def get_route(request: RoutingRequest):
    if not router_engine:
        raise HTTPException(status_code=503, detail="Routing engine not initialized")
    
    # Check cache first
    cached = cache_manager.get(f"route:{request.start_node}:{request.end_node}:{request.congestion_aware}")
    if cached:
        logger.info("🚀 Returning cached route")
        return cached

    # Update congestion from Redis before routing
    router_engine.load_congestion_from_redis()
    
    path, cost, meta = router_engine.get_optimal_route(
        request.start_node, request.end_node, use_congestion=request.congestion_aware
    )
    
    if not path:
        raise HTTPException(status_code=404, detail="No route found between nodes")

    # Construct steps (basic for now)
    steps = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        data = router_engine.optimizer.graph[u][v][0]
        steps.append(RouteStep(
            u=u, v=v, 
            name=data.get('name', 'Unnamed Road'),
            length=data.get('length', 0),
            speed=data.get('travel_time', data.get('length', 0)) # Placeholder speed
        ))

    response = RoutingResponse(
        total_length=sum(s.length for s in steps),
        estimated_time=cost,
        path=path,
        steps=steps
    )
    
    # Cache result
    cache_manager.set(f"route:{request.start_node}:{request.end_node}:{request.congestion_aware}", response.dict())
    return response

@app.websocket("/ws/traffic")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We just need to keep the connection open, 
            # the background task handles the broadcasting
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/analytics/stats", response_model=SystemStats)
async def get_stats():
    # In real world, we'd query Prometheus or DB. Mocking for now.
    return SystemStats(
        predictions_count=12540,
        avg_latency_ms=0.08,
        drift_events_detected=2,
        active_challenger=True,
        performance_change_pct=65.2
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
