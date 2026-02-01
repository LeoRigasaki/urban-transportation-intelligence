from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PredictionRequest(BaseModel):
    u: int
    v: int
    key: int

class PredictionResponse(BaseModel):
    u: int
    v: int
    key: int
    current_speed: float
    predicted_speed_15min: float
    shadow_pred: Optional[float] = None
    drift_detected: bool
    prediction_timestamp: int
    latency_ms: float
    model_type: str

class RoutingRequest(BaseModel):
    start_node: int
    end_node: int
    congestion_aware: bool = True

class RouteStep(BaseModel):
    u: int
    v: int
    name: Optional[str] = None
    length: float
    speed: float

class RoutingResponse(BaseModel):
    total_length: float
    estimated_time: float
    path: List[int]
    steps: List[RouteStep]

class SystemStats(BaseModel):
    predictions_count: int
    avg_latency_ms: float
    drift_events_detected: int
    active_challenger: bool
    performance_change_pct: Optional[float] = None
