"""
Online Predictor for Real-Time Traffic Speed Prediction.
Loads trained models and performs live inference on streaming features.
"""
import torch
import numpy as np
import logging
import json
import time
from typing import Dict, List, Any, Optional
from kafka import KafkaConsumer, KafkaProducer
from pathlib import Path
from lahore.src.ml_models.drift_detector import DriftDetector
from lahore.src.ml_models.ab_tester import ABTestingFramework

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlinePredictor:
    """
    Performs real-time traffic prediction using trained models.
    Consumes from feature topic and publishes to prediction topic.
    """
    
    def __init__(
        self,
        model_path: str = "lahore/models/trained/transformer.pth",
        input_topic: str = "lahore_traffic_features",
        output_topic: str = "lahore_traffic_predictions",
        bootstrap_servers: str = "localhost:9092"
    ):
        self.model_path = Path(model_path)
        self.shadow_model_path = Path("lahore/models/trained/transformer_shadow.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.model = self._load_model(self.model_path, "Champion")
        self.shadow_model = self._load_model(self.shadow_model_path, "Challenger")
        
        # ML Pipeline Components
        self.drift_detector = DriftDetector(reference_data=list(np.random.normal(30, 5, 500)))
        self.ab_tester = ABTestingFramework(metrics_window=100)
        
        # Kafka setup
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='lahore_prediction_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.output_topic = output_topic
        logger.info(f"✅ OnlinePredictor initialized on {self.device}")
    
    def _load_model(self, path: Path, name: str) -> Optional[torch.nn.Module]:
        """Load a trained Transformer model."""
        try:
            if path.exists():
                model = torch.load(path, map_location=self.device)
                model.eval()
                logger.info(f"✅ Loaded {name} model from {path}")
                return model
            else:
                logger.warning(f"⚠️ {name} model not found at {path}.")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load {name} model: {e}")
            return None
    
    def _prepare_features(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to model input tensor.
        """
        # Extract relevant features for prediction
        features = [
            feature_dict.get('avg_speed', 30.0),
            feature_dict.get('std_speed', 5.0),
            feature_dict.get('min_speed', 10.0),
            feature_dict.get('max_speed', 50.0),
            feature_dict.get('total_volume', 100),
            feature_dict.get('congestion_index', 0.5),
        ]
        return np.array(features, dtype=np.float32)
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate prediction for the next time window.
        """
        start_time = time.time()
        
        # 1. Check for data drift
        current_speed = features.get('avg_speed', 30.0)
        drift_status = self.drift_detector.check_data_drift([current_speed])
        
        # 2. Champion Prediction
        predicted_speed = self._run_inference(self.model, features)
        
        # 3. Challenger Prediction (if exists)
        shadow_pred = None
        if self.shadow_model:
            shadow_pred = self._run_inference(self.shadow_model, features)
            # In real system, we'd record actual later. Here we mock comparison data.
            # In a true verifier, we'd wait for the 'ground truth' update to match.
            self.ab_tester.record_prediction('champion', predicted_speed, current_speed)
            self.ab_tester.record_prediction('challenger', shadow_pred, current_speed)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'u': features.get('u'),
            'v': features.get('v'),
            'key': features.get('key'),
            'current_speed': current_speed,
            'predicted_speed_15min': round(predicted_speed, 2),
            'shadow_pred': round(shadow_pred, 2) if shadow_pred else None,
            'drift_detected': drift_status['drift_detected'],
            'prediction_timestamp': int(time.time()),
            'latency_ms': round(latency_ms, 2),
            'model_type': 'transformer' if self.model else 'mock'
        }

    def _run_inference(self, model: Optional[torch.nn.Module], features: Dict[str, Any]) -> float:
        """Helper to run model or mock inference."""
        if model is not None:
            try:
                input_tensor = torch.tensor(
                    self._prepare_features(features)
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    return float(model(input_tensor).cpu().numpy()[0])
            except Exception:
                return features.get('avg_speed', 30.0) * 0.95
        return features.get('avg_speed', 30.0) * (1 - features.get('congestion_index', 0.5) * 0.1)
    
    def run(self, max_predictions: Optional[int] = None):
        """
        Main loop: consume features, predict, and publish.
        """
        logger.info(f"🚀 Starting prediction loop...")
        pred_count = 0
        total_latency = 0.0
        
        try:
            for message in self.consumer:
                features = message.value
                prediction = self.predict(features)
                
                # Publish prediction
                self.producer.send(self.output_topic, value=prediction)
                
                pred_count += 1
                total_latency += prediction['latency_ms']
                
                if pred_count % 50 == 0:
                    avg_latency = total_latency / pred_count
                    logger.info(f"📊 Predictions: {pred_count}, Avg Latency: {avg_latency:.2f}ms")
                    
                    if self.shadow_model:
                        logger.info(self.ab_tester.get_summary())
                
                if max_predictions and pred_count >= max_predictions:
                    break
        
        except KeyboardInterrupt:
            logger.info("Stopping predictor...")
        finally:
            self.producer.flush()
            self.producer.close()
            self.consumer.close()
            
            if pred_count > 0:
                avg_latency = total_latency / pred_count
                logger.info(f"✅ Final Stats: {pred_count} predictions, Avg Latency: {avg_latency:.2f}ms")

if __name__ == "__main__":
    predictor = OnlinePredictor()
    predictor.run(max_predictions=100)
