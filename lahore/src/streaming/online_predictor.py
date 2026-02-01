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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        
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
    
    def _load_model(self) -> Optional[torch.nn.Module]:
        """Load the trained Transformer model."""
        try:
            if self.model_path.exists():
                model = torch.load(self.model_path, map_location=self.device)
                model.eval()
                logger.info(f"✅ Loaded model from {self.model_path}")
                return model
            else:
                logger.warning(f"⚠️ Model not found at {self.model_path}. Using mock predictor.")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
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
        
        if self.model is not None:
            # Real model inference
            try:
                input_tensor = torch.tensor(
                    self._prepare_features(features)
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_tensor)
                    predicted_speed = float(output.cpu().numpy()[0])
            except Exception as e:
                logger.warning(f"Model inference failed: {e}. Using fallback.")
                predicted_speed = features.get('avg_speed', 30.0) * 0.95
        else:
            # Mock prediction: slight decline from current speed
            current_speed = features.get('avg_speed', 30.0)
            congestion = features.get('congestion_index', 0.5)
            predicted_speed = current_speed * (1 - congestion * 0.1)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'u': features.get('u'),
            'v': features.get('v'),
            'key': features.get('key'),
            'current_speed': features.get('avg_speed'),
            'predicted_speed_15min': round(predicted_speed, 2),
            'congestion_index': features.get('congestion_index'),
            'prediction_timestamp': int(time.time()),
            'latency_ms': round(latency_ms, 2),
            'model_type': 'transformer' if self.model else 'mock'
        }
    
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
