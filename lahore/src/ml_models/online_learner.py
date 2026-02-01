"""
Online Learner for Incremental Model Updates.
Consumes ground truth data from Kafka and fine-tunes the Traffic Transformer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import json
import time
import os
from typing import Dict, List, Any, Optional
from kafka import KafkaConsumer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OnlineLearner:
    """
    Handles incremental updates to the Traffic Transformer model.
    """
    def __init__(
        self,
        base_model_path: str = "lahore/models/trained/transformer.pth",
        shadow_model_dir: str = "lahore/models/checkpoints/shadow/",
        bootstrap_servers: str = "localhost:9092",
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_model_path = Path(base_model_path)
        self.shadow_model_dir = Path(shadow_model_dir)
        self.shadow_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and optimizer
        self.model = self._load_model()
        if self.model:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()
        
        self.batch_size = batch_size
        self.buffer = []
        
        # Kafka setup for ground truth
        self.consumer = KafkaConsumer(
            'lahore_traffic_updates',
            bootstrap_servers=bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='lahore_online_learner_group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        logger.info(f"✅ OnlineLearner initialized on {self.device}")

    def _load_model(self) -> Optional[nn.Module]:
        try:
            if self.base_model_path.exists():
                # Note: In real scenarios, we'd need the class definition or a state_dict
                # For this implementation, we assume we can load the full model object
                model = torch.load(self.base_model_path, map_location=self.device)
                model.train()
                return model
            return None
        except Exception as e:
            logger.error(f"Failed to load model for online learning: {e}")
            return None

    def _prepare_data(self, samples: List[Dict[str, Any]]):
        """Convert list of dicts to training tensors."""
        # This mapping must match the transformer's expected input
        # Standard features: [avg_speed, std_speed, min_speed, max_speed, volume, congestion]
        features = []
        targets = []
        
        for s in samples:
            feat = [
                s.get('speed', 30.0),
                0.0, # std missing in raw update
                s.get('speed', 30.0), # min
                s.get('speed', 30.0), # max
                s.get('volume', 50),
                max(0.0, min(1.0, 1.0 - (s.get('speed', 30.0) / 60.0))) # congestion
            ]
            # Mock target: next speed (for verification we use slightly different value)
            target = [s.get('speed', 30.0) * 0.98] 
            
            features.append(feat)
            targets.append(target)
            
        return torch.tensor(features, dtype=torch.float32).to(self.device), \
               torch.tensor(targets, dtype=torch.float32).to(self.device)

    def fine_tune_step(self, samples: List[Dict[str, Any]]):
        """Perform a single weight update step."""
        if not self.model or not samples:
            return
        
        X, y = self._prepare_data(samples)
        
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        
        logger.info(f"🌱 Incremental Update Loss: {loss.item():.6f}")
        return loss.item()

    def run(self, update_threshold: int = 100):
        """Listen for data and update model when threshold is reached."""
        logger.info(f"🚀 Waiting for traffic data (Update every {update_threshold} samples)...")
        
        try:
            for message in self.consumer:
                self.buffer.append(message.value)
                
                if len(self.buffer) >= update_threshold:
                    loss = self.fine_tune_step(self.buffer)
                    self._save_shadow_model()
                    self.buffer = []
        except KeyboardInterrupt:
            logger.info("Stopping OnlineLearner...")
        finally:
            self.consumer.close()

    def _save_shadow_model(self):
        """Save a shadow model for A/B testing."""
        ts = int(time.time())
        path = self.shadow_model_dir / f"transformer_shadow_{ts}.pth"
        torch.save(self.model, path)
        logger.info(f"💾 Saved shadow model to {path}")

if __name__ == "__main__":
    learner = OnlineLearner()
    learner.run()
