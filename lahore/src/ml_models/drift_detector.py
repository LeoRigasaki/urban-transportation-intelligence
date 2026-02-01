"""
Drift Detector for Monitoring Traffic Pattern Changes.
Uses statistical tests to detect distribution shifts in real-time data.
"""
import numpy as np
from scipy import stats
import logging
import time
from typing import List, Dict, Any, Optional
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects distribution shifts (Data Drift) or prediction performance drops (Concept Drift).
    """
    def __init__(
        self,
        reference_data: List[float] = None,
        window_size: int = 500,
        significance_level: float = 0.05,
        error_threshold: float = 5.0 # MAE threshold in km/h
    ):
        self.reference_data = reference_data if reference_data else []
        self.current_window = deque(maxlen=window_size)
        self.error_window = deque(maxlen=window_size)
        self.p_value_threshold = significance_level
        self.error_threshold = error_threshold
        
    def add_reference_data(self, data: List[float]):
        self.reference_data.extend(data)
        logger.info(f"Loaded {len(self.reference_data)} samples into reference distribution.")

    def check_data_drift(self, new_samples: List[float]) -> Dict[str, Any]:
        """
        Performs Kolmogorov-Smirnov test to check if new samples come from the same distribution.
        """
        self.current_window.extend(new_samples)
        
        if len(self.reference_data) < 100 or len(self.current_window) < 100:
            return {'drift_detected': False, 'p_value': 1.0, 'reason': 'insufficient_data'}
        
        # We compare a recent slice of the window to our reference
        recent_data = list(self.current_window)
        ks_stat, p_value = stats.ks_2samp(self.reference_data, recent_data)
        
        drift = p_value < self.p_value_threshold
        
        if drift:
            logger.warning(f"🚨 DATA DRIFT DETECTED! (KS-stat: {ks_stat:.4f}, p-value: {p_value:.6f})")
            
        return {
            'drift_detected': drift,
            'stat': float(ks_stat),
            'p_value': float(p_value),
            'timestamp': time.time()
        }

    def check_concept_drift(self, errors: List[float]) -> Dict[str, Any]:
        """
        Checks if the model prediction error is trending upwards.
        """
        self.error_window.extend(errors)
        
        if len(self.error_window) < 50:
            return {'drift_detected': False, 'mean_error': 0.0}
            
        current_mae = np.mean(self.error_window)
        drift = current_mae > self.error_threshold
        
        if drift:
            logger.warning(f"🚨 CONCEPT DRIFT DETECTED! (MAE: {current_mae:.2f} > {self.error_threshold})")
            
        return {
            'drift_detected': drift,
            'mean_error': float(current_mae),
            'threshold': self.error_threshold,
            'timestamp': time.time()
        }

if __name__ == "__main__":
    # Test Drift Detector
    detector = DriftDetector(reference_data=list(np.random.normal(30, 5, 500)))
    
    # 1. Normal data (no drift)
    res_normal = detector.check_data_drift(list(np.random.normal(30, 5, 200)))
    print(f"Normal Data Drift: {res_normal['drift_detected']}")
    
    # 2. Shifted data (drift)
    res_drift = detector.check_data_drift(list(np.random.normal(15, 3, 200)))
    print(f"Shifted Data Drift: {res_drift['drift_detected']}")
