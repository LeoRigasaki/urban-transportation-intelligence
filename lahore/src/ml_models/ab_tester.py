"""
A/B Testing Framework for Model Evaluation.
Manages Champion (Production) and Challenger (Shadow) model evaluation.
"""
import logging
import time
import json
from typing import Dict, Any, List, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ABTestingFramework:
    """
    Evaluates multiple models simultaneously on real-time data streams.
    """
    def __init__(self, metrics_window: int = 100):
        self.champion_errors = []
        self.challenger_errors = []
        self.window = metrics_window
        self.metrics = {
            'champion': {'mae': 0.0, 'rmse': 0.0, 'count': 0},
            'challenger': {'mae': 0.0, 'rmse': 0.0, 'count': 0}
        }

    def record_prediction(self, model_type: str, prediction: float, actual: float):
        """Record a single prediction result."""
        error = abs(prediction - actual)
        sq_error = (prediction - actual)**2
        
        if model_type == 'champion':
            self.champion_errors.append((error, sq_error))
            if len(self.champion_errors) > self.window:
                self.champion_errors.pop(0)
        else:
            self.challenger_errors.append((error, sq_error))
            if len(self.challenger_errors) > self.window:
                self.challenger_errors.pop(0)
                
        self._update_metrics()

    def _update_metrics(self):
        for m_type, errs in [('champion', self.champion_errors), ('challenger', self.challenger_errors)]:
            if not errs:
                continue
            
            maes = [e[0] for e in errs]
            rmses = [e[1] for e in errs]
            
            self.metrics[m_type] = {
                'mae': float(np.mean(maes)),
                'rmse': float(np.sqrt(np.mean(rmses))),
                'count': len(errs)
            }

    def get_summary(self) -> str:
        """Returns a human-readable comparison summary."""
        champ = self.metrics['champion']
        chall = self.metrics['challenger']
        
        summary = (
            f"\n⚖️ A/B TEST SUMMARY ({champ['count']} samples)\n"
            f"----------------------------------------\n"
            f"🏆 CHAMPION: MAE={champ['mae']:.4f}, RMSE={champ['rmse']:.4f}\n"
            f"🚀 CHALLENGER: MAE={chall['mae']:.4f}, RMSE={chall['rmse']:.4f}\n"
        )
        
        if chall['count'] > 0 and champ['count'] > 0:
            improvement = (champ['mae'] - chall['mae']) / (champ['mae'] + 1e-6) * 100
            summary += f"📈 Performance Change: {improvement:+.2f}%\n"
            
            if improvement > 5.0 and chall['count'] >= self.window:
                summary += "⚠️ ACTION: Challenger shows significant improvement. Consider Promotion!\n"
                
        return summary

if __name__ == "__main__":
    # Test A/B Framework
    tester = ABTestingFramework(metrics_window=50)
    
    # Simulate 100 predictions
    for i in range(100):
        actual = 30.0 + i*0.1
        tester.record_prediction('champion', actual + np.random.normal(0, 5), actual)
        tester.record_prediction('challenger', actual + np.random.normal(0, 3), actual)
        
    print(tester.get_summary())
