"""
Traffic Trend Analysis for Lahore.
Analyzes long-term patterns, diurnal cycles, and seasonal variations.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    def __init__(self, traffic_df: pd.DataFrame):
        """
        Args:
            traffic_df: DataFrame with columns ['timestamp', 'speed', 'volume', 'congestion_level']
        """
        self.df = traffic_df.copy()
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            self.df['is_weekend'] = self.df['day_of_week'] >= 5

    def get_diurnal_patterns(self) -> pd.DataFrame:
        """Calculate average traffic metrics by hour of day."""
        logger.info("Calculating diurnal traffic patterns...")
        return self.df.groupby('hour')[['speed', 'volume', 'congestion_level']].mean()

    def identify_peak_hours(self) -> Dict[str, List[int]]:
        """Identify morning and evening peak hours."""
        hourly_stats = self.get_diurnal_patterns()
        
        # Simple logic: hours with highest average volume/congestion
        sorted_hours = hourly_stats.sort_values(by='congestion_level', ascending=False).index.tolist()
        
        morning_peak = [h for h in sorted_hours if 7 <= h <= 10][:2]
        evening_peak = [h for h in sorted_hours if 16 <= h <= 20][:2]
        
        return {
            "morning_peak": sorted(morning_peak),
            "evening_peak": sorted(evening_peak)
        }

    def compare_weekend_vs_weekday(self) -> pd.DataFrame:
        """Compare traffic behavior between weekdays and weekends."""
        logger.info("Comparing weekend vs weekday patterns...")
        return self.df.groupby(['is_weekend', 'hour'])[['speed', 'volume']].mean().reset_index()

    def get_growth_trends(self, window: str = '1D') -> pd.Series:
        """Calculate rolling average to see long-term growth/decline."""
        if 'timestamp' not in self.df.columns:
            return pd.Series()
            
        indexed_df = self.df.set_index('timestamp').sort_index()
        return indexed_df['volume'].resample(window).mean()

    def plot_patterns(self, output_dir: str = 'lahore/data/plots'):
        """Generate trend plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Diurnal Cycle
        plt.figure(figsize=(10, 6))
        hourly = self.get_diurnal_patterns()
        sns.lineplot(data=hourly, x=hourly.index, y='congestion_level', marker='o')
        plt.title('Average Hourly Congestion (Diurnal Cycle)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Congestion Level')
        plt.savefig(f"{output_dir}/diurnal_pattern.png")
        plt.close()
        
        # 2. Weekday vs Weekend
        plt.figure(figsize=(10, 6))
        comp = self.compare_weekend_vs_weekday()
        comp['Day Type'] = comp['is_weekend'].map({True: 'Weekend', False: 'Weekday'})
        sns.lineplot(data=comp, x='hour', y='speed', hue='Day Type')
        plt.title('Speed Trends: Weekday vs Weekend')
        plt.savefig(f"{output_dir}/weekend_comparison.png")
        plt.close()
        
        logger.info(f"Trend plots saved to {output_dir}")
