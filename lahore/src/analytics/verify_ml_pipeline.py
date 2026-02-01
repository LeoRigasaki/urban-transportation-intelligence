"""
Verification Script for Day 8: Advanced ML Pipeline.
Simulates data drift and verifies detection and A/B testing logic.
"""
import numpy as np
import logging
import time
import json
from pathlib import Path
from lahore.src.ml_models.drift_detector import DriftDetector
from lahore.src.ml_models.ab_tester import ABTestingFramework

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_verification():
    logger.info("=" * 60)
    logger.info("🚀 Day 8: Advanced ML Pipeline Verification")
    logger.info("=" * 60)

    # 1. Test Drift Detection
    logger.info("\n📋 Step 1: Testing Data Drift Detection (KS-Test)")
    
    # Reference data: Normal city speeds (30 km/h avg)
    ref_speeds = list(np.random.normal(30, 5, 500))
    detector = DriftDetector(reference_data=ref_speeds)
    
    # Scenario A: Normal Traffic
    logger.info("Scenario A: Normal Traffic (Avg 30 km/h)")
    normal_batch = list(np.random.normal(30.5, 5, 200))
    res_normal = detector.check_data_drift(normal_batch)
    logger.info(f"Normal Check: Drift Detected? {res_normal['drift_detected']} (p-value: {res_normal.get('p_value', 1.0):.4f})")
    
    # Scenario B: Sudden Congestion Shift (Drift)
    logger.info("Scenario B: Sudden Congestion Shift (Avg 15 km/h)")
    congested_batch = list(np.random.normal(15, 4, 200))
    res_drift = detector.check_data_drift(congested_batch)
    logger.info(f"Drift Check: Drift Detected? {res_drift['drift_detected']} (p-value: {res_drift.get('p_value', 1.0):.4f})")
    
    if res_drift['drift_detected']:
        logger.info("✅ Drift Detector correctly identified the distribution shift.")
    else:
        logger.error("❌ Drift Detector failed to identify the shift.")

    # 2. Test A/B Testing Framework
    logger.info("\n📋 Step 2: Testing A/B Testing Framework (Champion vs Challenger)")
    ab_tester = ABTestingFramework(metrics_window=50)
    
    # Simulate Evaluation: Challenger is better (less noise)
    logger.info("Simulating evaluation of shadow model (Challenger) vs production (Champion)")
    for _ in range(100):
        actual = np.random.uniform(20, 40)
        # Champion has 5 km/h noise
        champ_pred = actual + np.random.normal(0, 5)
        # Challenger has 2 km/h noise (Better!)
        chall_pred = actual + np.random.normal(0, 2)
        
        ab_tester.record_prediction('champion', champ_pred, actual)
        ab_tester.record_prediction('challenger', chall_pred, actual)
        
    summary = ab_tester.get_summary()
    logger.info(summary)
    
    if "ACTION: Challenger shows significant improvement" in summary:
        logger.info("✅ A/B Framework correctly identified a superior shadow model.")
    else:
        logger.error("❌ A/B Framework failed to detect shadow model improvement.")

    # 4. Generate Sexy Visualizations
    logger.info("\n📊 Step 4: Generating Advanced ML Pipeline Plots...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    # Plot A: Drift Visualization (Distribution Shift)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(ref_speeds, label="Reference (Normal Traffic)", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(congested_batch, label="Drifted (Sudden Congestion)", fill=True, color="red", alpha=0.3)
    plt.title("Day 8: Real-Time Data Drift Detection (KS-Test)", fontsize=14, fontweight='bold')
    plt.xlabel("Traffic Speed (km/h)")
    plt.ylabel("Density")
    plt.axvline(np.mean(ref_speeds), color='blue', linestyle='--', alpha=0.8)
    plt.axvline(np.mean(congested_batch), color='red', linestyle='--', alpha=0.8)
    plt.legend()
    
    drift_plot_path = Path("lahore/data/plots/drift_detection_shift.png")
    drift_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(drift_plot_path, dpi=150)
    logger.info(f"✅ Drift visualization saved to {drift_plot_path}")

    # Plot B: A/B Testing - Cumulative Error Comparison
    plt.figure(figsize=(10, 6))
    champ_cum_error = np.cumsum([e[0] for e in ab_tester.champion_errors])
    chall_cum_error = np.cumsum([e[0] for e in ab_tester.challenger_errors])
    
    plt.plot(champ_cum_error, label="Champion (Production)", color="gray", linewidth=2, alpha=0.8)
    plt.plot(chall_cum_error, label="Challenger (Proposed)", color="green", linewidth=3)
    
    plt.fill_between(range(len(champ_cum_error)), chall_cum_error, champ_cum_error, color='green', alpha=0.1, label="Saved Inaccuracy")
    
    plt.title("A/B Testing: Cumulative Prediction Error (Champion vs Challenger)", fontsize=14, fontweight='bold')
    plt.xlabel("Inference Cycles (Samples)")
    plt.ylabel("Cumulative MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ab_plot_path = Path("lahore/data/plots/ab_testing_performance.png")
    plt.savefig(ab_plot_path, dpi=150)
    logger.info(f"✅ A/B testing plot saved to {ab_plot_path}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ Day 8 Advanced ML Pipeline Verification Complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_verification()
