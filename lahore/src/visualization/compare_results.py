import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd

def plot_comparison():
    """
    Reads saved metrics from models/results and generates comparison plots.
    """
    results_dir = "lahore/models/results"
    output_dir = "lahore/data/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_files = [f for f in os.listdir(results_dir) if f.endswith("_metrics.json")]
    
    comparison_data = []
    for f in metrics_files:
        model_name = f.replace("_metrics.json", "").upper()
        with open(os.path.join(results_dir, f), "r") as json_file:
            history = json.load(json_file)
            last_epoch = history[-1]
            comparison_data.append({
                "Model": model_name,
                "MAE": last_epoch["MAE"],
                "RMSE": last_epoch["RMSE"],
                "MAPE (%)": last_epoch["MAPE"]
            })
    
    if not comparison_data:
        print("No results found to compare.")
        return
        
    df = pd.DataFrame(comparison_data)
    
    # 1. MAPE Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="MAPE (%)", palette="viridis")
    plt.title("Model Performance Comparison (MAPE)")
    plt.ylabel("Mean Absolute Percentage Error (%)")
    plt.savefig(f"{output_dir}/model_comparison_mape.png")
    plt.close()

    # 2. RMSE/MAE Comparison
    df_melted = df.melt(id_vars="Model", value_vars=["MAE", "RMSE"], var_name="Metric", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", palette="magma")
    plt.title("Model Accuracy Metrics (MAE vs RMSE)")
    plt.savefig(f"{output_dir}/model_metrics_comparison.png")
    plt.close()

    print(f"✅ Comparison plots saved to {output_dir}")

if __name__ == "__main__":
    plot_comparison()
