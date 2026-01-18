import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine
import geopandas as gpd
from shared.config.database.config import DATABASE_URL

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def generate_plots():
    """Reads data from PostGIS and generates summary plots."""
    print("🎨 Generating validation plots...")
    engine = create_engine(DATABASE_URL)
    
    # Ensure output directory exists
    output_dir = "lahore/data/plots"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Road Network Plot
    print("🗺️  Loading road network...")
    edges_query = "SELECT geometry, highway, length FROM lahore_edges LIMIT 50000"
    edges_gdf = gpd.read_postgis(edges_query, engine, geom_col='geometry')
    
    plt.figure(figsize=(12, 12))
    edges_gdf.plot(column='length', cmap='viridis', legend=True, alpha=0.6)
    plt.title("Lahore Road Network (Sub-sample: Length Distribution)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(f"{output_dir}/road_network.png")
    plt.close()
    print(f"✅ Saved road network map to {output_dir}/road_network.png")

    # 2. Traffic Flow Distribution
    print("📊 Loading traffic history...")
    traffic_query = "SELECT speed, volume, timestamp FROM lahore_traffic_history"
    df = pd.read_sql(traffic_query, engine)
    
    if df.empty:
        print("⚠️  No traffic history found. Skip distribution plots.")
    else:
        # Speed Distribution
        plt.figure()
        sns.histplot(df['speed'], kde=True, color='skyblue')
        plt.title("Traffic Speed Distribution (Simulated)")
        plt.xlabel("Speed (km/h)")
        plt.savefig(f"{output_dir}/speed_dist.png")
        plt.close()
        
        # Volume Distribution
        plt.figure()
        sns.histplot(df['volume'], kde=True, color='salmon')
        plt.title("Traffic Volume Distribution (Simulated)")
        plt.xlabel("Vehicle Volume")
        plt.savefig(f"{output_dir}/volume_dist.png")
        plt.close()
        
        # Temporal Plot (Aggregation by timestamp)
        plt.figure()
        temp_df = df.groupby('timestamp')['speed'].mean().reset_index()
        sns.lineplot(data=temp_df, x='timestamp', y='speed', marker='o')
        plt.title("Average Traffic Speed Over Time (Simulation Sequence)")
        plt.xlabel("Timestamp")
        plt.ylabel("Avg Speed")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/temporal_speed.png")
        plt.close()
        
        print(f"✅ Saved statistical plots to {output_dir}/")

if __name__ == "__main__":
    generate_plots()
