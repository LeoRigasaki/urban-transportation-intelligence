import folium
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import os
from shared.config.database.config import DATABASE_URL

def create_congestion_map():
    """
    Generates an interactive Leaflet map (HTML) showing traffic congestion.
    """
    print("🌍 Generating interactive congestion map...")
    engine = create_engine(DATABASE_URL)
    
    # 1. Load latest traffic state (Average speed per edge)
    traffic_query = """
    SELECT u, v, key, AVG(speed) as avg_speed 
    FROM lahore_traffic_history 
    GROUP BY u, v, key
    """
    traffic_df = pd.read_sql(traffic_query, engine)
    
    if traffic_df.empty:
        print("⚠️  No traffic data available. Map will only show roads.")
    
    # 2. Load road geometries
    # Filtering for main roads to keep map size manageable
    road_query = """
    SELECT u, v, key, geometry, maxspeed, highway 
    FROM lahore_edges 
    WHERE highway IN ('motorway', 'trunk', 'primary', 'secondary', 'tertiary')
    """
    roads_gdf = gpd.read_postgis(road_query, engine, geom_col='geometry')
    
    # 3. Merge traffic data onto roads
    merged_gdf = roads_gdf.merge(traffic_df, on=['u', 'v', 'key'], how='left')
    
    # Convert maxspeed to numeric for congestion calculations
    merged_gdf['maxspeed'] = pd.to_numeric(merged_gdf['maxspeed'], errors='coerce').fillna(40)
    
    # 4. Define Color Mapping (Speed % of Maxspeed)
    def get_color(row):
        if pd.isna(row['avg_speed']):
            return '#808080' # Gray for no data
        
        ratio = row['avg_speed'] / row['maxspeed']
        if ratio < 0.3:
            return '#d7191c' # Dark Red (Heavy Congestion)
        elif ratio < 0.6:
            return '#fdae61' # Orange (Moderate)
        elif ratio < 0.8:
            return '#a6d96a' # Light Green (Light)
        else:
            return '#1a9641' # Green (Free Flow)

    merged_gdf['color'] = merged_gdf.apply(get_color, axis=1)

    # 5. Generate Static PNG Map (Professional standard for README)
    print("🖼️  Generating static congestion plot...")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(15, 15))
    # Plot background roads in light gray
    merged_gdf[merged_gdf['avg_speed'].isna()].plot(ax=ax, color='#e0e0e0', linewidth=0.5, alpha=0.5)
    # Plot traffic data with colors
    merged_gdf[merged_gdf['avg_speed'].notna()].plot(ax=ax, color=merged_gdf[merged_gdf['avg_speed'].notna()]['color'], linewidth=2)
    
    ax.set_title("Lahore Geospatial Congestion Analysis", fontsize=15)
    ax.set_axis_off()
    
    output_png = "lahore/data/plots/congestion_static.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Static map saved to {output_png}")

    # 6. Build Folium Map (Optional/Local)
    m = folium.Map(location=[31.5204, 74.3587], zoom_start=12, tiles="cartodbpositron")
    
    # Add road segments to map
    for _, row in merged_gdf.iterrows():
        # Extact coords from geometry (Linestring)
        # GeoPandas geometries are in lon, lat - Folium needs lat, lon
        coords = [(lat, lon) for lon, lat in row['geometry'].coords]
        
        popup_text = f"Road: {row['highway']}<br>Speed: {row['avg_speed']:.1f} / {row['maxspeed']} km/h" if not pd.isna(row['avg_speed']) else "No data"
        
        folium.PolyLine(
            locations=coords,
            color=row['color'],
            weight=3,
            opacity=0.8,
            popup=popup_text
        ).add_to(m)

    # Save to data folder
    output_path = "lahore/data/plots/congestion_map.html"
    m.save(output_path)
    print(f"✅ Interactive map saved to {output_path}")

if __name__ == "__main__":
    create_congestion_map()
