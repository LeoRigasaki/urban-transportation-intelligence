import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import logging
from shared.config.database.config import DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features():
    """
    Extracts spatial features from the road network.
    """
    logger.info("Starting feature engineering for Lahore network...")
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # Load edges (road segments)
        logger.info("Loading road segments from PostGIS...")
        edges_df = gpd.read_postgis("SELECT * FROM lahore_edges", engine, geom_col="geometry")
        
        # Feature Engineering Ideas:
        # 1. Road Complexity (number of points in geometry)
        logger.info("Computing road complexity...")
        edges_df['point_count'] = edges_df['geometry'].map(lambda g: len(g.coords))
        
        # 2. Highway Type Encoding (basic)
        logger.info("Encoding highway types...")
        edges_df['highway_type'] = edges_df['highway'].astype(str)
        
        # 3. Calculate road density/clustering (simplified for now)
        # We can count edges per "area" but let's keep it simple for Day 2
        
        # Select important columns for the features table
        features_df = edges_df[['u', 'v', 'key', 'length', 'highway_type', 'point_count', 'geometry']]
        
        # Save to database
        logger.info("Saving features to 'lahore_features' table...")
        features_df.to_postgis("lahore_features", engine, if_exists="replace", index=False)
        
        logger.info(f"✅ Processed {len(features_df)} features and saved to database.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature engineering failed: {e}")
        return False

if __name__ == "__main__":
    extract_features()
