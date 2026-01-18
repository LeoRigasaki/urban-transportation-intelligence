import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import logging
from shared.config.database.config import DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_validation():
    """
    Validates data quality for the processed road network.
    """
    logger.info("Starting data quality validation for Lahore network...")
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # 1. Connectivity Check (via nodes/edges existence)
        nodes_count = pd.read_sql("SELECT COUNT(*) FROM lahore_nodes", engine).iloc[0,0]
        edges_count = pd.read_sql("SELECT COUNT(*) FROM lahore_edges", engine).iloc[0,0]
        
        if nodes_count > 0 and edges_count > 0:
            logger.info(f"✅ Basic connectivity: {nodes_count} nodes, {edges_count} edges found.")
        else:
            logger.error("❌ Basic connectivity failed: Empty tables.")
            
        # 2. Features Check
        features_count = pd.read_sql("SELECT COUNT(*) FROM lahore_features", engine).iloc[0,0]
        if features_count == edges_count:
            logger.info(f"✅ Feature consistency: {features_count} feature entries match edge count.")
        else:
            logger.warning(f"⚠️ Feature mismatch: {features_count} features vs {edges_count} edges.")
            
        # 3. Spatial Validity
        logger.info("Checking spatial validity of geometries...")
        invalid_count = pd.read_sql("SELECT COUNT(*) FROM lahore_features WHERE ST_IsValid(geometry) = false", engine).iloc[0,0]
        if invalid_count == 0:
            logger.info("✅ All geometries are spatially valid.")
        else:
            logger.error(f"❌ Found {invalid_count} invalid geometries.")
            
        logger.info("🚀 Data quality validation complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    run_validation()
