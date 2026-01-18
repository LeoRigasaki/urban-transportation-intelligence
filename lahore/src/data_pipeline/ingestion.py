import osmnx as ox
import logging
from sqlalchemy import create_engine
import geopandas as gpd
from shared.config.database.config import DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ingest_lahore_roads():
    """
    Downloads Lahore road network from OpenStreetMap and saves to PostgreSQL.
    """
    logger.info("Starting Lahore road network ingestion...")
    
    try:
        # Define the location - using District for better polygon match
        place_name = "Lahore District, Pakistan"
        logger.info(f"Downloading network for {place_name}...")
        
        # Download the graph
        # Using network_type='drive' for all drivable roads
        G = ox.graph_from_place(place_name, network_type='drive')
        
        # Convert to GeoDataFrames
        nodes, edges = ox.graph_to_gdfs(G)
        
        logger.info(f"Successfully downloaded {len(nodes)} nodes and {len(edges)} edges.")
        
        # Save to PostgreSQL
        engine = create_engine(DATABASE_URL)
        
        logger.info("Saving nodes to database...")
        # Note: In a production environment, we'd use GeoAlchemy2 or PostGIS specific methods
        # For Day 1, we use geopandas.to_postgis if available, or simple to_sql
        # Ensuring we have geometry support
        nodes.to_postgis("lahore_nodes", engine, if_exists="replace", index=True)
        
        logger.info("Saving edges to database...")
        edges.to_postgis("lahore_edges", engine, if_exists="replace", index=True)
        
        logger.info("✅ Lahore road network ingestion complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}")
        return False

if __name__ == "__main__":
    ingest_lahore_roads()
