import networkx as nx
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
import logging
import pickle
import os
from shared.config.database.config import DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_lahore_graph():
    """
    Constructs a NetworkX graph from PostGIS road network data.
    """
    logger.info("Starting Lahore network graph construction...")
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # Load nodes and edges from PostGIS
        logger.info("Loading data from PostGIS...")
        nodes_df = gpd.read_postgis("SELECT * FROM lahore_nodes", engine, geom_col="geometry", index_col="osmid")
        edges_df = gpd.read_postgis("SELECT * FROM lahore_edges", engine, geom_col="geometry")
        
        logger.info(f"Loaded {len(nodes_df)} nodes and {len(edges_df)} edges.")
        
        # Create MultiDiGraph
        G = nx.MultiDiGraph()
        
        # Add nodes with attributes
        for osmid, data in nodes_df.iterrows():
            G.add_node(osmid, x=data['x'], y=data['y'], **data.drop(['x', 'y', 'geometry']).to_dict())
            
        # Add edges with attributes
        # edges_df has columns 'u' and 'v' which are the node IDs
        for _, data in edges_df.iterrows():
            u, v = data['u'], data['v']
            attr = data.drop(['u', 'v', 'geometry']).to_dict()
            G.add_edge(u, v, **attr)
            
        logger.info(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        
        # Basic Topology Analysis
        logger.info("Performing basic topology analysis...")
        is_connected = nx.is_weakly_connected(G)
        logger.info(f"Is weakly connected: {is_connected}")
        
        if not is_connected:
            components = list(nx.weakly_connected_components(G))
            logger.info(f"Number of weakly connected components: {len(components)}")
            main_component_size = len(max(components, key=len))
            logger.info(f"Main component size: {main_component_size} nodes")

        # Save the graph for downstream tasks
        # We'll save it to the models/trained directory as a base for future models
        save_path = "lahore/models/trained/lahore_graph.pickle"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(G, f)
            
        logger.info(f"✅ Graph saved to {save_path}")
        return G
        
    except Exception as e:
        logger.error(f"❌ Graph construction failed: {e}")
        return None

if __name__ == "__main__":
    build_lahore_graph()
