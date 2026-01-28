"""
Interactive Route Demo for Lahore Traffic Intelligence System.
Creates a Folium map with known landmarks and demonstrates routing.
Exports both HTML (interactive) and PNG (static) versions.
"""
import folium
from folium import plugins
import pickle
import logging
import os
import time
from lahore.src.optimization.route_optimizer import RouteOptimizer
from lahore.src.optimization.congestion_router import CongestionRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Known Lahore Landmarks with approximate coordinates
LAHORE_LANDMARKS = {
    "Minar-e-Pakistan": (31.5925, 74.3095),
    "Badshahi Mosque": (31.5880, 74.3107),
    "Lahore Fort": (31.5880, 74.3150),
    "Data Darbar": (31.5695, 74.3131),
    "Anarkali Bazaar": (31.5630, 74.3225),
    "Mall Road": (31.5560, 74.3300),
    "Gulberg": (31.5150, 74.3520),
    "Model Town": (31.4830, 74.3250),
    "DHA Phase 5": (31.4650, 74.4050),
    "Johar Town": (31.4700, 74.2900),
    "Allama Iqbal Airport": (31.5216, 74.4036),
    "Shaukat Khanum Hospital": (31.4750, 74.4100),
    "Liberty Market": (31.5180, 74.3450),
    "Lahore Museum": (31.5680, 74.3110),
    "Gaddafi Stadium": (31.5134, 74.3384),
}


def save_map_as_png(html_path: str, png_path: str = None, delay: int = 3) -> str:
    """
    Convert a Folium HTML map to PNG using selenium.
    
    Args:
        html_path: Path to the HTML file
        png_path: Output PNG path (defaults to same name as HTML with .png extension)
        delay: Seconds to wait for map to fully render
        
    Returns:
        Path to saved PNG file
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:
        logger.warning("⚠️ selenium not installed. Run: pip install selenium")
        logger.info("Skipping PNG export...")
        return None
    
    if png_path is None:
        png_path = html_path.replace('.html', '.png')
    
    # Get absolute path for file:// URL
    abs_html_path = os.path.abspath(html_path)
    
    logger.info(f"📸 Converting HTML to PNG: {png_path}")
    
    try:
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1400,900")
        
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f"file://{abs_html_path}")
        
        # Wait for map tiles to load
        time.sleep(delay)
        
        # Save screenshot
        driver.save_screenshot(png_path)
        driver.quit()
        
        logger.info(f"✅ PNG saved: {png_path}")
        return png_path
        
    except Exception as e:
        logger.error(f"❌ Failed to convert to PNG: {e}")
        logger.info("Make sure chromedriver is installed and in PATH")
        return None


def find_nearest_node(graph, lat, lon):
    """Find the nearest node in the graph to given lat/lon coordinates."""
    min_dist = float('inf')
    nearest_node = None

    for node, data in graph.nodes(data=True):
        node_lat = data.get('y', 0)
        node_lon = data.get('x', 0)
        dist = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            nearest_node = node

    return nearest_node


def create_interactive_route_map(start_name: str, end_name: str, output_path: str = None, export_png: bool = True):
    """
    Create an interactive Folium map showing the route between two landmarks.

    Args:
        start_name: Name of starting landmark (from LAHORE_LANDMARKS)
        end_name: Name of destination landmark
        output_path: Optional path to save the HTML file
        export_png: If True, also export a PNG version of the map
    """
    if start_name not in LAHORE_LANDMARKS:
        logger.error(f"Unknown landmark: {start_name}")
        logger.info(f"Available landmarks: {list(LAHORE_LANDMARKS.keys())}")
        return None

    if end_name not in LAHORE_LANDMARKS:
        logger.error(f"Unknown landmark: {end_name}")
        logger.info(f"Available landmarks: {list(LAHORE_LANDMARKS.keys())}")
        return None

    start_coords = LAHORE_LANDMARKS[start_name]
    end_coords = LAHORE_LANDMARKS[end_name]

    logger.info(f"🚗 Finding route: {start_name} → {end_name}")
    logger.info(f"   Start: {start_coords}")
    logger.info(f"   End: {end_coords}")

    # Initialize router
    router = CongestionRouter()
    graph = router.optimizer.graph

    # Find nearest nodes to landmarks
    logger.info("Finding nearest road network nodes...")
    start_node = find_nearest_node(graph, start_coords[0], start_coords[1])
    end_node = find_nearest_node(graph, end_coords[0], end_coords[1])

    if not start_node or not end_node:
        logger.error("Could not find nodes near landmarks")
        return None

    logger.info(f"   Start node: {start_node}")
    logger.info(f"   End node: {end_node}")

    # Simulate congestion
    router.simulate_congestion(congestion_percentage=0.15, severity_range=(2.0, 4.0))

    # Get routes
    static_path, static_cost = router.optimizer.get_shortest_path_dijkstra(start_node, end_node, weight='length')
    dynamic_path, dynamic_cost = router.optimizer.get_shortest_path_astar(start_node, end_node, weight='travel_time')

    if not static_path or not dynamic_path:
        logger.warning("Could not find paths between these locations")
        return None

    # Create base map centered on Lahore
    center_lat = (start_coords[0] + end_coords[0]) / 2
    center_lon = (start_coords[1] + end_coords[1]) / 2
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='cartodbdark_matter'
    )

    # Extract path coordinates
    def get_path_coords(path):
        return [[graph.nodes[n]['y'], graph.nodes[n]['x']] for n in path]

    static_coords = get_path_coords(static_path)
    dynamic_coords = get_path_coords(dynamic_path)

    # Add static route (red)
    folium.PolyLine(
        static_coords,
        color='#e94560',
        weight=5,
        opacity=0.8,
        popup=f"Static Route: {len(static_path)} nodes, {static_cost/1000:.2f} km"
    ).add_to(m)

    # Add dynamic route (green)
    folium.PolyLine(
        dynamic_coords,
        color='#16c79a',
        weight=5,
        opacity=0.8,
        popup=f"Congestion-Aware Route: {len(dynamic_path)} nodes"
    ).add_to(m)

    # Add start marker
    folium.Marker(
        location=start_coords,
        popup=f"<b>START: {start_name}</b>",
        icon=folium.Icon(color='green', icon='play', prefix='fa'),
        tooltip=f"Start: {start_name}"
    ).add_to(m)

    # Add end marker
    folium.Marker(
        location=end_coords,
        popup=f"<b>END: {end_name}</b>",
        icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa'),
        tooltip=f"End: {end_name}"
    ).add_to(m)

    # Add other landmarks as reference points
    for name, coords in LAHORE_LANDMARKS.items():
        if name not in [start_name, end_name]:
            folium.CircleMarker(
                location=coords,
                radius=5,
                color='#ffd700',
                fill=True,
                fill_opacity=0.7,
                popup=name,
                tooltip=name
            ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: rgba(0,0,0,0.8); padding: 15px; border-radius: 10px;
                font-family: Arial; color: white; font-size: 14px;">
        <div style="margin-bottom: 10px;"><b>Route Comparison</b></div>
        <div style="margin-bottom: 5px;">
            <span style="background-color: #e94560; padding: 2px 15px; margin-right: 10px;"></span>
            Static Route (Shortest Distance)
        </div>
        <div style="margin-bottom: 5px;">
            <span style="background-color: #16c79a; padding: 2px 15px; margin-right: 10px;"></span>
            Congestion-Aware Route
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #aaa;">
            🟡 Yellow dots = Other Landmarks
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add title
    title_html = f'''
    <div style="position: fixed; top: 20px; left: 50%; transform: translateX(-50%); z-index: 1000;
                background-color: rgba(0,0,0,0.9); padding: 15px 30px; border-radius: 10px;
                font-family: Arial; text-align: center;">
        <div style="color: #16c79a; font-size: 20px; font-weight: bold;">Lahore Route Optimization</div>
        <div style="color: white; font-size: 16px; margin-top: 5px;">
            {start_name} → {end_name}
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    if output_path is None:
        output_path = f"lahore/data/plots/route_demo_{start_name.replace(' ', '_')}_to_{end_name.replace(' ', '_')}.html"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logger.info(f"✅ Interactive map saved to: {output_path}")

    # Export PNG if requested
    png_path = None
    if export_png:
        png_path = save_map_as_png(output_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Route Summary")
    logger.info("=" * 60)
    logger.info(f"Start: {start_name} ({start_coords})")
    logger.info(f"End: {end_name} ({end_coords})")
    logger.info(f"Static Route: {len(static_path)} nodes, {static_cost/1000:.2f} km")
    logger.info(f"Congestion-Aware: {len(dynamic_path)} nodes, {dynamic_cost/1000:.2f} travel cost")
    logger.info(f"Routes Different: {static_path != dynamic_path}")
    logger.info("=" * 60)

    return output_path, png_path


def list_available_landmarks():
    """Print all available landmarks."""
    print("\n🏛️ Available Lahore Landmarks:")
    print("-" * 40)
    for i, (name, coords) in enumerate(LAHORE_LANDMARKS.items(), 1):
        print(f"{i:2}. {name}: {coords}")
    print("-" * 40)


if __name__ == "__main__":
    list_available_landmarks()

    # Demo route: Minar-e-Pakistan to Gaddafi Stadium
    print("\n🚗 Demo: Minar-e-Pakistan → Gaddafi Stadium")
    create_interactive_route_map("Minar-e-Pakistan", "Gaddafi Stadium", export_png=True)

    # Another demo: Data Darbar to DHA Phase 5
    print("\n🚗 Demo: Data Darbar → DHA Phase 5")
    create_interactive_route_map("Data Darbar", "DHA Phase 5", export_png=True)

