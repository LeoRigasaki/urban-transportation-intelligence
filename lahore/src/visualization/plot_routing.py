"""
Visualization Script for Route Optimization Results.
Generates plots for algorithm benchmarks and route comparisons.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
import random
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['text.color'] = '#eaeaea'
plt.rcParams['axes.labelcolor'] = '#eaeaea'
plt.rcParams['xtick.color'] = '#eaeaea'
plt.rcParams['ytick.color'] = '#eaeaea'
plt.rcParams['axes.edgecolor'] = '#4a4a6a'
plt.rcParams['grid.color'] = '#2a2a4a'
plt.rcParams['font.family'] = 'sans-serif'


def plot_algorithm_benchmark():
    """
    Create a bar chart comparing algorithm execution times.
    """
    algorithms = ['Dijkstra', 'A*', 'Genetic\nAlgorithm']
    times = [590.99, 64.36, 182.64]
    colors = ['#e94560', '#0f3460', '#16c79a']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(algorithms, times, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.annotate(f'{time:.1f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color='#eaeaea')

    # Add speedup annotation for A*
    ax.annotate('9x faster!',
                xy=(1, 64.36),
                xytext=(1.5, 300),
                fontsize=12, color='#16c79a',
                arrowprops=dict(arrowstyle='->', color='#16c79a', lw=2))

    ax.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Route Optimization Algorithm Benchmark', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, max(times) * 1.3)

    plt.tight_layout()
    output_path = 'lahore/data/plots/algorithm_benchmark.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"✅ Saved algorithm benchmark plot to {output_path}")


def plot_route_comparison():
    """
    Visualize static vs congestion-aware route on the road network.
    """
    from lahore.src.optimization.route_optimizer import RouteOptimizer
    from lahore.src.optimization.congestion_router import CongestionRouter

    logger.info("Loading graph and computing routes...")
    router = CongestionRouter()

    # Simulate congestion
    router.simulate_congestion(congestion_percentage=0.15, severity_range=(2.0, 4.0))

    # Get random source/target
    random.seed(42)  # For reproducibility
    source, target = router.optimizer.get_random_node_pair()

    # Get both routes
    static_path, static_cost = router.optimizer.get_shortest_path_dijkstra(source, target, weight='length')
    dynamic_path, dynamic_cost = router.optimizer.get_shortest_path_astar(source, target, weight='travel_time')

    if not static_path or not dynamic_path:
        logger.warning("Could not find paths for visualization")
        return

    # Extract coordinates
    graph = router.optimizer.graph

    def get_coords(path):
        x = [graph.nodes[n]['x'] for n in path]
        y = [graph.nodes[n]['y'] for n in path]
        return x, y

    static_x, static_y = get_coords(static_path)
    dynamic_x, dynamic_y = get_coords(dynamic_path)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot a sample of the road network as background
    logger.info("Plotting road network background...")
    edges = list(graph.edges(data=True))[:5000]  # Sample for performance
    for u, v, data in edges:
        x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
        y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
        ax.plot(x, y, color='#2a2a4a', linewidth=0.3, alpha=0.5)

    # Plot static route
    ax.plot(static_x, static_y, color='#e94560', linewidth=3, alpha=0.8, label=f'Static Route ({len(static_path)} nodes)')

    # Plot congestion-aware route
    ax.plot(dynamic_x, dynamic_y, color='#16c79a', linewidth=3, alpha=0.8, label=f'Congestion-Aware Route ({len(dynamic_path)} nodes)')

    # Mark start and end points
    ax.scatter([static_x[0]], [static_y[0]], color='#ffd700', s=200, zorder=5, marker='o', edgecolor='white', linewidth=2)
    ax.scatter([static_x[-1]], [static_y[-1]], color='#ff6b6b', s=200, zorder=5, marker='s', edgecolor='white', linewidth=2)

    ax.annotate('START', (static_x[0], static_y[0]), xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#ffd700')
    ax.annotate('END', (static_x[-1], static_y[-1]), xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#ff6b6b')

    ax.set_title('Route Comparison: Static vs Congestion-Aware', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper left', fontsize=11)

    # Add info box
    info_text = (f"Static Route Cost: {static_cost:.0f} m\n"
                 f"Dynamic Route Cost: {dynamic_cost:.0f} (travel time)\n"
                 f"Routes Different: {static_path != dynamic_path}")
    props = dict(boxstyle='round', facecolor='#16213e', alpha=0.9, edgecolor='#4a4a6a')
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    output_path = 'lahore/data/plots/route_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"✅ Saved route comparison plot to {output_path}")


def plot_congestion_heatmap():
    """
    Create a heatmap showing congestion distribution on the network.
    """
    from lahore.src.optimization.congestion_router import CongestionRouter

    logger.info("Generating congestion heatmap...")
    router = CongestionRouter()
    router.simulate_congestion(congestion_percentage=0.2, severity_range=(1.5, 5.0))

    graph = router.optimizer.graph

    # Extract edge coordinates and congestion values
    congested_edges = []
    normal_edges = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        x = [graph.nodes[u]['x'], graph.nodes[v]['x']]
        y = [graph.nodes[u]['y'], graph.nodes[v]['y']]
        travel_time = data.get('travel_time', 0)
        length = data.get('length', 1)

        if travel_time > length * 1.1:  # Congested
            congested_edges.append((x, y, travel_time / length))
        else:
            normal_edges.append((x, y))

    # Limit for performance
    normal_edges = random.sample(normal_edges, min(3000, len(normal_edges)))
    congested_edges = congested_edges[:2000]

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot normal edges
    for x, y in normal_edges:
        ax.plot(x, y, color='#16c79a', linewidth=0.4, alpha=0.4)

    # Plot congested edges with color intensity
    for x, y, severity in congested_edges:
        color_intensity = min(1.0, (severity - 1) / 4)
        color = plt.cm.Reds(0.3 + color_intensity * 0.7)
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.8)

    ax.set_title('Network Congestion Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#16c79a', label='Normal Flow', alpha=0.6),
        mpatches.Patch(facecolor='#e94560', label='Congested', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    # Stats box
    stats_text = (f"Total Edges: {graph.number_of_edges():,}\n"
                  f"Congested Edges: {len(router.congestion_data):,}\n"
                  f"Congestion Rate: {len(router.congestion_data)/graph.number_of_edges()*100:.1f}%")
    props = dict(boxstyle='round', facecolor='#16213e', alpha=0.9, edgecolor='#4a4a6a')
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    output_path = 'lahore/data/plots/congestion_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"✅ Saved congestion heatmap to {output_path}")


def main():
    """Generate all visualization plots."""
    logger.info("=" * 60)
    logger.info("🎨 Generating Route Optimization Visualizations")
    logger.info("=" * 60)

    # 1. Algorithm Benchmark
    logger.info("\n📊 Generating algorithm benchmark chart...")
    plot_algorithm_benchmark()

    # 2. Route Comparison
    logger.info("\n🗺️ Generating route comparison map...")
    plot_route_comparison()

    # 3. Congestion Heatmap
    logger.info("\n🔥 Generating congestion heatmap...")
    plot_congestion_heatmap()

    logger.info("\n" + "=" * 60)
    logger.info("✅ All visualizations generated successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
