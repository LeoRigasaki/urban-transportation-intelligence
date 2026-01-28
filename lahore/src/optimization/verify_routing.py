"""
Verification Script for Route Optimization Module.
Benchmarks and validates all routing algorithms.
"""
import time
import logging
import random
import pickle
from typing import Dict, Any

from lahore.src.optimization.route_optimizer import RouteOptimizer
from lahore.src.optimization.genetic_optimizer import GeneticRouteOptimizer
from lahore.src.optimization.congestion_router import CongestionRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def benchmark_algorithm(func, *args, **kwargs) -> tuple:
    """
    Benchmark a routing function.
    Returns (result, execution_time_ms).
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
    return result, elapsed


def run_verification() -> Dict[str, Any]:
    """
    Run comprehensive verification of all routing algorithms.
    """
    logger.info("=" * 60)
    logger.info("🚀 Starting Route Optimization Verification")
    logger.info("=" * 60)

    results = {
        'dijkstra': {},
        'astar': {},
        'genetic': {},
        'congestion_comparison': {}
    }

    # 1. Load graph and initialize optimizers
    logger.info("\n📂 Loading graph...")
    graph_path = "lahore/models/trained/lahore_graph.pickle"

    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"✅ Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except FileNotFoundError:
        logger.error(f"❌ Graph file not found: {graph_path}")
        logger.info("Please run: python lahore/src/data_pipeline/graph.py first")
        return {'error': 'Graph not found'}

    # Initialize optimizers
    route_optimizer = RouteOptimizer(graph_path)
    genetic_optimizer = GeneticRouteOptimizer(graph, population_size=30, generations=30)
    congestion_router = CongestionRouter(graph_path)

    # 2. Select test node pairs
    logger.info("\n🎯 Selecting test node pairs...")
    nodes = list(graph.nodes())
    test_pairs = []
    for _ in range(3):  # Test 3 random pairs
        source = random.choice(nodes)
        target = random.choice(nodes)
        while target == source:
            target = random.choice(nodes)
        test_pairs.append((source, target))
    logger.info(f"Test pairs: {test_pairs}")

    # 3. Benchmark Dijkstra
    logger.info("\n📊 Benchmarking Dijkstra's Algorithm...")
    dijkstra_times = []
    for source, target in test_pairs:
        (path, cost), elapsed = benchmark_algorithm(
            route_optimizer.get_shortest_path_dijkstra, source, target
        )
        dijkstra_times.append(elapsed)
        if path:
            logger.info(f"  {source} -> {target}: {len(path)} nodes, cost={cost:.2f}, time={elapsed:.2f}ms")
        else:
            logger.warning(f"  {source} -> {target}: No path found")

    results['dijkstra'] = {
        'avg_time_ms': sum(dijkstra_times) / len(dijkstra_times),
        'min_time_ms': min(dijkstra_times),
        'max_time_ms': max(dijkstra_times)
    }

    # 4. Benchmark A*
    logger.info("\n📊 Benchmarking A* Algorithm...")
    astar_times = []
    for source, target in test_pairs:
        (path, cost), elapsed = benchmark_algorithm(
            route_optimizer.get_shortest_path_astar, source, target
        )
        astar_times.append(elapsed)
        if path:
            logger.info(f"  {source} -> {target}: {len(path)} nodes, cost={cost:.2f}, time={elapsed:.2f}ms")
        else:
            logger.warning(f"  {source} -> {target}: No path found")

    results['astar'] = {
        'avg_time_ms': sum(astar_times) / len(astar_times),
        'min_time_ms': min(astar_times),
        'max_time_ms': max(astar_times)
    }

    # 5. Benchmark Genetic Algorithm (single test due to runtime)
    logger.info("\n📊 Benchmarking Genetic Algorithm...")
    source, target = test_pairs[0]
    (path, fitness, stats), elapsed = benchmark_algorithm(
        genetic_optimizer.optimize, source, target
    )
    if path:
        logger.info(f"  {source} -> {target}: {len(path)} nodes, fitness={fitness:.2f}, time={elapsed:.2f}ms")
        logger.info(f"  GA Stats: {stats}")
    else:
        logger.warning(f"  {source} -> {target}: No path found")

    results['genetic'] = {
        'time_ms': elapsed,
        'fitness': fitness if path else None,
        'stats': stats
    }

    # 6. Congestion-Aware Routing Comparison
    logger.info("\n📊 Testing Congestion-Aware Routing...")
    congestion_router.simulate_congestion(congestion_percentage=0.2, severity_range=(2.0, 5.0))

    comparison = congestion_router.compare_routes(source, target)
    results['congestion_comparison'] = comparison

    logger.info(f"  Static Route: {comparison['static_route']}")
    logger.info(f"  Congestion-Aware Route: {comparison['congestion_aware_route']}")
    logger.info(f"  Routes Different: {comparison['paths_different']}")
    if comparison['paths_different']:
        logger.info(f"  ✅ Congestion-aware routing successfully diverts traffic!")
        logger.info(f"  Time Saved: {comparison['time_saved']:.2f}")

    # 7. Summary
    logger.info("\n" + "=" * 60)
    logger.info("📈 VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dijkstra Avg Time: {results['dijkstra']['avg_time_ms']:.2f}ms")
    logger.info(f"A* Avg Time: {results['astar']['avg_time_ms']:.2f}ms")
    logger.info(f"GA Time: {results['genetic']['time_ms']:.2f}ms")
    logger.info(f"Congestion Diversion: {'✅ Working' if comparison['paths_different'] else '⚠️ Same path'}")
    logger.info("=" * 60)
    logger.info("✅ Route Optimization Verification Complete!")

    return results


if __name__ == "__main__":
    run_verification()
