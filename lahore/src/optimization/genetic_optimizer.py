"""
Genetic Algorithm Route Optimizer for Lahore Traffic Intelligence System.
Implements a simple GA for multi-objective path optimization (time + distance).
"""
import random
import logging
from typing import List, Tuple, Optional, Dict, Any
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GeneticRouteOptimizer:
    """
    A Genetic Algorithm-based route optimizer for multi-objective path finding.
    Optimizes for a combination of travel time and distance.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        time_weight: float = 0.6,
        distance_weight: float = 0.4
    ):
        """
        Initialize the Genetic Algorithm optimizer.

        Args:
            graph: NetworkX MultiDiGraph with edge weights.
            population_size: Number of individuals in the population.
            generations: Number of generations to run.
            mutation_rate: Probability of mutation per individual.
            crossover_rate: Probability of crossover between parents.
            time_weight: Weight for travel time in fitness (0-1).
            distance_weight: Weight for distance in fitness (0-1).
        """
        self.graph = graph
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.time_weight = time_weight
        self.distance_weight = distance_weight

    def _get_neighbors(self, node: int) -> List[int]:
        """Get all neighbors of a node."""
        return list(self.graph.successors(node))

    def _generate_random_path(self, source: int, target: int, max_steps: int = 1000) -> Optional[List[int]]:
        """
        Generate a random valid path from source to target using random walk.
        Returns None if no path found within max_steps.
        """
        path = [source]
        current = source
        visited = {source}
        steps = 0

        while current != target and steps < max_steps:
            neighbors = [n for n in self._get_neighbors(current) if n not in visited]
            if not neighbors:
                # Dead end, restart
                return None
            current = random.choice(neighbors)
            path.append(current)
            visited.add(current)
            steps += 1

        return path if current == target else None

    def _initialize_population(self, source: int, target: int) -> List[List[int]]:
        """
        Initialize the population with random valid paths.
        Uses shortest path as a seed to ensure at least one valid solution.
        """
        population = []

        # Add the Dijkstra shortest path as a seed
        try:
            seed_path = nx.dijkstra_path(self.graph, source, target, weight='length')
            population.append(seed_path)
        except nx.NetworkXNoPath:
            logger.warning("No seed path found via Dijkstra")

        # Generate random paths
        attempts = 0
        max_attempts = self.population_size * 10
        while len(population) < self.population_size and attempts < max_attempts:
            path = self._generate_random_path(source, target)
            if path:
                population.append(path)
            attempts += 1

        if len(population) < 2:
            logger.warning(f"Could only generate {len(population)} valid paths")

        return population

    def _calculate_path_cost(self, path: List[int]) -> Tuple[float, float]:
        """
        Calculate the total distance and travel time of a path.

        Returns:
            Tuple of (total_distance, total_travel_time)
        """
        total_distance = 0.0
        total_time = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                edge_data = self.graph[u][v][0]  # Get first edge in MultiDiGraph
                total_distance += edge_data.get('length', 100)
                total_time += edge_data.get('travel_time', edge_data.get('length', 100))

        return total_distance, total_time

    def _fitness(self, path: List[int]) -> float:
        """
        Calculate fitness of a path. Lower is better.
        Combines distance and travel time with configurable weights.
        """
        distance, time = self._calculate_path_cost(path)
        # Penalize very long paths
        length_penalty = len(path) * 0.01
        return (self.distance_weight * distance + self.time_weight * time) + length_penalty

    def _select_parents(self, population: List[List[int]], fitnesses: List[float]) -> Tuple[List[int], List[int]]:
        """
        Tournament selection to choose two parents.
        """
        def tournament(k: int = 3) -> List[int]:
            candidates = random.sample(list(zip(population, fitnesses)), min(k, len(population)))
            return min(candidates, key=lambda x: x[1])[0]

        return tournament(), tournament()

    def _crossover(self, parent1: List[int], parent2: List[int]) -> Optional[List[int]]:
        """
        Single-point crossover that maintains path validity.
        Finds common nodes between parents and crosses over at a random common point.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()

        # Find common nodes
        common = set(parent1) & set(parent2)
        if len(common) <= 2:  # Only source and target are common
            return parent1.copy()

        common_list = [n for n in parent1 if n in common and n != parent1[0] and n != parent1[-1]]
        if not common_list:
            return parent1.copy()

        crossover_point = random.choice(common_list)
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)

        # Create child: start of parent1 + end of parent2
        child = parent1[:idx1] + parent2[idx2:]

        # Validate path connectivity
        for i in range(len(child) - 1):
            if not self.graph.has_edge(child[i], child[i + 1]):
                return parent1.copy()  # Invalid path, return parent

        return child

    def _mutate(self, path: List[int]) -> List[int]:
        """
        Mutation: randomly replace a segment of the path with a new random walk.
        """
        if random.random() > self.mutation_rate or len(path) < 3:
            return path

        # Select a random point (not source or target) and try to find alternative route
        idx = random.randint(1, len(path) - 2)
        node = path[idx]
        target = path[-1]

        # Try to find alternative path from this node
        alt_path = self._generate_random_path(node, target, max_steps=50)
        if alt_path:
            return path[:idx] + alt_path

        return path

    def optimize(self, source: int, target: int) -> Tuple[Optional[List[int]], float, Dict[str, Any]]:
        """
        Run the genetic algorithm to find an optimized route.

        Args:
            source: Source node ID.
            target: Target node ID.

        Returns:
            Tuple of (best_path, best_fitness, stats_dict)
        """
        logger.info(f"Starting GA optimization: {source} -> {target}")

        # Initialize population
        population = self._initialize_population(source, target)
        if not population:
            logger.error("Failed to initialize population")
            return None, float('inf'), {}

        best_path = None
        best_fitness = float('inf')
        history = []

        for gen in range(self.generations):
            # Calculate fitness for all individuals
            fitnesses = [self._fitness(p) for p in population]

            # Track best
            gen_best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_path = population[gen_best_idx].copy()

            history.append(best_fitness)

            # Early stopping if fitness is stable
            if gen > 20 and len(set(history[-10:])) == 1:
                logger.info(f"Early stopping at generation {gen}")
                break

            # Create new population
            new_population = [best_path.copy()]  # Elitism: keep best

            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents(population, fitnesses)
                child = self._crossover(parent1, parent2)
                if child:
                    child = self._mutate(child)
                    new_population.append(child)

            population = new_population

        distance, time = self._calculate_path_cost(best_path) if best_path else (0, 0)
        stats = {
            'generations_run': gen + 1,
            'final_population_size': len(population),
            'best_distance': distance,
            'best_time': time,
            'path_length': len(best_path) if best_path else 0
        }

        logger.info(f"✅ GA Optimization complete: fitness={best_fitness:.2f}, path_length={len(best_path) if best_path else 0}")
        return best_path, best_fitness, stats


if __name__ == "__main__":
    import pickle

    # Load graph
    with open("lahore/models/trained/lahore_graph.pickle", 'rb') as f:
        graph = pickle.load(f)

    optimizer = GeneticRouteOptimizer(graph, population_size=30, generations=50)

    # Get random nodes
    nodes = list(graph.nodes())
    source = random.choice(nodes)
    target = random.choice(nodes)
    while target == source:
        target = random.choice(nodes)

    logger.info(f"Testing GA route from {source} to {target}")
    path, fitness, stats = optimizer.optimize(source, target)

    if path:
        logger.info(f"✅ GA found path: {len(path)} nodes, fitness: {fitness:.2f}")
        logger.info(f"   Stats: {stats}")
