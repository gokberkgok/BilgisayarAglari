"""
Genetic Algorithm Solver for QoS-based Multi-Objective Routing.
Adapter for integration with NetworkGraph format.
"""

import random
import math
from .network_model import NetworkGraph

# Default GA Parameters
DEFAULT_POP_SIZE = 60
DEFAULT_GENERATIONS = 120
DEFAULT_MUTATION_RATE = 0.2


class GA_Solver:
    """
    Genetic Algorithm Solver adapted for NetworkGraph format.
    Uses evolutionary optimization to find optimal QoS paths.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        source: int,
        destination: int,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.33,
        pop_size: int = DEFAULT_POP_SIZE,
        generations: int = DEFAULT_GENERATIONS,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        seed: int = None,
    ):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.seed = seed

        # Normalize weights
        total = w_delay + w_rel + w_res
        if total > 0:
            self.w_delay /= total
            self.w_rel /= total
            self.w_res /= total

    def weighted_cost(self, path):
        """Calculate weighted QoS cost for a path."""
        if not path or len(path) < 2:
            return float("inf")

        # Delay: link delays + processing delays
        delay = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.graph.get_edge_data(u, v)
            if not data:
                return float("inf")
            delay += data.get("link_delay", data.get("delay", 0))

        # Add node processing delays (intermediate nodes)
        for n in path[1:-1]:
            delay += self.graph.get_node_delay(n)

        # Reliability: -log(product of reliabilities)
        reliability = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.graph.get_edge_data(u, v)
            rel = data.get("link_rel", data.get("reliability", 0.99))
            reliability += -math.log(max(rel, 1e-12))

        for n in path:
            node_rel = self.graph.graph.nodes[n].get("node_rel", 0.99) if hasattr(self.graph, 'graph') else 0.99
            reliability += -math.log(max(node_rel, 1e-12))

        # Resource: inverse bandwidth
        resource = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.graph.get_edge_data(u, v)
            bw = data.get("bandwidth", 1000)
            resource += 1000.0 / max(bw, 1e-6)

        return self.w_delay * delay + self.w_rel * reliability + self.w_res * resource

    def random_path(self, max_steps=60):
        """Generate a random path using random walk."""
        if self.seed is not None:
            random.seed(self.seed)

        path = [self.source]
        current = self.source

        for _ in range(max_steps):
            neighbors = list(self.graph.get_neighbors(current).keys())
            # Avoid revisiting nodes
            nbrs = [n for n in neighbors if n not in path]

            if not nbrs:
                return None

            if self.destination in nbrs:
                return path + [self.destination]

            current = random.choice(nbrs)
            path.append(current)

        return None

    def solve(self):
        """Execute the Genetic Algorithm."""
        if self.seed is not None:
            random.seed(self.seed)

        # Initialize population
        population = []
        max_attempts = self.pop_size * 20
        attempts = 0

        while len(population) < self.pop_size and attempts < max_attempts:
            attempts += 1
            p = self.random_path()
            if p:
                population.append(p)

        if len(population) < max(3, self.pop_size // 20):
            return None, float("inf")

        best_path = None
        best_cost = float("inf")

        # Evolution loop
        for gen in range(self.generations):
            # Evaluate population
            scored = []
            for p in population:
                cost = self.weighted_cost(p)
                if cost < float("inf"):
                    scored.append((p, cost))

            if not scored:
                break

            # Sort by cost
            scored.sort(key=lambda x: x[1])

            # Update best
            if scored[0][1] < best_cost:
                best_cost = scored[0][1]
                best_path = scored[0][0]

            # Elitism: keep top 10%
            elite = [p for p, _ in scored[:max(1, self.pop_size // 10)]]
            population = elite[:]

            # Fill population with elite copies (simplified reproduction)
            while len(population) < self.pop_size:
                population.append(random.choice(elite))

        return best_path, best_cost
