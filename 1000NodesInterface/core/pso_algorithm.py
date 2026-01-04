"""
PSO (Particle Swarm Optimization) Solver for QoS-based Multi-Objective Routing.
Adapter for integration with NetworkGraph format.
"""

import random
import math
import copy
from collections import deque
from .network_model import NetworkGraph

# Default PSO Parameters
DEFAULT_NUM_PARTICLES = 30
DEFAULT_ITERATIONS = 100


class Particle:
    """Represents a single solution candidate (path)."""

    def __init__(self, path, cost):
        self.position = list(path)  # Current path
        self.cost = cost  # Current cost
        self.pbest = list(path)  # Personal best path
        self.pbest_cost = cost  # Personal best cost


class PSO_Solver:
    """
    PSO Solver adapted for NetworkGraph format.
    Uses particle swarm optimization for discrete path finding.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        source: int,
        destination: int,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.33,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        iterations: int = DEFAULT_ITERATIONS,
        seed: int = None,
    ):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        self.num_particles = num_particles
        self.iterations = iterations
        self.seed = seed

        # Normalize weights
        total = w_delay + w_rel + w_res
        if total > 0:
            self.w_delay /= total
            self.w_rel /= total
            self.w_res /= total

        self.particles = []
        self.gbest = None
        self.gbest_cost = float("inf")

    def calculate_cost(self, path):
        """Calculate total QoS cost for a path."""
        if not path or path[0] != self.source or path[-1] != self.destination:
            return float("inf")

        delay = 0.0
        reliability = 0.0
        resource = 0.0

        # Edge costs
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.graph.get_edge_data(u, v)
            if not data:
                return float("inf")

            delay += data.get("link_delay", data.get("delay", 0))
            rel = data.get("link_rel", data.get("reliability", 0.99))
            reliability += -math.log(max(rel, 1e-12))
            bw = data.get("bandwidth", 1000)
            resource += 1000.0 / max(bw, 1e-6)

        # Node costs
        for n in path[1:-1]:
            delay += self.graph.get_node_delay(n)

        return self.w_delay * delay + self.w_rel * reliability + self.w_res * resource

    def shortest_valid_path(self):
        """Find initial path using BFS."""
        queue = deque([(self.source, [self.source])])
        visited = {self.source}

        while queue:
            current, path = queue.popleft()
            if current == self.destination:
                return path

            neighbors = list(self.graph.get_neighbors(current).keys())
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def initialize(self):
        """Initialize particle swarm."""
        if self.seed is not None:
            random.seed(self.seed)

        self.particles.clear()
        self.gbest = None
        self.gbest_cost = float("inf")

        # Find base path
        base = self.shortest_valid_path()
        if not base:
            return False

        # Create particles
        for _ in range(self.num_particles):
            cost = self.calculate_cost(base)
            p = Particle(base, cost)
            self.particles.append(p)

        # Set initial gbest
        self.gbest = list(base)
        self.gbest_cost = self.calculate_cost(base)
        return True

    def shake(self, path):
        """Shake operation: modify path by combining with gbest."""
        if len(self.gbest) < 4:
            return path

        # Random crossover point
        cut = random.randint(1, len(self.gbest) - 2)

        # Combine gbest start with current path end
        candidate = self.gbest[:cut] + path[cut:]

        # Validate
        if not candidate or candidate[0] != self.source or candidate[-1] != self.destination:
            return path

        return candidate

    def solve(self):
        """Execute PSO algorithm."""
        if not self.initialize():
            return None, float("inf")

        for iteration in range(self.iterations):
            for p in self.particles:
                # Shake: create candidate solution
                candidate = self.shake(p.position)

                # Evaluate
                cost = self.calculate_cost(candidate)
                if cost == float("inf"):
                    continue

                # Update personal best
                if cost < p.pbest_cost:
                    p.pbest = list(candidate)
                    p.pbest_cost = cost

                # Update global best
                if cost < self.gbest_cost:
                    self.gbest = list(candidate)
                    self.gbest_cost = cost

        return list(self.gbest), float(self.gbest_cost)
