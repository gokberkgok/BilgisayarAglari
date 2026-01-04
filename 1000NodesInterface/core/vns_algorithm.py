"""
VNS (Variable Neighborhood Search) Solver for QoS-based Multi-Objective Routing.
Adapter for integration with NetworkGraph format.
"""

import random
import math
import copy
from collections import deque
from .network_model import NetworkGraph

# Default VNS Parameters
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_K_MAX = 3


class VNS_Solver:
    """
    VNS Solver adapted for NetworkGraph format.
    Uses variable neighborhood search metaheuristic.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        source: int,
        destination: int,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.33,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        k_max: int = DEFAULT_K_MAX,
        seed: int = None,
    ):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        self.max_iterations = max_iterations
        self.k_max = k_max
        self.seed = seed

        # Normalize weights
        total = w_delay + w_rel + w_res
        if total > 0:
            self.w_delay /= total
            self.w_rel /= total
            self.w_res /= total

    def calculate_cost(self, path):
        """Calculate total QoS cost for a path."""
        if not path or len(path) < 2:
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

    def initial_path(self):
        """Find initial path using BFS with randomization."""
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

    def shake(self, path, k):
        """Shaking: perturb current solution to escape local minima."""
        if len(path) < 4:
            return path

        new_path = copy.deepcopy(path)

        # Select random segment
        i = random.randint(1, len(new_path) - 3)
        j = min(len(new_path) - 1, i + k + 1)

        start = new_path[i - 1]
        end = new_path[j]

        # Find alternative sub-path using DFS
        sub = []
        visited = set(new_path[:i])

        def dfs(cur):
            if cur == end:
                return True
            if len(sub) > 6:  # Depth limit
                return False

            neighbors = list(self.graph.get_neighbors(cur).keys())
            random.shuffle(neighbors)

            for n in neighbors:
                if n not in visited:
                    visited.add(n)
                    sub.append(n)
                    if dfs(n):
                        return True
                    sub.pop()
                    visited.remove(n)
            return False

        if dfs(start):
            return new_path[:i] + sub + new_path[j:]

        return path

    def local_search(self, path):
        """Local search: find shortcuts in current path."""
        best = path
        best_cost = self.calculate_cost(best)

        improved = True
        while improved:
            improved = False
            # Try all possible shortcuts (2-opt style)
            for i in range(len(best) - 2):
                for j in range(i + 2, len(best)):
                    u, v = best[i], best[j]
                    # Check if direct edge exists
                    if v in self.graph.get_neighbors(u):
                        candidate = best[: i + 1] + best[j:]
                        cost = self.calculate_cost(candidate)
                        if cost < best_cost:
                            best = candidate
                            best_cost = cost
                            improved = True
                            break
                if improved:
                    break
        return best

    def solve(self):
        """Execute VNS algorithm."""
        if self.seed is not None:
            random.seed(self.seed)

        path = self.initial_path()
        if not path:
            return None, float("inf")

        cost = self.calculate_cost(path)
        best_path, best_cost = path, cost

        for _ in range(self.max_iterations):
            k = 1
            while k <= self.k_max:
                # Shaking
                shaken = self.shake(best_path, k)
                # Local search
                improved = self.local_search(shaken)
                # Evaluation
                c = self.calculate_cost(improved)

                if c < best_cost:  # Improvement found
                    best_path, best_cost = improved, c
                    k = 1  # Reset to nearest neighborhood
                else:
                    k += 1  # Try farther neighborhood

        return best_path, best_cost
