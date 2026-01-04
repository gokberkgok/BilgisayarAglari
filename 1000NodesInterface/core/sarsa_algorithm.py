"""
SARSA Solver for QoS-based Multi-Objective Routing.
Adapter for integration with NetworkGraph format.
"""

import random
import math
from collections import defaultdict
from .network_model import NetworkGraph

# Default SARSA Parameters
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.95
DEFAULT_EPSILON = 0.3
DEFAULT_EPISODES = 2000


class SARSA_Solver:
    """
    SARSA (On-Policy) Solver adapted for NetworkGraph format.
    Uses reinforcement learning to find optimal QoS paths.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        source: int,
        destination: int,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.33,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        epsilon: float = DEFAULT_EPSILON,
        episodes: int = DEFAULT_EPISODES,
        seed: int = None,
    ):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.seed = seed

        # Normalize weights
        total = w_delay + w_rel + w_res
        if total > 0:
            self.w_delay /= total
            self.w_rel /= total
            self.w_res /= total

        # Q-Table
        self.Q = defaultdict(float)

    def get_valid_neighbors(self, node):
        """Get neighbors of a node."""
        return list(self.graph.get_neighbors(node).keys())

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        neighbors = self.get_valid_neighbors(state)
        if not neighbors:
            return None

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(neighbors)

        # Exploitation
        return max(neighbors, key=lambda a: self.Q[(state, a)])

    def calculate_path_cost(self, path):
        """Calculate total QoS cost for a path."""
        if not path or len(path) < 2:
            return float("inf")

        delay = 0.0
        reliability = 0.0
        resource = 0.0

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

        for n in path[1:-1]:
            delay += self.graph.get_node_delay(n)

        return self.w_delay * delay + self.w_rel * reliability + self.w_res * resource

    def solve(self):
        """Execute SARSA algorithm."""
        if self.seed is not None:
            random.seed(self.seed)

        best_path = None
        best_cost = float("inf")

        for episode in range(self.episodes):
            state = self.source
            path = [state]

            valid_neighbors = self.get_valid_neighbors(state)
            if not valid_neighbors:
                continue

            # Choose initial action (SARSA characteristic)
            action = self.choose_action(state)

            while state != self.destination:
                next_state = action
                path.append(next_state)

                # Reached destination
                if next_state == self.destination:
                    cost = self.calculate_path_cost(path)
                    reward = 1000 - cost

                    # Terminal update
                    self.Q[(state, action)] += self.alpha * (
                        reward - self.Q[(state, action)]
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_path = list(path)
                    break

                # Get next neighbors
                next_neighbors = self.get_valid_neighbors(next_state)
                if not next_neighbors:
                    # Dead end
                    reward = -500
                    self.Q[(state, action)] += self.alpha * (
                        reward - self.Q[(state, action)]
                    )
                    break

                # Choose next action (On-policy)
                next_action = self.choose_action(next_state)

                # Calculate step reward (edge cost)
                edge_data = self.graph.get_edge_data(state, next_state)
                if edge_data:
                    d_val = edge_data.get("link_delay", edge_data.get("delay", 0))
                    r_val = -math.log(
                        max(edge_data.get("link_rel", edge_data.get("reliability", 0.99)), 1e-12)
                    )
                    b_val = 1000.0 / max(edge_data.get("bandwidth", 1), 1e-6)
                    edge_cost = self.w_delay * d_val + self.w_rel * r_val + self.w_res * b_val
                    reward = -edge_cost
                else:
                    reward = -100

                # SARSA Update: Q(s,a) = Q(s,a) + alpha * (R + gamma * Q(s',a') - Q(s,a))
                current_q = self.Q[(state, action)]
                next_q = self.Q[(next_state, next_action)]
                self.Q[(state, action)] = current_q + self.alpha * (
                    reward + self.gamma * next_q - current_q
                )

                # Move to next state-action pair
                state = next_state
                action = next_action

        return best_path, best_cost
