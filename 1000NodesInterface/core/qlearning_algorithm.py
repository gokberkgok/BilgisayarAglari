"""
Q-Learning Solver for QoS-based Multi-Objective Routing.
Adapter for integration with NetworkGraph format.
"""

import random
import math
from collections import defaultdict
from .network_model import NetworkGraph

# Default Q-Learning Parameters
DEFAULT_ALPHA = 0.1
DEFAULT_GAMMA = 0.90
DEFAULT_EPSILON = 0.9
DEFAULT_EPISODES = 300
DEFAULT_MAX_STEPS = 250


class QLearning_Solver:
    """
    Q-Learning Solver adapted for NetworkGraph format.
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
        max_steps: int = DEFAULT_MAX_STEPS,
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
        self.max_steps = max_steps
        self.seed = seed

        # Normalize weights
        total = w_delay + w_rel + w_res
        if total > 0:
            self.w_delay /= total
            self.w_rel /= total
            self.w_res /= total

        # Q-Table: {node: {neighbor: Q-value}}
        self.Q = defaultdict(lambda: defaultdict(float))

    def initialize_q_table(self):
        """Initialize Q-table with all possible state-action pairs."""
        for node in self.graph.adj_list:
            neighbors = list(self.graph.get_neighbors(node).keys())
            for neighbor in neighbors:
                self.Q[node][neighbor] = 0.0

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        neighbors = list(self.graph.get_neighbors(state).keys())
        if not neighbors:
            return None

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(neighbors)

        # Exploitation: choose best known action
        max_q = max(self.Q[state].values()) if self.Q[state] else 0
        best = [a for a, q in self.Q[state].items() if q == max_q and a in neighbors]
        return random.choice(best) if best else random.choice(neighbors)

    def update_q(self, state, action, reward, next_state):
        """Update Q-value using Bellman equation."""
        max_next = 0
        if next_state is not None and next_state in self.Q:
            next_neighbors = list(self.graph.get_neighbors(next_state).keys())
            if next_neighbors:
                max_next = max(self.Q[next_state].values()) if self.Q[next_state] else 0

        # Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
        td_target = reward + self.gamma * max_next
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])

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

        # Add node delays
        for n in path[1:-1]:
            delay += self.graph.get_node_delay(n)

        return self.w_delay * delay + self.w_rel * reliability + self.w_res * resource

    def solve(self):
        """Execute Q-Learning algorithm."""
        if self.seed is not None:
            random.seed(self.seed)

        self.initialize_q_table()
        best_path = None
        best_cost = float("inf")

        # Training loop
        for episode in range(self.episodes):
            state = self.source
            path = [state]

            for step in range(self.max_steps):
                action = self.choose_action(state)
                if action is None:
                    break

                path.append(action)

                # Check if reached destination
                if action == self.destination:
                    cost = self.calculate_path_cost(path)
                    if cost > 0:
                        reward = 10000 / cost
                    else:
                        reward = 10000

                    self.update_q(state, action, reward, None)

                    if cost < best_cost:
                        best_cost = cost
                        best_path = list(path)
                    break

                # Intermediate step: penalty for longer paths
                self.update_q(state, action, -1, action)
                state = action

        return best_path, best_cost
