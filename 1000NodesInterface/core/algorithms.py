import random
import numpy as np
import math
from .network_model import NetworkGraph

# ACO Algorithm Constants
DEFAULT_NUM_ANTS = 20
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_ALPHA = 1.0  # Pheromone importance
DEFAULT_BETA = 2.0   # Heuristic importance
DEFAULT_RHO = 0.1    # Evaporation rate
DEFAULT_Q0 = 0.9     # Exploitation vs Exploration threshold

# Normalization Constants
NORM_DELAY_DIVISOR = 25000.0
NORM_REL_DIVISOR = 50.0
NORM_BW_DIVISOR = 10.0

# Candidate List Strategy
TOP_HEURISTIC_NEIGHBORS = 20
RANDOM_NEIGHBORS = 5
CANDIDATE_LIST_THRESHOLD = 25

class ACO_Solver:
    """
    Ant Colony Optimization Solver for Multi-Objective QoS Routing.
    Optimizes for: Delay, Reliability, Bandwidth.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        source: int,
        destination: int,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.33,
        num_ants: int = DEFAULT_NUM_ANTS,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        rho: float = DEFAULT_RHO,
        q0: float = DEFAULT_Q0,
    ):
        self.graph = graph
        self.source = source
        self.destination = destination

        # Weights for the objective function
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res

        # ACO Parameters
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.rho = rho      # Evaporation rate
        self.q0 = q0        # Exploration vs Exploitation probability

        # Pheromones: Dictionary {(u,v): pheromone_value}
        self.pheromones = {}
        self.initialize_pheromones()

    def initialize_pheromones(self):
        """Initializes pheromones on all edges to a small constant."""
        initial_pheromone = 1.0
        for u in self.graph.adj_list:
            for v in self.graph.adj_list[u]:
                self.pheromones[(u, v)] = initial_pheromone

    def calculate_heuristic(self, u, v):
        r"""
        Calculates heuristic information \eta_{ij}.
        Heuristic = 1 / Cost_{ij}

        For multi-objective routing, a local scalarized cost is used:
            C = w_delay * delay + w_rel * (-log(reliability)) + w_res * (1 / bandwidth)
        """
        data = self.graph.get_edge_data(u, v)
        if not data:
            
            return 0.0001

        # Rough normalization for heuristic guidance
        delay_cost = data["link_delay"] / 20.0
        rel_cost = -math.log(data["reliability"]) / 0.05
        bw_cost = (1000.0 / data["bandwidth"]) / 10.0

        cost = (
            self.w_delay * delay_cost
            + self.w_rel * rel_cost
            + self.w_res * bw_cost
        )

        return 1.0 / (cost + 1e-9)

    def get_candidate_list(self, current_node):
        """
        Returns a Candidate List of neighbors to reduce search space.
        Strategy: Top 20 heuristic neighbors + 5 random neighbors.
        """
        neighbors = list(self.graph.get_neighbors(current_node).keys())
        if not neighbors:
            return []

        if len(neighbors) <= CANDIDATE_LIST_THRESHOLD:
            return neighbors

        neighbors_sorted = sorted(
            neighbors,
            key=lambda v: self.calculate_heuristic(current_node, v),
            reverse=True,
        )

        top_n = neighbors_sorted[:TOP_HEURISTIC_NEIGHBORS]
        remaining = neighbors_sorted[TOP_HEURISTIC_NEIGHBORS:]
        random_n = random.sample(remaining, min(RANDOM_NEIGHBORS, len(remaining)))

        return top_n + random_n

    def select_next_node(self, current_node, visited):
        """Selects the next node using the ACO transition rule."""
        candidates = [
            n for n in self.get_candidate_list(current_node) if n not in visited
        ]
        if not candidates:
            return None

        # Exploitation
        if random.random() < self.q0:
            return max(
                candidates,
                key=lambda v: (
                    self.pheromones.get((current_node, v), 1.0) ** self.alpha
                )
                * (self.calculate_heuristic(current_node, v) ** self.beta),
            )

        # Exploration (roulette wheel)
        probabilities = []
        denominator = 0.0

        for v in candidates:
            tau = self.pheromones.get((current_node, v), 1.0) ** self.alpha
            eta = self.calculate_heuristic(current_node, v) ** self.beta
            prob = tau * eta
            probabilities.append(prob)
            denominator += prob

        if denominator == 0:
            return random.choice(candidates)

        probabilities = [p / denominator for p in probabilities]
        return random.choices(candidates, weights=probabilities, k=1)[0]

    def solve(self):
        """Executes the Ant Colony Optimization algorithm."""
        global_best_path = None
        global_best_cost = float("inf")

        for _ in range(self.max_iterations):
            paths = []
            costs = []

            for _ in range(self.num_ants):
                path = [self.source]
                visited = {self.source}
                curr = self.source

                while curr != self.destination:
                    next_node = self.select_next_node(curr, visited)
                    if next_node is None:
                        break
                    path.append(next_node)
                    visited.add(next_node)
                    curr = next_node

                if curr == self.destination:
                    cost = self.evaluate_path(path)
                    paths.append(path)
                    costs.append(cost)

                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_path = path

            self.update_pheromones(paths, costs)

            if global_best_path:
                self.deposit_pheromone(
                    global_best_path, global_best_cost, weight=2.0
                )

        return global_best_path, global_best_cost

    def evaluate_path(self, path):
        """Calculates the scalarized cost of a path."""
        total_delay = 0.0
        total_rel_log = 0.0
        total_bw_inv = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.graph.get_edge_data(u, v)

            total_delay += data["link_delay"] + self.graph.get_node_delay(v)
            total_rel_log += -math.log(data["reliability"])
            total_bw_inv += 1.0 / data["bandwidth"]

        norm_delay = total_delay / NORM_DELAY_DIVISOR
        norm_rel = total_rel_log / NORM_REL_DIVISOR
        norm_bw = total_bw_inv / NORM_BW_DIVISOR

        return (
            self.w_delay * norm_delay
            + self.w_rel * norm_rel
            + self.w_res * norm_bw
        )

    def update_pheromones(self, paths, costs):
        """Applies pheromone evaporation and deposition."""
        for key in self.pheromones:
            self.pheromones[key] *= (1.0 - self.rho)

        for path, cost in zip(paths, costs):
            self.deposit_pheromone(path, cost)

    def deposit_pheromone(self, path, cost, weight=1.0):
        deposit = (1.0 / (cost + 1e-9)) * weight
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) in self.pheromones:
                self.pheromones[(u, v)] += deposit


try:
    import pulp

    PULP_AVAILABLE = True
except ImportError:
    pulp = None
    PULP_AVAILABLE = False


class ILP_Solver:
    """
    Integer Linear Programming baseline solver using PuLP.
    """

    def __init__(
        self,
        graph: NetworkGraph,
        source: int,
        destination: int,
        w_delay: float = 0.33,
        w_rel: float = 0.33,
        w_res: float = 0.33,
        time_limit: int = 30,
    ):
        self.graph = graph
        self.source = source
        self.destination = destination
        self.w_delay = w_delay
        self.w_rel = w_rel
        self.w_res = w_res
        self.time_limit = time_limit

    def solve(self):
        if not PULP_AVAILABLE:
            return None, float("inf")

        prob = pulp.LpProblem("QoS_Routing_Problem", pulp.LpMinimize)

        x = {}
        for u in self.graph.adj_list:
            for v in self.graph.adj_list[u]:
                x[(u, v)] = pulp.LpVariable(f"x_{u}_{v}", cat="Binary")

        objective_terms = []
        for u, v in x:
            data = self.graph.get_edge_data(u, v)

            norm_delay = (
                data["link_delay"] + self.graph.get_node_delay(v)
            ) / 25000.0
            norm_rel = -math.log(data["reliability"]) / 50.0
            norm_bw = (1.0 / data["bandwidth"]) / 10.0

            cost = (
                self.w_delay * norm_delay
                + self.w_rel * norm_rel
                + self.w_res * norm_bw
            )
            objective_terms.append(cost * x[(u, v)])

        prob += pulp.lpSum(objective_terms)

        nodes = list(self.graph.adj_list.keys())
        for i in nodes:
            inflow = pulp.lpSum(
                [x[(j, i)] for j in nodes if i in self.graph.adj_list.get(j, {})]
            )
            outflow = pulp.lpSum([x[(i, j)] for j in self.graph.adj_list[i]])

            if i == self.source:
                prob += outflow - inflow == 1
            elif i == self.destination:
                prob += outflow - inflow == -1
            else:
                prob += outflow - inflow == 0

        solver = pulp.PULP_CBC_CMD(timeLimit=self.time_limit, msg=False)
        status = prob.solve(solver)

        if status != pulp.LpStatusOptimal:
            return None, float("inf")

        path = [self.source]
        curr = self.source
        visited = {self.source}

        while curr != self.destination:
            for v in self.graph.adj_list[curr]:
                if pulp.value(x[(curr, v)]) == 1.0 and v not in visited:
                    path.append(v)
                    visited.add(v)
                    curr = v
                    break
            else:
                break

        return path, pulp.value(prob.objective)


class Pareto_Analyzer:
    """
    Finds the Pareto front by running ACO with random weight combinations.
    """

    def __init__(self, graph: NetworkGraph, source: int, destination: int):
        self.graph = graph
        self.source = source
        self.destination = destination

    def run_analysis(self, num_simulations=10):
        """
        Runs Pareto analysis with multiple weight combinations.
        
        Args:
            num_simulations: Number of ACO runs with random weights (default: 10)
                            Reduced from 50 for performance (25x faster)
        """
        all_solutions = []

        for _ in range(num_simulations):
            w = np.random.dirichlet(np.ones(3))
            aco = ACO_Solver(
                self.graph,
                self.source,
                self.destination,
                w_delay=w[0],
                w_rel=w[1],
                w_res=w[2],
                num_ants=10,      # Reduced from 20 for performance
                max_iterations=8,  # Reduced from 20 for performance
            )

            path, _ = aco.solve()
            if path:
                metrics = self.calculate_metrics(path)
                all_solutions.append({"path": path, "metrics": metrics})

        return self.filter_dominated(all_solutions)

    def calculate_metrics(self, path):
        c_delay = 0.0
        c_rel = 0.0
        c_res = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            data = self.graph.get_edge_data(u, v)

            c_delay += data["link_delay"] + self.graph.get_node_delay(v)
            c_rel += -math.log(data["reliability"])
            c_res += 1.0 / data["bandwidth"]

        return (c_delay, c_rel, c_res)

    def filter_dominated(self, solutions):
        pareto = []

        for i, a in enumerate(solutions):
            dominated = False
            for j, b in enumerate(solutions):
                if i == j:
                    continue
                if (
                    b["metrics"][0] <= a["metrics"][0]
                    and b["metrics"][1] <= a["metrics"][1]
                    and b["metrics"][2] <= a["metrics"][2]
                    and b["metrics"] != a["metrics"]
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(a)

        return pareto
