import random

class ACOSolver:
    @staticmethod
    def solve(network_model, source, target, weights):
        graph = network_model.graph
        w_d, w_r, w_res = weights

        num_ants = 20
        num_iterations = 30
        alpha = 1.0  
        beta = 2.0   
        evaporation_rate = 0.5
        Q = 100.0

        pheromones = {}
        for u, v in graph.edges():
            pheromones[(u, v)] = 1.0
            pheromones[(v, u)] = 1.0

        best_path = None
        best_cost = float('inf')

        for iteration in range(num_iterations):
            paths_in_iteration = []

            for ant in range(num_ants):
                path = ACOSolver._ant_walk(graph, source, target, pheromones, alpha, beta)
                if path:
                    cost, _, _, _ = network_model.get_fitness(path, w_d, w_r, w_res)
                    paths_in_iteration.append((path, cost))
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_path = list(path)

            for key in pheromones:
                pheromones[key] *= (1.0 - evaporation_rate)

            for path, cost in paths_in_iteration:
                deposit = Q / cost if cost > 0 else Q
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (u, v) in pheromones: 
                        pheromones[(u, v)] += deposit
                    else: 
                        pheromones[(v, u)] += deposit

        return best_path

    @staticmethod
    def _ant_walk(graph, start_node, end_node, pheromones, alpha, beta):
        current_node = start_node
        path = [current_node]
        visited = set(path)

        while current_node != end_node:
            neighbors = list(graph.neighbors(current_node))
            valid_neighbors = [n for n in neighbors if n not in visited]

            if not valid_neighbors: 
                return None

            probabilities = []
            denominator = 0.0

            for neighbor in valid_neighbors:
                tau = pheromones.get((current_node, neighbor), 1.0)
                link_delay = graph[current_node][neighbor]['link_delay']
                eta = 1.0 / link_delay if link_delay > 0 else 1.0
                
                prob = (tau ** alpha) * (eta ** beta)
                probabilities.append(prob)
                denominator += prob

            if denominator == 0: 
                return None
            
            probabilities = [p / denominator for p in probabilities]
            
            next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            
            if len(path) > 100: 
                return None

        return path