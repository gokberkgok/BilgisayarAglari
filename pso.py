####
import random
from copy import deepcopy
import networkx as nx

from .metrics import total_cost

class Particle:
   
    def __init__(self, path, cost):
        self.position = list(path)
        self.cost = float(cost)
        self.pbest_position = list(path)
        self.pbest_cost = float(cost)


class PSO:
   

    def __init__(self, G, S, D, w_delay=0.33, w_rel=0.33, w_res=0.34,
                 num_particles=30, iterations=100, seed=None):
        self.G = G
        self.S = S
        self.D = D
        self.w_delay = float(w_delay)
        self.w_rel = float(w_rel)
        self.w_res = float(w_res)
        self.num_particles = max(3, int(num_particles))
        self.iterations = max(1, int(iterations))
        self.particles = []
        self.gbest = None
        self.gbest_cost = float('inf')
        if seed is not None:
            random.seed(seed)

   
    def _safe_shortest(self, a, b):
        
        try:
            return nx.shortest_path(self.G, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def _ensure_valid_path(self, path):
        
        if not path or len(path) < 2:
            return self._safe_shortest(self.S, self.D)

        
        new = list(path)

        
        if new[0] != self.S:
            new.insert(0, self.S)
        if new[-1] != self.D:
            new.append(self.D)

        repaired = [new[0]]
        i = 0
        while i < len(new) - 1:
            u = repaired[-1]
            v = new[i+1]
            if self.G.has_edge(u, v):
                repaired.append(v)
                i += 1
                continue
            
            sp = self._safe_shortest(u, v)
            if sp is None:
                
                sp2 = self._safe_shortest(u, self.D)
                if sp2 is None:
                    
                    return self._safe_shortest(self.S, self.D)
                
                for node in sp2[1:]:
                    repaired.append(node)
                break
            else:
                
                for node in sp[1:]:
                    repaired.append(node)
                
                i += 1

        seen = set()
        pruned = []
        for n in repaired:
            if n in seen:
                
                while pruned and pruned[-1] != n:
                    removed = pruned.pop()
                    seen.remove(removed)
                
                continue
            pruned.append(n)
            seen.add(n)

        
        if not pruned or pruned[0] != self.S or pruned[-1] != self.D:
            sp_final = self._safe_shortest(self.S, self.D)
            return sp_final

        
        for i in range(len(pruned) - 1):
            if not self.G.has_edge(pruned[i], pruned[i+1]):
                return self._safe_shortest(self.S, self.D)

        return pruned

   
    def _random_path(self):
        
        base = self._safe_shortest(self.S, self.D)
        if base is None:
            return None

        path = list(base)

        
        for _ in range(random.randint(0, 3)):
            if len(path) > 3 and random.random() < 0.5:
                
                idx = random.randint(1, len(path) - 2)
                candidate = path[:idx] + path[idx+1:]
                repaired = self._ensure_valid_path(candidate)
                if repaired:
                    path = repaired
            elif random.random() < 0.4:
                
                idx = random.randint(0, len(path) - 2)
                u = path[idx]
                neighbors = list(self.G.neighbors(u))
                if neighbors:
                    v = random.choice(neighbors)
                    if v not in path:
                        candidate = path[:idx+1] + [v] + path[idx+1:]
                        repaired = self._ensure_valid_path(candidate)
                        if repaired:
                            path = repaired

        path = self._ensure_valid_path(path)
        return path

    def initialize(self):
        
        self.particles = []
        self.gbest = None
        self.gbest_cost = float('inf')

        attempts = 0
        
        while len(self.particles) < self.num_particles and attempts < max(200, self.num_particles * 10):
            attempts += 1
            p = self._random_path()
            if p is None:
                continue
            c = total_cost(self.G, p, self.w_delay, self.w_rel, self.w_res)
            particle = Particle(p, c)
            self.particles.append(particle)
            if c < self.gbest_cost:
                self.gbest = list(p)
                self.gbest_cost = float(c)

        if not self.particles:
            
            sp = self._safe_shortest(self.S, self.D)
            if sp is None:
                raise RuntimeError("PSO initialization failed: no S->D path exists in the graph.")
            c = total_cost(self.G, sp, self.w_delay, self.w_rel, self.w_res)
            particle = Particle(sp, c)
            self.particles.append(particle)
            self.gbest = list(sp)
            self.gbest_cost = float(c)

   
    def _combine_towards(self, base_path, target_path, prob=0.6):
       
        if not base_path or not target_path:
            return base_path

        base = list(base_path)
        target = list(target_path)

        new = [self.S]

       
        for node in target[1:]:
            if node == new[-1]:
                continue
            last = new[-1]
           
            if self.G.has_edge(last, node) and random.random() < prob:
                new.append(node)
            else:
               
                sp = self._safe_shortest(last, node)
                if sp:
                    
                    for n in sp[1:]:
                        if n in new:
                          
                            break
                        new.append(n)
               
            if new[-1] == self.D:
                break

        
        if new[-1] != self.D:
            sp_tail = self._safe_shortest(new[-1], self.D)
            if sp_tail:
                for n in sp_tail[1:]:
                    new.append(n)

        
        repaired = self._ensure_valid_path(new)
        if repaired:
            return repaired
        
        fallback = self._safe_shortest(self.S, self.D)
        return fallback if fallback else base

    def _mutate(self, path, mutation_rate=0.15):
       
        if not path:
            return self._safe_shortest(self.S, self.D)

        new = list(path)

        
        if len(new) > 4 and random.random() < mutation_rate:
            i = random.randint(1, len(new) - 3)
            j = random.randint(i+1, len(new) - 2)
            new[i], new[j] = new[j], new[i]

        
        if len(new) > 3 and random.random() < mutation_rate:
            idx = random.randint(1, len(new) - 2)
            candidate = new[:idx] + new[idx+1:]
            candidate_repaired = self._ensure_valid_path(candidate)
            if candidate_repaired:
                new = candidate_repaired

        
        if random.random() < mutation_rate:
            idx = random.randint(0, len(new) - 2)
            u = new[idx]
            neigh = list(self.G.neighbors(u))
            if neigh:
                v = random.choice(neigh)
                if v not in new:
                    candidate = new[:idx+1] + [v] + new[idx+1:]
                    candidate_repaired = self._ensure_valid_path(candidate)
                    if candidate_repaired:
                        new = candidate_repaired

        
        final = self._ensure_valid_path(new)
        return final if final else self._safe_shortest(self.S, self.D)

    
    def update_particle(self, particle):
       
        current = list(particle.position)

        
        if random.random() < 0.6:
            candidate = self._combine_towards(current, particle.pbest_position, prob=0.6)
            if candidate:
                repaired = self._ensure_valid_path(candidate)
                if repaired:
                    current = repaired

        
        if self.gbest and random.random() < 0.6:
            candidate = self._combine_towards(current, self.gbest, prob=0.7)
            if candidate:
                repaired = self._ensure_valid_path(candidate)
                if repaired:
                    current = repaired

       
        if random.random() < 0.35:
            candidate = self._mutate(current, mutation_rate=0.25)
            if candidate:
                repaired = self._ensure_valid_path(candidate)
                if repaired:
                    current = repaired

        
        validated = self._ensure_valid_path(current)
        if validated is None:
            validated = self._safe_shortest(self.S, self.D)
            if validated is None:
                
                return

        
        cost = total_cost(self.G, validated, self.w_delay, self.w_rel, self.w_res)
        particle.position = validated
        particle.cost = cost

        
        if cost < particle.pbest_cost:
            particle.pbest_cost = cost
            particle.pbest_position = list(validated)

        
        if cost < self.gbest_cost:
            self.gbest_cost = cost
            self.gbest = list(validated)

    def run(self, verbose=False):
        
        self.initialize()
        if verbose:
            print(f"[PSO] Initialized {len(self.particles)} particles. gbest_cost={self.gbest_cost:.6f}")

        for it in range(self.iterations):
            for p in self.particles:
                self.update_particle(p)

            if verbose and (it % max(1, self.iterations // 10) == 0):
                print(f"[PSO] Iter {it+1}/{self.iterations} gbest_cost={self.gbest_cost:.6f}")

        
        if self.gbest is None:
            final = self._safe_shortest(self.S, self.D)
            if final is None:
                return None, float('inf')
            final_cost = total_cost(self.G, final, self.w_delay, self.w_rel, self.w_res)
            return final, final_cost

        return list(self.gbest), float(self.gbest_cost)

