import random
import math
from core.qos_metrics import QoSMetrics


class ACOResult:
    def __init__(self, path, qos: QoSMetrics):
        self.path = path
        self.qos = qos


class ACOSolver:
    def __init__(self, graph, source, destination):
        self.G = graph
        self.source = source
        self.destination = destination

    def solve(self, iterations=20, ants=15):
        best = None
        best_cost = float("inf")

        for _ in range(iterations):
            for _ in range(ants):
                path = self.walk()
                if not path:
                    continue

                qos = self.calculate_qos(path)
                cost = qos.delay + qos.reliability + qos.resource

                if cost < best_cost:
                    best_cost = cost
                    best = ACOResult(path, qos)

        return best

    def walk(self):
        curr = self.source
        visited = {curr}
        path = [curr]

        while curr != self.destination:
            neighbors = [
                n for n in self.G.neighbors(curr) if n not in visited
            ]
            if not neighbors:
                return None
            curr = random.choice(list(neighbors))
            visited.add(curr)
            path.append(curr)

        return path

    def calculate_qos(self, path):
        delay = 0
        rel = 0
        res = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = self.G.edge(u, v)

            delay += e["delay"] + self.G.node_delay(v)
            rel += -math.log(e["reliability"])
            res += 1 / e["bandwidth"]

        return QoSMetrics(delay, rel, res)
