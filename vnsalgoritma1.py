import csv
import math
import random
import copy
import os
import networkx as nx
from collections import deque
#test deneme
# =================================================
# AYARLAR
# =================================================
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34
MAX_BANDWIDTH_MBPS = 1000.0

TEST_RUNS = 30

# =================================================
# DOSYA YOLLARI
# =================================================
NODE_FILE = r"BSM307_317_Guz2025_TermProject_NodeData.csv"
EDGE_FILE = r"BSM307_317_Guz2025_TermProject_EdgeData.csv"
DEMAND_FILE = r"BSM307_317_Guz2025_TermProject_DemandData.csv"

# =================================================
# NETWORK GRAPH
# =================================================
class NetworkGraph:
    def __init__(self):
        self.G = nx.Graph()

    def load_data(self, node_file, edge_file):
        with open(node_file, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                self.G.add_node(
                    int(r["node_id"]),
                    s_ms=float(r["s_ms"]),
                    r_node=float(r["r_node"])
                )

        with open(edge_file, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                self.G.add_edge(
                    int(r["src"]),
                    int(r["dst"]),
                    bw=float(r["capacity_mbps"]),
                    delay=float(r["delay_ms"]),
                    r_link=float(r["r_link"])
                )

    def calculate_metrics(self, path):
        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        dest = path[-1]

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.G[u][v]
            node = self.G.nodes[v]

            total_delay += edge["delay"]
            reliability_cost += -math.log(edge["r_link"])
            resource_cost += MAX_BANDWIDTH_MBPS / edge["bw"]

            if v != dest:
                total_delay += node["s_ms"]
                reliability_cost += -math.log(node["r_node"])

        noise = random.uniform(-0.01, 0.01)

        cost = (
            W_DELAY * total_delay +
            W_RELIABILITY * reliability_cost +
            W_RESOURCE * resource_cost
        ) * (1 + noise)

        return cost, {
            "Cost": cost,
            "Delay": total_delay,
            "Reliability": math.exp(-reliability_cost),
            "Resource": resource_cost
        }

# =================================================
# VNS OPTIMIZER
# =================================================
class VNS:
    def __init__(self, graph):
        self.graph = graph
        self.G = graph.G

    def initial_path(self, src, dst):
        paths = []

        for _ in range(5):
            queue = deque([(src, [src])])
            visited = {src}

            while queue:
                cur, path = queue.popleft()
                if cur == dst:
                    paths.append(path)
                    break

                nbrs = list(self.G.neighbors(cur))
                random.shuffle(nbrs)

                for n in nbrs:
                    if n not in visited:
                        visited.add(n)
                        queue.append((n, path + [n]))

        return random.choice(paths) if paths else None

    def shake(self, path, k):
        if len(path) < 4:
            return path

        i = random.randint(1, len(path) - 3)
        j = min(len(path) - 1, i + k + random.randint(1, 3))

        start = path[i - 1]
        end = path[j]

        sub = []
        visited = set(path[:i])

        def dfs(cur):
            if cur == end:
                return True
            if len(sub) > random.randint(4, 10):
                return False

            nbrs = list(self.G.neighbors(cur))
            random.shuffle(nbrs)

            for n in nbrs:
                if n not in visited:
                    visited.add(n)
                    sub.append(n)
                    if dfs(n):
                        return True
                    sub.pop()
                    visited.remove(n)
            return False

        if dfs(start):
            return path[:i] + sub + path[j:]
        return path

    def local_search(self, path):
        best = path
        best_cost, _ = self.graph.calculate_metrics(best)

        for _ in range(2):  # local search baskısını azalttık
            for i in range(len(best) - 2):
                for j in range(i + 2, len(best)):
                    u, v = best[i], best[j]
                    if self.G.has_edge(u, v):
                        cand = best[:i+1] + best[j:]
                        cost, _ = self.graph.calculate_metrics(cand)
                        if cost < best_cost:
                            best = cand
                            best_cost = cost
        return best

    def run(self, src, dst):
        path = self.initial_path(src, dst)
        if not path:
            return None, None

        best_path = path
        best_cost, _ = self.graph.calculate_metrics(path)

        MAX_VNS_ITER = random.randint(15, 30)
        K_MAX = random.randint(3, 6)

        for _ in range(MAX_VNS_ITER):
            k = 1
            while k <= K_MAX:
                shaken = self.shake(best_path, k)
                improved = self.local_search(shaken)
                cost, _ = self.graph.calculate_metrics(improved)

                if cost < best_cost:
                    best_path = improved
                    best_cost = cost
                    k = 1
                else:
                    k += 1

        return best_path, self.graph.calculate_metrics(best_path)

# =================================================
# MAIN
# =================================================
def main():
    print("\n📡 QoS Aware Stochastic VNS \n")

    graph = NetworkGraph()
    graph.load_data(NODE_FILE, EDGE_FILE)
    vns = VNS(graph)

    demands = []
    with open(DEMAND_FILE, "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            demands.append((int(r["src"]), int(r["dst"])))

    for i, (s, d) in enumerate(demands, start=1):
        print("\n" + "-" * 60)
        print(f"Senaryo {i}: {s} → {d}")

        best_cost = float("inf")
        best_path = None
        best_metrics = None

        for _ in range(TEST_RUNS):
            path, result = vns.run(s, d)
            if path:
                cost = result[1]["Cost"]
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_metrics = result[1]

        print("EN İYİ YOL :", " → ".join(map(str, best_path)))
        print(f"Cost       : {best_metrics['Cost']:.4f}")
        print(f"Delay      : {best_metrics['Delay']:.2f} ms")
        print(f"Reliability: {best_metrics['Reliability']:.4f}")
        print(f"Resource   : {best_metrics['Resource']:.2f}")

    print("\n✅ Program tamamlandı.")

if __name__ == "__main__":
    main()
