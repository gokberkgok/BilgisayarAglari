import csv
import math
import random
import time
import copy
import os
from collections import deque

# =================================================
# AYARLAR
# =================================================
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34
MAX_BANDWIDTH_MBPS = 1000.0

MAX_VNS_ITER = 20
K_MAX = 3
TEST_RUNS = 30

# =================================================
# DOSYA YOLLARI (TAŞINABİLİR)
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================
# NETWORK GRAPH
# =================================================
class NetworkGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def load_data(self, node_file, edge_file):
        with open(node_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [n.strip() for n in reader.fieldnames]
            for r in reader:
                nid = int(r["node_id"])
                self.nodes[nid] = {
                    "s_ms": float(r["s_ms"]),
                    "r_node": float(r["r_node"])
                }
                self.edges.setdefault(nid, {})

        with open(edge_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [n.strip() for n in reader.fieldnames]
            for r in reader:
                u = int(r["src"])
                v = int(r["dst"])
                props = {
                    "bw": float(r["capacity_mbps"]),
                    "delay": float(r["delay_ms"]),
                    "r_link": float(r["r_link"])
                }
                self.edges.setdefault(u, {})[v] = props
                self.edges.setdefault(v, {})[u] = props  # çift yönlü

    def calculate_metrics(self, path):
        if not path or len(path) < 2:
            return float("inf"), None

        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        dest = path[-1]

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.edges[u][v]
            node = self.nodes[v]

            total_delay += edge["delay"]
            reliability_cost += -math.log(edge["r_link"])
            resource_cost += MAX_BANDWIDTH_MBPS / edge["bw"]

            if v != dest:
                total_delay += node["s_ms"]
                reliability_cost += -math.log(node["r_node"])

        cost = (
            W_DELAY * total_delay +
            W_RELIABILITY * reliability_cost +
            W_RESOURCE * resource_cost
        )

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

    def initial_path(self, src, dst):
        queue = deque([(src, [src])])
        visited = {src}

        while queue:
            cur, path = queue.popleft()
            if cur == dst:
                return path

            nbrs = list(self.graph.edges[cur].keys())
            random.shuffle(nbrs)

            for n in nbrs:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, path + [n]))
        return None

    def shake(self, path, k):
        if len(path) < 4:
            return path

        new_path = copy.deepcopy(path)
        i = random.randint(1, len(new_path) - 3)
        j = min(len(new_path) - 1, i + k + 1)

        start = new_path[i - 1]
        end = new_path[j]

        sub = []
        visited = set(new_path[:i])

        def dfs(cur):
            if cur == end:
                return True
            if len(sub) > 6:
                return False
            nbrs = list(self.graph.edges[cur].keys())
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
            return new_path[:i] + sub + new_path[j:]
        return path

    def local_search(self, path):
        best = path
        best_cost, _ = self.graph.calculate_metrics(best)

        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 2):
                for j in range(i + 2, len(best)):
                    u, v = best[i], best[j]
                    if v in self.graph.edges[u]:
                        cand = best[:i+1] + best[j:]
                        cost, _ = self.graph.calculate_metrics(cand)
                        if cost < best_cost:
                            best = cand
                            best_cost = cost
                            improved = True
                            break
                if improved:
                    break
        return best

    def run(self, src, dst):
        path = self.initial_path(src, dst)
        if not path:
            return None, None

        cost, _ = self.graph.calculate_metrics(path)
        best_path, best_cost = path, cost

        for _ in range(MAX_VNS_ITER):
            k = 1
            while k <= K_MAX:
                shaken = self.shake(best_path, k)
                improved = self.local_search(shaken)
                c, _ = self.graph.calculate_metrics(improved)
                if c < best_cost:
                    best_path, best_cost = improved, c
                    k = 1
                else:
                    k += 1

        return best_path, self.graph.calculate_metrics(best_path)

# =================================================
# MAIN – SENARYO BAŞINA 20 RUN
# =================================================
def main():
    print("📡 BSM307 – QoS Odaklı VNS (Senaryo Başına 20 Run)\n")

    graph = NetworkGraph()
    graph.load_data(NODE_FILE, EDGE_FILE)
    vns = VNS(graph)

    demands = []
    with open(DEMAND_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for r in reader:
            demands.append((int(r["src"]), int(r["dst"])))

    for i, (s, d) in enumerate(demands, start=1):
        print("\n" + "-" * 55)
        print(f"Senaryo {i}: S={s} D={d}")

        best_path = None
        best_cost = float("inf")
        best_metrics = None

        for _ in range(TEST_RUNS):
            path, result = vns.run(s, d)
            if path:
                cost = result[1]["Cost"]
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_metrics = result[1]

        if best_path:
            print("EN İYİ YOL :", " → ".join(map(str, best_path)))
            print(f"Cost       : {best_metrics['Cost']:.4f}")
            print(f"Delay      : {best_metrics['Delay']:.2f} ms")
            print(f"Reliability: {best_metrics['Reliability']:.4f}")
            print(f"Resource   : {best_metrics['Resource']:.2f}")
        else:
            print("❌ Yol bulunamadı")

    print("\n✅ Program tamamlandı.")

if __name__ == "__main__":
    main()
