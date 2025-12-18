# =================================================
# QoS Tabanlı Yol Bulma – PSO (Particle Swarm Optimization)
# Kullanıcı Modu + Demand Test Modu
# =================================================

import networkx as nx
import random
import math
import csv
import os

# =================================================
# DOSYA YOLLARI
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================
# AĞIRLIKLAR
# =================================================
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34
MAX_BANDWIDTH = 1000.0

# =================================================
# GRAF OLUŞTURMA
# =================================================
def create_graph_from_csv():
    G = nx.Graph()

    with open(NODE_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            G.add_node(
                int(r["node_id"]),
                processing_delay=float(r["s_ms"]),
                reliability=float(r["r_node"])
            )

    with open(EDGE_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            u = int(r["src"])
            v = int(r["dst"])
            G.add_edge(
                u, v,
                bandwidth=float(r["capacity_mbps"]),
                delay=float(r["delay_ms"]),
                reliability=float(r["r_link"])
            )

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G

# =================================================
# COST FONKSİYONU (TAM & GÜVENLİ)
# =================================================
def total_cost(G, path, D, min_bw):
    if not path or path[0] not in G or path[-1] != D:
        return float("inf")

    delay = 0.0
    rel_cost = 0.0
    res_cost = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        if not G.has_edge(u, v):
            return float("inf")

        e = G[u][v]
        if e["bandwidth"] < min_bw:
            return float("inf")

        delay += e["delay"]
        rel_cost += -math.log(max(e["reliability"], 1e-12))
        res_cost += MAX_BANDWIDTH / max(e["bandwidth"], 1e-6)

    for n in path[1:-1]:
        delay += G.nodes[n]["processing_delay"]
        rel_cost += -math.log(max(G.nodes[n]["reliability"], 1e-12))

    return (
        W_DELAY * delay +
        W_RELIABILITY * rel_cost +
        W_RESOURCE * res_cost
    )

# =================================================
# PSO SINIFLARI
# =================================================
class Particle:
    def __init__(self, path, cost):
        self.position = list(path)
        self.cost = cost
        self.pbest = list(path)
        self.pbest_cost = cost


class PSO:
    def __init__(self, G, S, D, min_bw,
                 num_particles=30, iterations=100):
        self.G = G
        self.S = S
        self.D = D
        self.min_bw = min_bw
        self.num_particles = num_particles
        self.iterations = iterations

        self.particles = []
        self.gbest = None
        self.gbest_cost = float("inf")

    # -----------------------------
    def shortest_valid_path(self):
        try:
            path = nx.shortest_path(self.G, self.S, self.D)
            if total_cost(self.G, path, self.D, self.min_bw) < float("inf"):
                return path
        except:
            return None
        return None

    # -----------------------------
    def initialize(self):
        self.particles.clear()
        self.gbest = None
        self.gbest_cost = float("inf")

        base = self.shortest_valid_path()
        if not base:
            return

        for _ in range(self.num_particles):
            p = Particle(base, total_cost(self.G, base, self.D, self.min_bw))
            self.particles.append(p)

        self.gbest = list(base)
        self.gbest_cost = p.cost

    # -----------------------------
    def run(self):
        self.initialize()

        if not self.gbest:
            return None, float("inf")

        for _ in range(self.iterations):
            for p in self.particles:

                if len(self.gbest) < 4:
                    continue

                cut = random.randint(1, len(self.gbest) - 2)
                candidate = self.gbest[:cut] + p.position[cut:]

                # 🔒 ZORUNLU PSO KONTROLLERİ
                if candidate[0] != self.S or candidate[-1] != self.D:
                    continue

                cost = total_cost(self.G, candidate, self.D, self.min_bw)
                if cost == float("inf"):
                    continue

                if cost < p.pbest_cost:
                    p.pbest = list(candidate)
                    p.pbest_cost = cost

                if cost < self.gbest_cost:
                    self.gbest = list(candidate)
                    self.gbest_cost = cost

        return list(self.gbest), float(self.gbest_cost)

# =================================================
# DEMAND YÜKLE
# =================================================
def load_demands():
    demands = []
    with open(DEMAND_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            demands.append((
                int(r["src"]),
                int(r["dst"]),
                float(r["demand_mbps"])
            ))
    return demands

# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    print("📡 QoS Tabanlı Yol Bulma – PSO\n")

    G = create_graph_from_csv()
    print(f"Graf: {G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar\n")

    # ------------------------------
    # KULLANICI MODU
    # ------------------------------
    print("🎯 KULLANICI MODU")
    S = int(input("Source: "))
    D = int(input("Destination: "))
    B = float(input("Bandwidth (Mbps): "))

    pso = PSO(G, S, D, B)
    path, cost = pso.run()

    if path:
        print("\n✅ EN İYİ YOL:")
        print(" → ".join(map(str, path)))
        print(f"Cost: {cost:.4f}")
    else:
        print("❌ Yol bulunamadı")

    # ------------------------------
    # TEST MODU – DEMAND CSV
    # ------------------------------
    print("\n🧪 TEST MODU – DEMAND DATA\n")

    demands = load_demands()
    for i, (s, d, bw) in enumerate(demands, 1):
        pso = PSO(G, s, d, bw)
        path, cost = pso.run()

        if path:
            print(f"#{i:02d} {s}->{d} | Cost={cost:.4f}")
        else:
            print(f"#{i:02d} {s}->{d} | ❌ Yol bulunamadı")

    print("\n✅ Tüm PSO testleri tamamlandı.")
