# =================================================
# QoS Tabanlı Yol Bulma – Genetik Algoritma
# Ana Mod + Test Modu (Demand CSV, 20 Run)
# =================================================

import pandas as pd
import networkx as nx
import os, math, random

# =================================================
# DOSYA YOLLARI
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================
# GÜVENLİ DÖNÜŞÜMLER
# =================================================
def safe_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except:
        return default

def safe_int(x, default=0):
    try:
        return int(float(str(x).replace(",", ".")))
    except:
        return default

# =================================================
# GRAF YÜKLEME
# =================================================
def load_graph(node_csv, edge_csv):
    nd = pd.read_csv(node_csv)
    ed = pd.read_csv(edge_csv)

    G = nx.Graph()

    for _, r in nd.iterrows():
        G.add_node(
            safe_int(r["node_id"]),
            processing_delay=safe_float(r["s_ms"]),
            reliability=safe_float(r["r_node"])
        )

    for _, r in ed.iterrows():
        G.add_edge(
            safe_int(r["src"]),
            safe_int(r["dst"]),
            bandwidth=safe_float(r["capacity_mbps"]),
            delay=safe_float(r["delay_ms"]),
            reliability=safe_float(r["r_link"])
        )

    return G

# =================================================
# DEMAND YÜKLEME (TEST MODU)
# =================================================
def load_demands(csv_file):
    df = pd.read_csv(csv_file)
    demands = []

    for _, r in df.iterrows():
        demands.append({
            "source": safe_int(r["src"]),
            "target": safe_int(r["dst"]),
            "bandwidth": safe_float(r["demand_mbps"])
        })

    return demands

# =================================================
# YOL KONTROLLERİ
# =================================================
def is_valid_path(G, path):
    if not path or len(path) < 2:
        return False
    for u, v in zip(path, path[1:]):
        if not G.has_edge(u, v):
            return False
    return True

def check_bandwidth(G, path, bw):
    if not is_valid_path(G, path):
        return False
    return min(G[u][v]["bandwidth"] for u, v in zip(path, path[1:])) >= bw

# =================================================
# QoS COST
# =================================================
def weighted_cost(G, path, w1, w2, w3):
    delay = sum(G[u][v]["delay"] for u, v in zip(path, path[1:]))
    delay += sum(G.nodes[n]["processing_delay"] for n in path[1:-1])

    reliability = 0.0
    for u, v in zip(path, path[1:]):
        reliability += -math.log(max(G[u][v]["reliability"], 1e-12))
    for n in path:
        reliability += -math.log(max(G.nodes[n]["reliability"], 1e-12))

    resource = sum(1000.0 / G[u][v]["bandwidth"] for u, v in zip(path, path[1:]))

    return w1 * delay + w2 * reliability + w3 * resource

# =================================================
# GENETİK ALGORİTMA (TEK ÇALIŞMA)
# =================================================
def genetic_algorithm(G, source, target, bw, w1, w2, w3,
                      pop_size=60, generations=120, mutation_rate=0.2):

    # ağırlık normalize
    s = w1 + w2 + w3
    w1, w2, w3 = w1/s, w2/s, w3/s

    def random_path(max_steps=60):
        path = [source]
        current = source

        for _ in range(max_steps):
            nbrs = [n for n in G.neighbors(current) if n not in path]
            if not nbrs:
                return None
            if target in nbrs:
                return path + [target]
            current = random.choice(nbrs)
            path.append(current)

        return None

    population = []
    while len(population) < pop_size:
        p = random_path()
        if p and check_bandwidth(G, p, bw):
            population.append(p)

    best_path = None
    best_cost = float("inf")

    for gen in range(generations):
        scored = []
        for p in population:
            if check_bandwidth(G, p, bw):
                scored.append((p, weighted_cost(G, p, w1, w2, w3)))

        if not scored:
            break

        scored.sort(key=lambda x: x[1])

        if scored[0][1] < best_cost:
            best_cost = scored[0][1]
            best_path = scored[0][0]

        elite = [p for p, _ in scored[:max(1, pop_size // 10)]]
        population = elite[:]

        while len(population) < pop_size:
            population.append(random.choice(elite))

    return best_path, best_cost

# =================================================
# MAIN
# =================================================
if __name__ == "__main__":

    print("📡 QoS Tabanlı Yol Bulma – Genetik Algoritma")

    G = load_graph(NODE_FILE, EDGE_FILE)
    print(f"Graf: {len(G.nodes)} düğüm, {len(G.edges)} bağlantı")

    # -----------------------------
    # KULLANICI SEÇİMLİ TEK ÇALIŞMA
    # -----------------------------
    print("\n🎯 KULLANICI SEÇİMİ (ARAYÜZ MODU)")

    source = int(input("Kaynak düğüm (source): "))
    target = int(input("Hedef düğüm (target): "))
    bw     = float(input("Bandwidth (Mbps): "))

    w1, w2, w3 = 0.4, 0.3, 0.3

    path, cost = genetic_algorithm(G, source, target, bw, w1, w2, w3)

    if path:
        print("\n✅ EN İYİ YOL:")
        print(" → ".join(map(str, path)))
        print(f"Toplam Cost: {cost:.2f}")
    else:
        print("❌ Uygun yol bulunamadı")

    # -----------------------------
    # TEST MODU – DEMAND CSV (20 RUN)
    # -----------------------------
    print("\n🧪 TEST MODU – DEMAND CSV (20 Run)")

    demands = load_demands(DEMAND_FILE)

    for i, d in enumerate(demands, start=1):
        print("\n" + "-" * 50)
        print(f"Senaryo {i}: S={d['source']} D={d['target']} B={d['bandwidth']}")

        best_path = None
        best_cost = float("inf")

        for _ in range(20):
            p, c = genetic_algorithm(
                G, d["source"], d["target"], d["bandwidth"],
                w1, w2, w3
            )
            if p and c < best_cost:
                best_cost = c
                best_path = p

        if best_path:
            print("EN İYİ YOL :", " → ".join(map(str, best_path)))
            print(f"EN İYİ COST: {best_cost:.2f}")
        else:
            print("❌ Yol bulunamadı")

    print("\n✅ Program tamamlandı.")
