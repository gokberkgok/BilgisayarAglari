# =================================================
# QoS Tabanlı Yol Bulma – SARSA
# Kullanıcı Modu + Demand Test Modu
# =================================================

import networkx as nx
import random
import math
import time
import csv
import os
from collections import defaultdict

# =================================================
# DOSYA YOLLARI (TAŞINABİLİR)
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
# METRİK HESAPLAMA
# =================================================
def compute_cost(G, path):
    delay = 0
    rel_cost = 0
    res_cost = 0

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        e = G[u][v]
        # GUI uyumlu key isimleri
        delay += e.get("link_delay", e.get("delay", 0))
        rel_cost += -math.log(max(e.get("link_rel", e.get("reliability", 0.99)), 1e-12))
        res_cost += 1000 / max(e["bandwidth"], 1e-6)

    for n in path[1:-1]:
        # GUI uyumlu key isimleri
        delay += G.nodes[n].get("proc_delay", G.nodes[n].get("processing_delay", 0))
        rel_cost += -math.log(max(G.nodes[n].get("node_rel", G.nodes[n].get("reliability", 0.99)), 1e-12))

    return (
        W_DELAY * delay +
        W_RELIABILITY * rel_cost +
        W_RESOURCE * res_cost
    )

# =================================================
# SARSA
# =================================================
def sarsa_route(G, S, D, min_bw, episodes=2000):
    Q = defaultdict(float)
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.3

    best_path = None
    best_cost = float("inf")

    def neighbors(u):
        return [
            v for v in G.neighbors(u)
            if G[u][v]["bandwidth"] >= min_bw
        ]

    for _ in range(episodes):
        state = S
        path = [state]

        if not neighbors(state):
            continue

        action = random.choice(neighbors(state))

        while state != D:
            next_state = action
            path.append(next_state)

            if next_state == D:
                # ✅ YENİ ÖDÜL TASARIMI: Hedefe ulaşma bonusu + yol maliyeti cezası
                cost = compute_cost(G, path)
                reward = 1000 - cost  # Büyük bonus + maliyet cezası
                Q[(state, action)] += alpha * (reward - Q[(state, action)])

                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
                break

            next_neighbors = neighbors(next_state)
            if not next_neighbors:
                # Çıkmaz sokak - büyük ceza
                reward = -500
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break

            if random.random() < epsilon:
                next_action = random.choice(next_neighbors)
            else:
                next_action = max(next_neighbors, key=lambda a: Q[(next_state, a)])

            # ✅ YENİ ÖDÜL TASARIMI: Her adım için edge maliyetini cezalandır
            edge = G[state][next_state]
            edge_cost = (
                W_DELAY * edge.get("link_delay", edge.get("delay", 0)) +
                W_RELIABILITY * (-math.log(max(edge.get("link_rel", edge.get("reliability", 0.99)), 1e-12))) +
                W_RESOURCE * (1000 / max(edge["bandwidth"], 1e-6))
            )
            reward = -edge_cost  # Negatif maliyet = ceza
            
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )

            state, action = next_state, next_action

    return best_path, best_cost

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
    print("QoS Tabanlı Yol Bulma – SARSA\n")

    G = create_graph_from_csv()
    print(f"Graf yüklendi: {G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar\n")

    # ------------------------------
    # KULLANICI MODU
    # ------------------------------
    print("KULLANICI MODU (TEK ÇALIŞMA)")
    S = int(input("Source: "))
    D = int(input("Destination: "))
    B = float(input("Bandwidth (Mbps): "))

    path, cost = sarsa_route(G, S, D, B)

    if path:
        print("\nEN İYİ YOL:")
        print(" → ".join(map(str, path)))
        print(f"Cost: {cost:.4f}")
    else:
        print("Yol bulunamadı")

    # ------------------------------
    # TEST MODU – DEMAND CSV
    # ------------------------------
    print("\nTEST MODU – DEMAND DATA\n")

    demands = load_demands()

    for i, (s, d, bw) in enumerate(demands, 1):
        path, cost = sarsa_route(G, s, d, bw)
        if path:
            print(f"#{i:02d} {s}->{d} | Cost={cost:.4f}")
        else:
            print(f"#{i:02d} {s}->{d} | Yol bulunamadı")

    print("\nTüm testler tamamlandı.")
