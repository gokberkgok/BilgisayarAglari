import networkx as nx
import random
import math
import time
import csv
import os
from collections import defaultdict

# =================================================
# 1. DOSYA YOLLARI VE AYARLAR
# =================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

WEIGHTS = (0.33, 0.33, 0.34)  # (w_delay, w_rel, w_res)

# =================================================
# 2. GRAF OLUŞTURMA VE METRİKLER
# =================================================
def create_graph_from_csv():
    G = nx.Graph()
    
    # Düğümleri Yükle
    try:
        with open(NODE_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader) 
            for row in reader:
                if len(row) < 3: continue
                try:
                    nid = int(row[0])
                    pd = float(row[1].replace(',', '.'))
                    rel = float(row[2].replace(',', '.'))
                    G.add_node(nid, processing_delay=pd, reliability=rel)
                except ValueError: continue
    except FileNotFoundError: print(f"HATA: {NODE_FILE} bulunamadı.")

    # Kenarları Yükle
    try:
        with open(EDGE_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader:
                if len(row) < 5: continue
                try:
                    u, v = int(row[0]), int(row[1])
                    bw = float(row[2].replace(',', '.'))
                    d = float(row[3].replace(',', '.'))
                    r = float(row[4].replace(',', '.'))
                    G.add_edge(u, v, bandwidth=bw, delay=d, reliability=r)
                except ValueError: continue
    except FileNotFoundError: print(f"HATA: {EDGE_FILE} bulunamadı.")

    return G

def calculate_total_cost(G, path):
    if not path: return float('inf')
    
    total_delay = 0
    rel_log_sum = 0
    res_cost_sum = 0
    
    w_d, w_r, w_res = WEIGHTS

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = G[u][v]
        d = edge.get("delay", 10)
        r = edge.get("reliability", 0.9)
        bw = edge.get("bandwidth", 100)

        total_delay += d
        if r <= 0: r = 0.0001
        rel_log_sum += -math.log(r)
        if bw <= 0: bw = 0.1
        res_cost_sum += (1000.0 / bw)

    for node in path:
        node_data = G.nodes[node]
        proc = node_data.get("processing_delay", 0)
        nr = node_data.get("reliability", 0.99)
        
        total_delay += proc
        if nr <= 0: nr = 0.0001
        rel_log_sum += -math.log(nr)

    return (w_d * total_delay) + (w_r * rel_log_sum) + (w_res * res_cost_sum)

# =================================================
# 3. ACO ALGORİTMASI
# =================================================

class ACOSolver:
    @staticmethod
    def solve(graph, source, target, min_bw):
        num_ants = 20
        num_iterations = 30
        alpha = 1.0
        beta = 2.0
        evaporation = 0.1
        Q = 100.0
        tau_min = 0.1
        tau_max = 10.0

        pheromones = defaultdict(lambda: 1.0)
        global_best_path = None
        global_best_cost = float('inf')

        for _ in range(num_iterations):
            paths = []
            for _ in range(num_ants):
                path = ACOSolver._ant_walk(graph, source, target, pheromones, alpha, beta, min_bw)
                if path:
                    cost = calculate_total_cost(graph, path)
                    paths.append((path, cost))
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_path = list(path)

            for k in list(pheromones.keys()):
                pheromones[k] *= (1.0 - evaporation)
                if pheromones[k] < tau_min: pheromones[k] = tau_min

            for p, c in paths:
                deposit = Q / c if c > 0 else Q
                for i in range(len(p)-1):
                    u, v = p[i], p[i+1]
                    key = tuple(sorted((u,v)))
                    pheromones[key] = min(tau_max, pheromones[key] + deposit)

            if global_best_path:
                deposit = (Q / global_best_cost) * 2.0
                for i in range(len(global_best_path)-1):
                    u, v = global_best_path[i], global_best_path[i+1]
                    key = tuple(sorted((u,v)))
                    pheromones[key] = min(tau_max, pheromones[key] + deposit)

        return global_best_path, global_best_cost

    @staticmethod
    def _ant_walk(graph, start, end, pheromones, alpha, beta, min_bw):
        current = start
        path = [current]
        visited = set(path)
        w_d, w_r, w_res = WEIGHTS

        while current != end:
            neighbors = [n for n in graph.neighbors(current) 
                         if n not in visited and graph[current][n].get('bandwidth', 0) >= min_bw]
            
            if not neighbors: return None

            probs = []
            denom = 0.0
            
            for n in neighbors:
                edge = graph[current][n]
                key = tuple(sorted((current, n)))
                tau = pheromones[key]
                
                d = edge.get('delay', 1); r = edge.get('reliability', 0.9); bw = edge.get('bandwidth', 100)
                if r<=0: r=0.001
                
                local_cost = (w_d * d) + (w_r * -math.log(r)) + (w_res * (1000/bw))
                eta = 1.0 / local_cost if local_cost > 0 else 1.0
                
                val = (tau ** alpha) * (eta ** beta)
                probs.append(val)
                denom += val
            
            if denom == 0: return None
            
            probs = [p/denom for p in probs]
            
            next_node = random.choices(neighbors, weights=probs, k=1)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
            if len(path) > 250: return None

        return path

# =================================================
# MAIN (ÇALIŞTIRMA KISMI)
# =================================================
if __name__ == "__main__":
    print("BSM307 - ACO Modülü (Terminal)")
    print("===================================")
    
    G = create_graph_from_csv()
    if G.number_of_nodes() == 0:
        print("HATA: Node/Edge dosyaları okunamadı.")
        exit()
        
    print(f"Graf Yüklendi: {G.number_of_nodes()} Düğüm, {G.number_of_edges()} Kenar\n")

    while True:
        print("\nSEÇENEKLER:")
        print("1. Tekli Test (Manuel Giriş)")
        print("2. Toplu Test (DemandData.csv Kullanarak)")
        print("0. Çıkış")
        
        choice = input("Seçiminiz: ")
        
        if choice == '0':
            print("Çıkılıyor...")
            break
            
        if choice == '1':
            try:
                print("\n--- TEKLİ TEST (ACO) ---")
                S = int(input("Kaynak (Source) ID: "))
                D = int(input("Hedef (Dest) ID: "))
                B = float(input("Bant Genişliği (Mbps): "))
                
                print("Karıncalar yola çıkıyor...")
                start_t = time.time()
                
                path, cost = ACOSolver.solve(G, S, D, min_bw=B)
                
                end_t = time.time()

                if path:
                    print(f"\nACO BAŞARILI:")
                    print(f"   Süre: {(end_t-start_t)*1000:.2f} ms")
                    print(f"   Maliyet (Cost): {cost:.4f}")
                    # Burada zaten tam yol yazdırılıyordu:
                    print(f"   En İyi Yol: {' -> '.join(map(str, path))}")
                else:
                    print(f"\nACO Yol Bulamadı (Bant genişliği yetersiz veya izole düğüm)!")
            except ValueError:
                print("Hatalı giriş! Lütfen sayı giriniz.")

        elif choice == '2':
            print("\n--- TOPLU TEST (DEMAND DATA) ---")
            # BAŞLIK GÜNCELLENDİ: En sağa 'En İyi Yol' eklendi
            print(f"{'#':<4} {'S->D':<10} {'Bant':<8} | {'Durum':<10} {'Maliyet':<10} {'Süre (ms)':<10} | {'En İyi Yol'}")
            print("-" * 100)
            
            demands = []
            try:
                with open(DEMAND_FILE, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f, delimiter=';')
                    next(reader)
                    for row in reader:
                        if len(row) >= 3:
                            demands.append((int(row[0]), int(row[1]), float(row[2].replace(',','.'))))
            except FileNotFoundError:
                print("DemandData.csv bulunamadı!")
                continue

            count = 0
            for S, D, B in demands:
                count += 1
                
                start_t = time.time()
                path, cost = ACOSolver.solve(G, S, D, min_bw=B)
                end_t = time.time()
                
                status = "BAŞARILI" if path else "X"
                c_str = f"{cost:.2f}" if path else "-"
                duration = (end_t - start_t) * 1000
                
                # Yolu string haline getir: "8 -> 44 -> 55" gibi
                if path:
                    path_str = " -> ".join(map(str, path))
                else:
                    path_str = "Yol Bulunamadı"
                
                # ÇIKTI GÜNCELLENDİ: path_str eklendi
                print(f"{count:<4} {S}->{D:<7} {int(B):<8} | {status:<10} {c_str:<10} {duration:<10.2f} | {path_str}")
                
    print("\nProgram Sonlandı.")