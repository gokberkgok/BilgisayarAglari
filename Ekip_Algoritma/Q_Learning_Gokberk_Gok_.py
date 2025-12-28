import random
import math
import networkx as nx
import pandas as pd

# =====================================================================
#                         PARAMETRELER
# =====================================================================

# Ağ Parametreleri
NODE_COUNT = 250
EDGE_PROBABILITY = 0.4

# Link Özellikleri
BANDWIDTH_MIN = 100
BANDWIDTH_MAX = 1000
LINK_DELAY_MIN = 3
LINK_DELAY_MAX = 15
LINK_RELIABILITY_MIN = 0.95
LINK_RELIABILITY_MAX = 0.999

# Q-Learning Parametreleri
ALPHA = 0.1          # Öğrenme oranı
GAMMA = 0.99         # İndirim faktörü
EPSILON = 0.2        # Keşif oranı
EPISODES = 300       # Episode sayısı
MAX_STEPS = 250      # Episode başına maksimum adım

# Ağırlıklar (Toplam = 1)
W_DELAY = 0.4
W_RELIABILITY = 0.4
W_RESOURCE = 0.2

# Kaynak ve Hedef Düğümler
SOURCE = 2
DESTINATION = 8


# =====================================================================
#                         COST FUNCTIONS
# =====================================================================

def path_total_delay(G, path):
    """Yol üzerindeki toplam gecikme (ms)"""
    delay = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        delay += G.edges[u, v]['link_delay']
    for k in path[1:-1]:
        delay += G.nodes[k]['proc_delay']
    return delay


def path_reliability_cost(G, path):
    """Yol üzerindeki güvenilirlik maliyeti"""
    cost = 0
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        cost += -math.log(G.edges[u, v]['link_rel'])
    for k in path:
        cost += -math.log(G.nodes[k]['node_rel'])
    return cost


def path_resource_cost(G, path):
    """Yol üzerindeki kaynak maliyeti (bant genişliği)"""
    cost = 0
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        bw = G.edges[u, v]['bandwidth']
        cost += (1000 / bw)
    return cost


def total_cost(G, path, w_delay, w_rel, w_res):
    """Ağırlıklı toplam maliyet"""
    return (w_delay * path_total_delay(G, path) +
            w_rel   * path_reliability_cost(G, path) +
            w_res   * path_resource_cost(G, path))


# =====================================================================
#                         Q-LEARNING AGENT
# =====================================================================

class QLearning:
    def __init__(self, G, alpha, gamma, epsilon):
        self.G = G
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {n: {nb: 0.0 for nb in G.neighbors(n)} for n in G.nodes()}

    def choose(self, s):
        """Epsilon-greedy aksiyon seçimi"""
        neighbors = list(self.Q[s].keys())
        if not neighbors:
            return None
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        max_q = max(self.Q[s].values())
        best = [a for a, q in self.Q[s].items() if q == max_q]
        return random.choice(best)

    def update(self, s, a, r, s_next):
        """Q-değerini güncelle"""
        max_next = 0
        if s_next is not None and len(self.Q.get(s_next, {})) > 0:
            max_next = max(self.Q[s_next].values())
        td = r + self.gamma * max_next
        self.Q[s][a] += self.alpha * (td - self.Q[s][a])


# =====================================================================
#                         GRAF OLUŞTURMA
# =====================================================================
"""
#def generate_graph(N, p):
  
    import os

    print(f"{'='*60}")
    print(f"Node Sayısı (N): {N}")
    print(f"Bağlantı Olasılığı (p): {p}")

    # ===============================================================
    #                      NODE DATA LOAD
    # ===============================================================
    try:
        cwd = os.getcwd()
        fpath_nodes = os.path.join(cwd, "BSM307_317_Guz2025_TermProject_NodeData.csv")
        try:
            df_nodes = pd.read_csv(fpath_nodes, sep=";", decimal=",")
        except:
            df_nodes = pd.read_csv(fpath_nodes, sep=",", decimal=".")

        df_nodes.columns = ["node_id", "processing_delay", "reliability"]

        if len(df_nodes) < N:
            print(f"⚠️ UYARI: CSV'de sadece {len(df_nodes)} düğüm var, N={N} istendi.")
            N = len(df_nodes)

    except Exception as e:
        print(f"❌ HATA: NodeData CSV okunamadı: {str(e)}")
        exit(1)

    # ===============================================================
    #                      EDGE DATA LOAD
    # ===============================================================
    try:
        fpath_edges = os.path.join(cwd, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
        df_edges = pd.read_csv(fpath_edges)

        df_edges.columns = ["src", "dst", "capacity_mbps", "delay_ms", "r_link"]

    except Exception as e:
        print(f"❌ HATA: EdgeData CSV okunamadı: {str(e)}")
        exit(1)

    # ===============================================================
    #                       GRAPH BUILD
    # ===============================================================
    G = nx.Graph()

    # Node Ekle
    for i in range(N):
        G.add_node(i)

    # EdgeData CSV’den gelen kenarlar ekleniyor
    for _, row in df_edges.iterrows():
        u = int(row["src"])
        v = int(row["dst"])

        if u < N and v < N:
            G.add_edge(u, v)
            G.edges[u, v]["bandwidth"] = float(row["capacity_mbps"])
            G.edges[u, v]["link_delay"] = float(row["delay_ms"])
            G.edges[u, v]["link_rel"] = float(row["r_link"])

    # Eğer CSV kenar yeterli değilse Erdos–Renyi ile tamamla
    target_edges = int(p * (N * (N - 1) / 2))
    while len(G.edges()) < target_edges:
        u = random.randint(0, N - 1)
        v = random.randint(0, N - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            G.edges[u, v]['bandwidth'] = random.uniform(BANDWIDTH_MIN, BANDWIDTH_MAX)
            G.edges[u, v]['link_delay'] = random.uniform(LINK_DELAY_MIN, LINK_DELAY_MAX)
            G.edges[u, v]['link_rel'] = random.uniform(LINK_RELIABILITY_MIN, LINK_RELIABILITY_MAX)

    # Bağlı değilse bağla
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            a = random.choice(list(comps[i]))
            b = random.choice(list(comps[i + 1]))
            G.add_edge(a, b)
        print("⚠️ Graf bağlı değildi, bağlantı eklendi.")

    # Node attributes
    for n in G.nodes():
        if n < len(df_nodes):
            G.nodes[n]['proc_delay'] = float(df_nodes.loc[n, "processing_delay"])
            G.nodes[n]['node_rel'] = float(df_nodes.loc[n, "reliability"])
        else:
            G.nodes[n]['proc_delay'] = 1.0
            G.nodes[n]['node_rel'] = 0.95

    print(f"✅ Graf başarıyla oluşturuldu ({len(G.nodes)} düğüm, {len(G.edges)} kenar)")
    print(f"{'='*60}")

    return G
"""
def generate_graph(N, p):
    """Graf oluştur - sadece CSV edge kullanılır, ekstra bağlantı yok"""
    import os

    print(f"{'='*60}")
    print(f"Node Sayısı (N): {N}")
    print(f"CSV Edge grafı yükleniyor...")

    # ===============================================================
    #                      NODE DATA LOAD
    # ===============================================================
    try:
        cwd = os.getcwd()
        fpath_nodes = os.path.join(cwd, "BSM307_317_Guz2025_TermProject_NodeData.csv")
        try:
            df_nodes = pd.read_csv(fpath_nodes, sep=";", decimal=",")
        except:
            df_nodes = pd.read_csv(fpath_nodes, sep=",", decimal=".")

        df_nodes.columns = ["node_id", "processing_delay", "reliability"]

        if len(df_nodes) < N:
            print(f"⚠️ UYARI: Node CSV'de {len(df_nodes)} düğüm var, N={N} istendi → N düşürüldü.")
            N = len(df_nodes)

    except Exception as e:
        print(f"❌ HATA: NodeData CSV okunamadı: {str(e)}")
        exit(1)

    # ===============================================================
    #                      EDGE DATA LOAD
    # ===============================================================
    try:
        fpath_edges = os.path.join(cwd, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
        df_edges = pd.read_csv(fpath_edges)

        df_edges.columns = ["src", "dst", "capacity_mbps", "delay_ms", "r_link"]

    except Exception as e:
        print(f"❌ HATA: EdgeData CSV okunamadı: {str(e)}")
        exit(1)

    # ===============================================================
    #                       GRAPH BUILD
    # ===============================================================
    G = nx.Graph()

    # Node ekle
    for i in range(N):
        G.add_node(i)

    # EdgeData CSV’den kenar ekle (SADECE CSV!)
    missing_edge_count = 0

    for _, row in df_edges.iterrows():
        u = int(row["src"])
        v = int(row["dst"])

        # Geçersiz düğüm varsa atla
        if u >= N or v >= N:
            missing_edge_count += 1
            continue

        G.add_edge(u, v)
        G.edges[u, v]["bandwidth"] = float(row["capacity_mbps"])
        G.edges[u, v]["link_delay"] = float(row["delay_ms"])
        G.edges[u, v]["link_rel"] = float(row["r_link"])

    # Node attributes yükle
    for n in G.nodes():
        G.nodes[n]['proc_delay'] = float(df_nodes.loc[n, "processing_delay"])
        G.nodes[n]['node_rel'] = float(df_nodes.loc[n, "reliability"])

    # Raporlama
    print(f"📌 CSV Edge sayısı: {len(df_edges)}")
    print(f"📌 Graf edge sayısı: {len(G.edges)}")
    print(f"📌 Graf node sayısı: {len(G.nodes)}")

    if missing_edge_count > 0:
        print(f"⚠️ UYARI: {missing_edge_count} edge, CSV node sayısı dışında olduğu için eklenmedi.")

    if not nx.is_connected(G):
        print(f"⚠️ UYARI: Graf BAĞLI değil!")
        print(f"   (İstersen bağlanma algoritması ekleyebilirim.)")

    print(f"✅ CSV graf yüklemesi tamamlandı.")
    print(f"{'='*60}")

    return G


# =====================================================================
#                         Q-LEARNING EĞİTİMİ
# =====================================================================

def train_q_learning(G, source, destination, alpha, gamma, epsilon, episodes, max_steps, w_delay, w_rel, w_res):
    """Q-Learning algoritması ile en iyi yolu bul"""
    
    #print(f"{'='*60}")
    #print(f"🎓 Q-LEARNING EĞİTİMİ BAŞLIYOR...")
    #print(f"{'='*60}")
    print(f"Kaynak (S): {source}")
    print(f"Hedef (D): {destination}")
    print(f"Alpha: {alpha}")
    print(f"Gamma: {gamma}")
    print(f"Epsilon: {epsilon}")
    print(f"Episodes: {episodes}")
    print(f"Max Steps/Episode: {max_steps}")
    print(f"\nAğırlıklar:")
    print(f"  w_delay: {w_delay}")
    print(f"  w_reliability: {w_rel}")
    print(f"  w_resource: {w_res}")
    print(f"{'='*60}\n")

    # Normalize weights
    total_w = w_delay + w_rel + w_res
    if total_w > 0:
        w_delay /= total_w
        w_rel /= total_w
        w_res /= total_w

    agent = QLearning(G, alpha, gamma, epsilon)

    best_path = None
    best_cost = float("inf")

    # Eğitim
    for ep in range(episodes):
        s = source
        path = [s]

        for step in range(max_steps):
            a = agent.choose(s)
            if a is None:
                break

            path.append(a)

            if a == destination:
                cost = total_cost(G, path, w_delay, w_rel, w_res)
                reward = 10000 / cost
                agent.update(s, a, reward, None)

                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)

                break

            agent.update(s, a, -1, a)
            s = a

        # İlerleme raporu
        if (ep + 1) % 100 == 0:
            print(f"📊 Episode {ep + 1}/{episodes} tamamlandı...")

    print(f"✅ Eğitim tamamlandı!\n")
    
    return best_path, best_cost


# =====================================================================
#                         SONUÇLARI YAZDIRMA
# =====================================================================

def print_results(G, path, cost):
    """Bulunan yolun sonuçlarını yazdır"""
    
    if path is None:
        print(f"{'='*60}")
        print("❌ YOL BULUNAMADI!")
        print(f"{'='*60}\n")
        return

    print(f"\n{'='*60}")
    print(f"🎯 SONUÇLAR")
    print(f"{'='*60}")
    print(f"🎯 En iyi yol: {' → '.join(map(str, path))}")
    print(f"📏 Yol uzunluğu: {len(path)} düğüm")
    print(f"💰 Toplam Maliyet: {cost:.4f}")

    # Metrikler
    delay = path_total_delay(G, path)
    rel = path_reliability_cost(G, path)
    res = path_resource_cost(G, path)

    print(f"\n📊 METRİKLER:")
    print(f"  ⏱️  Gecikme (Delay): {delay:.3f} ms")
    print(f"  🔒 Güvenilirlik Maliyeti: {rel:.4f}")
    print(f"  📊 Kaynak Maliyeti: {res:.4f}")
    
    # Yol üzerindeki düğümlerin detayları
    print(f"\n{'─'*60}")
    print("📍 YOL ÜZERİNDEKİ DÜĞÜM DETAYLARI:")
    print(f"{'─'*60}")
    print(f"{'Node ID':<10} {'Delay (ms)':<15} {'Reliability':<15}")
    print(f"{'─'*60}")
    for node in path:
        proc_delay = G.nodes[node]['proc_delay']
        node_rel = G.nodes[node]['node_rel']
        print(f"{node:<10} {proc_delay:<15.3f} {node_rel:<15.6f}")
    print(f"{'='*60}\n")


# =====================================================================
#                         MAIN PROGRAM
# =====================================================================

def main():
    """Ana program"""
    # Graf oluştur
    G = generate_graph(NODE_COUNT, EDGE_PROBABILITY)
    
    # Kaynak ve hedef kontrolü
    if SOURCE >= len(G.nodes) or DESTINATION >= len(G.nodes):
        print(f"❌ HATA: Kaynak ({SOURCE}) veya Hedef ({DESTINATION}) geçersiz!")
        print(f"Graf düğüm sayısı: {len(G.nodes)}")
        exit(1)
    
    if SOURCE == DESTINATION:
        print(f"❌ HATA: Kaynak ve hedef aynı olamaz!")
        exit(1)
    
    if not nx.has_path(G, SOURCE, DESTINATION):
        print(f"❌ HATA: {SOURCE} ile {DESTINATION} arasında yol yok!")
        exit(1)
    
    # Q-Learning eğitimi
    best_path, best_cost = train_q_learning(
        G, SOURCE, DESTINATION,
        ALPHA, GAMMA, EPSILON,
        EPISODES, MAX_STEPS,
        W_DELAY, W_RELIABILITY, W_RESOURCE
    )
    
    # Sonuçları yazdır
    print_results(G, best_path, best_cost)


if __name__ == "__main__":
    main()