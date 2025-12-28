# =================================================================================================
# QoS Tabanlı Yol Bulma – PSO (Particle Swarm Optimization) Algoritması
# =================================================================================================
# Bu modül, Parçacık Sürü Optimizasyonu (PSO) yöntemini kullanarak ağ optimizasyonu yapar.
#
# NORMALDE PSO NASIL ÇALIŞIR?
# - Sürekli uzayda (continuous space) parçacıklar hız ve konum vektörleri ile hareket eder.
# - V = w*V + c1*r1*(Pbest - X) + c2*r2*(Gbest - X)
# - X = X + V
#
# BU PROJEDEKİ (DISCRETE) PSO YAKLAŞIMI:
# - Yol bulma problemi süreksiz (discrete) olduğu için standart hız denklemleri kullanılamaz.
# - Bunun yerine "Yol Birleştirme / Mutasyon" mantığı kullanılır.
# - Her parçacık bir "Yol" temsil eder.
# - Parçacıklar, Global En İyi (Gbest) ve Kendi En İyileri (Pbest) ile yollarını parça parça takas ederek
#   daha iyi yollar bulmaya çalışır.
# =================================================================================================

import networkx as nx
import random
import math
import csv
import os

# =================================================================================================
# GLOBAL YAPILANDIRMA VE DOSYA YOLLARI
# =================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================================================================
# AĞIRLIKLAR VE SABİTLER
# =================================================================================================
# QoS Maliyet Ağırlıkları
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34
MAX_BANDWIDTH = 1000.0 # Normalizasyon için referans değer

# =================================================================================================
# GRAF OLUŞTURMA (CSV -> NetworkX)
# =================================================================================================
def create_graph_from_csv():
    """
    CSV dosyalarını okuyarak ağ topolojisini (Graf) oluşturur.
    Parçalı yapıyı önlemek için en büyük bağlı bileşeni (Largest Connected Component) döndürür.
    """
    G = nx.Graph()

    # Düğüm Özellikleri
    with open(NODE_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            G.add_node(
                int(r["node_id"]),
                processing_delay=float(r["s_ms"]),
                reliability=float(r["r_node"])
            )

    # Kenar Özellikleri
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

    # Bağlantısızlık Kontrolü
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G

# =================================================================================================
# COST (FITNESS) FONKSİYONU
# =================================================================================================
def total_cost(G, path, D, min_bw):
    """
    Bir yolun (parçacığın) kalitesini ölçer. Düşük maliyet = İyi Çözüm.
    Geçersiz yollar (kopuk, bant genişliği yetersiz) sonsuz maliyet alır.
    """
    # 1. Temel Geçerlilik Kontrolü
    if not path or path[0] not in G or path[-1] != D:
        return float("inf")

    delay = 0.0
    rel_cost = 0.0
    res_cost = 0.0

    # 2. Kenar (Link) Maliyetleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Yol üzerinde kopukluk var mı?
        if not G.has_edge(u, v):
            return float("inf")

        e = G[u][v]
        # Bant genişliği kısıtı
        if e["bandwidth"] < min_bw:
            return float("inf")

        delay += e["delay"]
        # Güvenilirlik (Logaritmik dönüşüm)
        rel_cost += -math.log(max(e["reliability"], 1e-12))
        # Kaynak (Ters orantılı maliyet)
        res_cost += MAX_BANDWIDTH / max(e["bandwidth"], 1e-6)

    # 3. Düğüm (Node) Maliyetleri
    for n in path[1:-1]:
        delay += G.nodes[n]["processing_delay"]
        rel_cost += -math.log(max(G.nodes[n]["reliability"], 1e-12))

    # Ağırlıklı Toplam
    return (
        W_DELAY * delay +
        W_RELIABILITY * rel_cost +
        W_RESOURCE * res_cost
    )

# =================================================================================================
# PSO SINIFLARI VE ALGORİTMASI
# =================================================================================================

class Particle:
    """Tek bir çözüm adayını (Yol) temsil eder."""
    def __init__(self, path, cost):
        self.position = list(path) # Mevcut Yol
        self.cost = cost           # Mevcut Maliyet
        self.pbest = list(path)    # Kişisel En İyi Yol
        self.pbest_cost = cost     # Kişisel En İyi Maliyet


class PSO:
    """Algoritma Yöneticisi"""
    def __init__(self, G, S, D, min_bw,
                 num_particles=30, iterations=100, seed=None):
        self.G = G
        self.S = S
        self.D = D
        self.min_bw = min_bw
        self.num_particles = num_particles
        self.iterations = iterations
        self.seed = seed

        self.particles = []
        self.gbest = None
        self.gbest_cost = float("inf")

    # -----------------------------
    # 1. Başlangıç Çözümü Üretme
    # -----------------------------
    def shortest_valid_path(self):
        """Referans olarak en kısa yolu bulur (Dijkstra/BFS)."""
        try:
            path = nx.shortest_path(self.G, self.S, self.D)
            # Yol geçerli mi diye kontrol et
            if total_cost(self.G, path, self.D, self.min_bw) < float("inf"):
                return path
        except:
            return None
        return None

    # -----------------------------
    # 2. Popülasyonu Başlatma (Initialization)
    # -----------------------------
    def initialize(self):
        self.particles.clear()
        self.gbest = None
        self.gbest_cost = float("inf")

        # Önce en az bir geçerli yol bulmamız lazım ki parçacıklar onun varyasyonlarını üretebilsin.
        base = self.shortest_valid_path()
        if not base:
            return

        # Tüm parçacıkları bu temel yoldan başlat (veya rastgele varyasyonlarla)
        for _ in range(self.num_particles):
            # İleride burada rastgelelik eklenebilir. Şu an hepsi aynı noktadan başlıyor.
            p = Particle(base, total_cost(self.G, base, self.D, self.min_bw))
            self.particles.append(p)

        # İlk Gbest'i ayarla
        self.gbest = list(base)
        self.gbest_cost = p.cost

    # -----------------------------
    # 3. Ana Döngü (Optimization Loop)
    # -----------------------------
    def run(self):
        if self.seed is not None:
            random.seed(self.seed)
        self.initialize()

        if not self.gbest:
            return None, float("inf")

        for _ in range(self.iterations):
            for p in self.particles:

                # "Sürekli Uzaydaki Hız" kavramının ayrık (discrete) karşılığı:
                # Gbest ile mevcut yolu bir noktadan kesip birleştirme (Crossover benzeri).
                # Bu işlem, parçacığı Gbest'e doğru "çeker".
                
                if len(self.gbest) < 4:
                    continue

                # Rastgele bir kesim noktası seç
                cut = random.randint(1, len(self.gbest) - 2)
                
                # Yeni yol (Aday): Gbest'in başı + Mevcut yolun sonu
                # Not: Bu çok basit bir kombinasyon, her zaman geçerli yol üretmeyebilir.
                candidate = self.gbest[:cut] + p.position[cut:]

                # 🔒 ZORUNLU GEÇERLİLİK KONTROLLERİ
                # Birleştirme sonucu kaynak ve hedef bozulmuş mu?
                if not candidate or candidate[0] != self.S or candidate[-1] != self.D:
                    continue

                # Maliyet Hesapla
                cost = total_cost(self.G, candidate, self.D, self.min_bw)
                if cost == float("inf"):
                    continue

                # Pbest (Kişisel En İyi) Güncellemesi
                if cost < p.pbest_cost:
                    p.pbest = list(candidate)
                    p.pbest_cost = cost

                # Gbest (Global En İyi) Güncellemesi
                if cost < self.gbest_cost:
                    self.gbest = list(candidate)
                    self.gbest_cost = cost

        return list(self.gbest), float(self.gbest_cost)

# =================================================================================================
# TALEP DOSYASI OKUMA (TEST MODU İÇİN)
# =================================================================================================
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

# =================================================================================================
# ANA PROGRAM
# =================================================================================================
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

    # Tekrarlanabilir sonuçlar için seed=42 kullan
    pso = PSO(G, S, D, B, seed=42)
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
        # Tekrarlanabilir sonuçlar için seed=42 kullan
        pso = PSO(G, s, d, bw, seed=42)
        path, cost = pso.run()

        if path:
            print(f"#{i:02d} {s}->{d} | Cost={cost:.4f}")
        else:
            print(f"#{i:02d} {s}->{d} | ❌ Yol bulunamadı")

    print("\n✅ Tüm PSO testleri tamamlandı.")
