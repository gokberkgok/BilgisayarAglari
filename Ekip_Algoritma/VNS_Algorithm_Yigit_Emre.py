# =================================================================================================
# QoS Tabanlı Yol Bulma – Değişken Komşuluk Arama (VNS) Algoritması
# =================================================================================================
# Bu modül, Variable Neighborhood Search (VNS) meta-sezgisel yöntemini kullanarak
# ağ üzerinde en uygun maliyetli (QoS) yolu bulur.
#
# TEMEL MANTIK:
# 1. Başlangıç Çözümü: Rastgele veya BFS ile bir ilk yol bulunur.
# 2. Shaking (Çalkalama): Mevcut çözümden rastgele uzaklaşılarak (path üzerinde değişiklik yapılarak)
#    yerel minimumlardan kaçılır.
# 3. Local Search (Yerel Arama): Çözümü iyileştirmek için sistematik aramalar (örn. kısayol bulma) yapılır.
# 4. Neighborhood Change: Eğer iyileşme varsa o noktadan devam edilir (K=1), yoksa daha uzağa bakılır (K artırılır).
# =================================================================================================

import csv
import math
import random
import time
import copy
import os
from collections import deque

# =================================================================================================
# YAPILANDIRMA VE PARAMETRELER
# =================================================================================================
# Maliyet Fonksiyonu Ağırlıkları
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34
MAX_BANDWIDTH_MBPS = 1000.0 # Normalizasyon

# VNS Parametreleri
MAX_VNS_ITER = 20  # Ana döngü sayısı
K_MAX = 3          # Maksimum komşuluk (Shaking) derinliği
TEST_RUNS = 30     # İstatistiksel güvenilirlik için test tekrar sayısı

# =================================================================================================
# DOSYA YOLLARI
# =================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================================================================
# AĞ MATRİSİ VE YARDIMCI SINIFLAR
# =================================================================================================
class NetworkGraph:
    """Ağ verilerini tutan ve maliyet hesaplamalarını yapan sınıf."""
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def load_data(self, node_file, edge_file):
        """CSV dosyalarından düğüm ve kenar bilgilerini yükler."""
        # Düğümler
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

        # Kenarlar
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
                self.edges.setdefault(v, {})[u] = props  # Yönsüz olduğu için çift taraflı

    def calculate_metrics(self, path):
        """
        Verilen bir yol (path) için toplam QoS maliyetini ve ayrıntılı metrikleri hesaplar.
        Dönüş: (Toplam Skor, {Ayrıntılar})
        """
        if not path or len(path) < 2:
            return float("inf"), None

        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        dest = path[-1]

        # Link Maliyetleri
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if v not in self.edges[u]: # Kenar kontrolü
                return float("inf"), None
                
            edge = self.edges[u][v]
            node = self.nodes[v]

            total_delay += edge["delay"]
            reliability_cost += -math.log(edge["r_link"])
            resource_cost += MAX_BANDWIDTH_MBPS / edge["bw"]

            # Ara düğümlerin maliyetleri (Hedef hariç)
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

# =================================================================================================
# VNS ALGORİTMASI
# =================================================================================================
class VNS:
    def __init__(self, graph):
        self.graph = graph

    def initial_path(self, src, dst):
        """BFS ile rastgele bir başlangıç yolu bulur. (VNS için bir tohum çözüm)"""
        queue = deque([(src, [src])])
        visited = {src}

        while queue:
            cur, path = queue.popleft()
            if cur == dst:
                return path

            nbrs = list(self.graph.edges[cur].keys())
            random.shuffle(nbrs) # Rastgelelik ekle (Hep aynı yolu bulmasın)

            for n in nbrs:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, path + [n]))
        return None

    def shake(self, path, k):
        """
        Çalkalama (Shaking) Fonksiyonu:
        Mevcut yoldan rastgele bir parçayı değiştirerek yerel minimumdan kaçmayı sağlar.
        k parametresi, değişikliğin (perturbation) şiddetini belirler (Komşuluk derecesi).
        """
        if len(path) < 4:
            return path

        new_path = copy.deepcopy(path)
        
        # Yol üzerinde rastgele bir segment seç (i -> j arası)
        # k arttıkça aralık genişleyebilir veya daha farklı bir node seçilebilir.
        i = random.randint(1, len(new_path) - 3)
        j = min(len(new_path) - 1, i + k + 1) # k burada segment uzunluğunu etkiliyor

        start = new_path[i - 1]
        end = new_path[j]

        # start ile end arasında yeni (alternatif) bir alt yol bul (Rastgele DFS)
        sub = []
        visited = set(new_path[:i])

        def dfs(cur):
            if cur == end:
                return True
            if len(sub) > 6: # Çok uzatmamak için derinlik sınırı
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
            # Yeni alt yolu ana yola monte et
            return new_path[:i] + sub + new_path[j:]
            
        return path # Eğer alternatif bulunamazsa eski yolu döndür

    def local_search(self, path):
        """
        Yerel Arama (Local Search):
        Mevcut yol üzerinde yapılabilecek "kısayol" (shortcut) iyileştirmelerini tarar.
        Örn: A -> B -> C -> D rotasında A ve D doğrudan bağlıysa, B ve C'yi atlar.
        """
        best = path
        best_cost, _ = self.graph.calculate_metrics(best)

        improved = True
        while improved:
            improved = False
            # Tüm olası kısayolları kontrol et (2-opt benzeri basit mantık)
            for i in range(len(best) - 2):
                for j in range(i + 2, len(best)):
                    u, v = best[i], best[j]
                    # Eğer u ile v arasında doğrudan bağlantı varsa, aradaki düğümleri atla!
                    if v in self.graph.edges[u]:
                        cand = best[:i+1] + best[j:]
                        cost, _ = self.graph.calculate_metrics(cand)
                        if cost < best_cost:
                            best = cand
                            best_cost = cost
                            improved = True
                            break # İyileşme bulundu, döngüyü başa sar
                if improved:
                    break
        return best

    def run(self, src, dst, seed=None):
        """
        VNS Algoritmasının Ana Döngüsü:
        1. Shaking -> Rastgele değiştir
        2. Local Search -> İyileştir
        3. Karşılaştır -> İyiyse kabul et, değilse K'yı artır (daha uzağa bak)
        4. seed -> Tekrarlanabilirlik için
        """
        if seed is not None:
            random.seed(seed)
        path = self.initial_path(src, dst)
        if not path:
            return None, None

        cost, _ = self.graph.calculate_metrics(path)
        best_path, best_cost = path, cost

        for _ in range(MAX_VNS_ITER):
            k = 1
            while k <= K_MAX:
                # 1. Shaking
                shaken = self.shake(best_path, k)
                # 2. Local Search
                improved = self.local_search(shaken)
                # 3. İyileşme Kontrolü
                c, _ = self.graph.calculate_metrics(improved)
                
                if c < best_cost: # Daha iyi bir yol bulundu
                    best_path, best_cost = improved, c
                    k = 1 # Başarı sağlandığı için en yakın komşuluğa dön
                else:
                    k += 1 # İyileşme yok, daha uzağa (derine) bak

        return best_path, self.graph.calculate_metrics(best_path)

# =================================================================================================
# MAIN – TEST SENARYOLARI
# =================================================================================================
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

        # Robustluk için algoritmayı defalarca çalıştır
        for _ in range(TEST_RUNS):
            # Tekrarlanabilir sonuçlar için seed=42 kullan
            path, result = vns.run(s, d, seed=42)
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
