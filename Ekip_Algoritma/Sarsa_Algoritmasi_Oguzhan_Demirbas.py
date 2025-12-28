# =================================================================================================
# QoS Tabanlı Yol Bulma – SARSA Algoritması
# =================================================================================================
# Bu modül, Reinforcement Learning (Pekiştirmeli Öğrenme) yöntemlerinden biri olan
# SARSA (State-Action-Reward-State-Action) algoritmasını gerçekler.
#
# SARSA vs Q-Learning FARKI:
# - Q-Learning (Off-Policy): Bir sonraki durum için "en iyi" (max) aksiyonu düşünerek güncelleme yapar.
# - SARSA (On-Policy): Bir sonraki durum için "gerçekten seçilen" aksiyonu kullanarak güncelleme yapar.
# Bu yüzden SARSA daha temkinli (conservative) yollar öğrenme eğilimindedir.
#
# GÜNCELLEME KURALI:
# Q(s, a) ← Q(s, a) + α * [ R + γ * Q(s', a') - Q(s, a) ]
# =================================================================================================

import networkx as nx
import random
import math
import time
import csv
import os
from collections import defaultdict

# =================================================================================================
# GLOBAL AYARLAR VE DOSYA YOLLARI
# =================================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# Varsayılan Ağırlıklar (Toplam ≈ 1.0)
W_DELAY = 0.33
W_RELIABILITY = 0.33
W_RESOURCE = 0.34

# =================================================================================================
# GRAF OLUŞTURMA (CSV -> NetworkX)
# =================================================================================================
def create_graph_from_csv():
    """
    NodeData.csv ve EdgeData.csv dosyalarını okuyarak yönlü olmayan (Undirected)
    bir NetworkX grafı oluşturur.
    """
    G = nx.Graph()

    # --- Düğümleri (Nodes) Ekle ---
    # CSV kolonları: node_id, s_ms (processing delay), r_node (reliability)
    with open(NODE_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for r in reader:
            G.add_node(
                int(r["node_id"]),
                processing_delay=float(r["s_ms"]),
                reliability=float(r["r_node"])
            )

    # --- Kenarları (Edges) Ekle ---
    # CSV kolonları: src, dst, capacity_mbps, delay_ms, r_link
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

    # --- Bağlılık Kontrolü ---
    # Eğer graf parçalıysa (bölük pörçük), en büyük parçayı (Giant Component) alırız.
    # Böylece algoritma erişilemeyen düğümlerde hata vermez.
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G

# =================================================================================================
# YARDIMCI METRİK HESAPLAMA
# =================================================================================================
def compute_cost(G, path):
    """
    Bir yolun (düğüm listesi) toplam QoS maliyetini hesaplar.
    Maliyet = Ağırlıklı (Gecikme + Güvenilirlik + Kaynak)
    """
    delay = 0
    rel_cost = 0
    res_cost = 0

    # 1. Kenar Maliyetleri (Edge Costs)
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        e = G[u][v]
        
        # Gecikme
        delay += e.get("link_delay", e.get("delay", 0))
        
        # Güvenilirlik (-log dönüşümü)
        # 1e-12 math domain error almamak için
        val_rel = max(e.get("link_rel", e.get("reliability", 0.99)), 1e-12)
        rel_cost += -math.log(val_rel)
        
        # Kaynak (1000 / BW)
        val_bw = max(e["bandwidth"], 1e-6) # 0 bölme hatası önlemi
        res_cost += 1000.0 / val_bw

    # 2. Düğüm Maliyetleri (Node Costs)
    # Başlangıç ve bitiş düğümleri dahil edilmez veya edilir (Burada ara düğümler alınıyor)
    for n in path[1:-1]:
        # Gecikme
        delay += G.nodes[n].get("proc_delay", G.nodes[n].get("processing_delay", 0))
        
        # Güvenilirlik
        val_rel = max(G.nodes[n].get("node_rel", G.nodes[n].get("reliability", 0.99)), 1e-12)
        rel_cost += -math.log(val_rel)

    # 3. Toplam Ağırlıklı Maliyet
    return (
        W_DELAY * delay +
        W_RELIABILITY * rel_cost +
        W_RESOURCE * res_cost
    )

# =================================================================================================
# SARSA ALGORİTMASI (CORE)
# =================================================================================================
def sarsa_route(G, S, D, min_bw, episodes=2000, seed=None):
    """
    SARSA algoritması ile Kaynak(S) -> Hedef(D) arasında yol bulur.
    min_bw: Sadece bant genişliği bu değerden yüksek olan kenarlar kullanılır.
    seed: Tekrarlanabilirlik için rastgele sayı üreteci başlangıç değeri.
    """
    if seed is not None:
        random.seed(seed)

    # Q-Tablosu: Varsayılan değeri 0.0 olan bir sözlük.
    # Anahtar (Key): (state, action) -> (mevcut_düğüm, gidilecek_komşu)
    Q = defaultdict(float)
    
    # Hiperparametreler
    alpha = 0.1     # Öğrenme hızı
    gamma = 0.95    # İndirim faktörü
    epsilon = 0.3   # Keşif oranı

    best_path = None
    best_cost = float("inf")

    # --- Yardımcı: Geçerli Komşuları Bul ---
    def neighbors(u):
        """Düğümün bant genişliği şartını sağlayan komşularını döndürür."""
        return [
            v for v in G.neighbors(u)
            if G[u][v].get("bandwidth", 0) >= min_bw
        ]

    # --- Episode (Eğitim) Döngüsü ---
    for _ in range(episodes):
        state = S
        path = [state]

        # Başlangıçta gidecek yer yoksa pes et
        valid_neighbors = neighbors(state)
        if not valid_neighbors:
            continue

        # İlk aksiyonu seç (Epsilon-Greedy)
        # SARSA, döngüye girmeden önce ilk aksiyonu seçer.
        if random.random() < epsilon:
            action = random.choice(valid_neighbors)
        else:
            # Henüz Q tablosu boşsa rastgele, doluysa en iyisini seç
            action = max(valid_neighbors, key=lambda a: Q[(state, a)]) if valid_neighbors else random.choice(valid_neighbors)

        # --- Adım (Step) Döngüsü ---
        while state != D:
            next_state = action
            path.append(next_state)

            # 1. HEDEFE VARILDI MI?
            if next_state == D:
                # Toplam yol maliyetini hesapla
                cost = compute_cost(G, path)
                
                # Ödül: Maliyet ne kadar azsa ödül o kadar çok (1000 - Cost)
                reward = 1000 - cost
                
                # Son güncellemeyi yap (Next state yok, terminal state)
                # Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
                Q[(state, action)] += alpha * (reward - Q[(state, action)])

                # En iyiyi güncelle
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
                break

            # 2. SONRAKİ DURUMUN ANALİZİ
            next_neighbors = neighbors(next_state)
            if not next_neighbors:
                # Çıkmaz sokak (Dead End)!
                # Çok büyük ceza ver (Negatif ödül)
                reward = -500
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break # Bu epizod yandı, çık.

            # 3. SONRAKİ AKSİYONU SEÇ (ON-POLICY)
            # SARSA'nın Q-Learning'den farkı burada:
            # Bir sonraki aksiyonu (next_action) ŞİMDİ seçiyoruz ve güncelleme formülünde onu kullanıyoruz.
            if random.random() < epsilon:
                next_action = random.choice(next_neighbors)
            else:
                next_action = max(next_neighbors, key=lambda a: Q[(next_state, a)])

            # 4. ANLIK ÖDÜL / CEZA (STEP REWARD)
            # Her adım bir maliyettir. Ajanın yolu uzatmasını engellemek için
            # o kenarın maliyetini negatif olarak (ceza) veriyoruz.
            edge = G[state][next_state]
            
            # Kenar maliyet bileşenleri
            d_val = edge.get("link_delay", edge.get("delay", 0))
            r_val = -math.log(max(edge.get("link_rel", edge.get("reliability", 0.99)), 1e-12))
            b_val = 1000.0 / max(edge.get("bandwidth", 1), 1e-6)
            
            edge_cost = (W_DELAY * d_val + W_RELIABILITY * r_val + W_RESOURCE * b_val)
            
            reward = -edge_cost  # Negatif maliyet
            
            # 5. SARSA GÜNCELLEMESİ
            # Q(s, a) = Q(s, a) + alpha * [ R + gamma * Q(s', a') - Q(s, a) ]
            current_q = Q[(state, action)]
            next_q = Q[(next_state, next_action)]
            
            Q[(state, action)] = current_q + alpha * (reward + gamma * next_q - current_q)

            # Durum ve Aksiyonu İlerle
            state = next_state
            action = next_action

    return best_path, best_cost

# =================================================================================================
# TALEP DOSYASI OKUMA
# =================================================================================================
def load_demands():
    """DemandData.csv dosyasını okuyup (src, dst, bw) listesi döndürür."""
    demands = []
    # encoding='utf-8-sig' BOM karakterini (Excel kaynaklı) temizler
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
    print("QoS Tabanlı Yol Bulma – SARSA\n")

    # Grafı yükle
    G = create_graph_from_csv()
    print(f"Graf yüklendi: {G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar\n")

    # ------------------------------
    # MOD 1: KULLANICI GİRİŞİ
    # ------------------------------
    print("KULLANICI MODU (TEK ÇALIŞMA)")
    try:
        S = int(input("Source (Kaynak): "))
        D = int(input("Destination (Hedef): "))
        B = float(input("Bandwidth (Mbps): "))
        
        # Tekrarlanabilir sonuçlar için seed=42 kullan (arayüz ile tutarlılık)
        path, cost = sarsa_route(G, S, D, B, seed=42)

        if path:
            print("\n✅ EN İYİ YOL BULUNDU:")
            print(" → ".join(map(str, path)))
            print(f"💰 Toplam Maliyet (Cost): {cost:.4f}")
        else:
            print("❌ Uygun bir yol bulunamadı.")
            
    except ValueError:
        print("Lütfen sayısal değer giriniz.")

    # ------------------------------
    # MOD 2: TOPLU TEST (DEMAND CSV)
    # ------------------------------
    print("\n-------------------------------------------")
    print("TEST MODU – DEMAND DATA (Toplu Analiz)")
    print("-------------------------------------------\n")

    demands = load_demands()

    for i, (s, d, bw) in enumerate(demands, 1):
        # Tekrarlanabilir sonuçlar için seed=42 kullan
        path, cost = sarsa_route(G, s, d, bw, seed=42)
        if path:
            print(f"Test #{i:02d} | {s} -> {d} ({bw} Mbps) | ✅ Cost={cost:.4f}")
        else:
            print(f"Test #{i:02d} | {s} -> {d} ({bw} Mbps) | ❌ Başarısız")

    print("\n✅ Tüm testler tamamlandı.")
