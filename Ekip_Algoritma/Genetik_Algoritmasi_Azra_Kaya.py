# =================================================================================================
# QoS Tabanlı Yol Bulma – Genetik Algoritma (Genetic Algorithm)
# =================================================================================================
# Bu modül, Genetik Algoritma (GA) kullanarak ağ üzerindeki en uygun yolu bulmayı amaçlar.
# GA, doğadaki evrim sürecini taklit eden bir optimizasyon yöntemidir.
#
# TEMEL MANTIK:
# 1. Başlangıçta rastgele yollar üretilir (Popülasyon).
# 2. Her yolun kalitesi (Fitness) hesaplanır (Gecikme, Güvenilirlik, Bant Genişliği).
# 3. En iyi yollar seçilir (Selection).
# 4. Seçilen yollar üzerinde değişiklikler yapılır (Mutation/Crossover - Bu kodda basitleştirilmiş mutasyon var).
# 5. Bu işlem belirli bir nesil (generation) sayısı kadar tekrarlanır.
# =================================================================================================

import pandas as pd
import networkx as nx
import os, math, random

# =================================================================================================
# DOSYA YOLLARI VE YAPILANDIRMA
# =================================================================================================
# Scriptin bulunduğu dizini temel alarak CSV dosyalarının yerini belirler.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================================================================
# YARDIMCI FONKSİYONLAR (GÜVENLİ VERİ DÖNÜŞÜMÜ)
# =================================================================================================
def safe_float(x, default=0.0):
    """CSV'den okunan string veriyi float'a çevirir, virgül/nokta hatasını düzeltir."""
    try:
        return float(str(x).replace(",", "."))
    except:
        return default

def safe_int(x, default=0):
    """CSV'den okunan string veriyi int'e çevirir."""
    try:
        return int(float(str(x).replace(",", ".")))
    except:
        return default

# =================================================================================================
# GRAF YÜKLEME (CSV -> NetworkX)
# =================================================================================================
def load_graph(node_csv, edge_csv):
    """
    Düğüm ve Kenar CSV dosyalarını okuyup NetworkX graf nesnesi oluşturur.
    Her düğüm ve kenara Gecikme, Güvenilirlik ve Bant Genişliği bilgilerini ekler.
    """
    nd = pd.read_csv(node_csv)
    ed = pd.read_csv(edge_csv)

    G = nx.Graph()

    # Düğümleri (Nodes) Ekleme
    for _, r in nd.iterrows():
        G.add_node(
            safe_int(r["node_id"]),
            # Farklı CSV formatlarına uyum sağlamak için alternatif anahtarlar:
            proc_delay=safe_float(r["s_ms"]),
            processing_delay=safe_float(r["s_ms"]),  # Geriye dönük uyumluluk
            node_rel=safe_float(r["r_node"]),
            reliability=safe_float(r["r_node"])      # Geriye dönük uyumluluk
        )

    # Kenarları (Edges/Links) Ekleme
    for _, r in ed.iterrows():
        G.add_edge(
            safe_int(r["src"]),
            safe_int(r["dst"]),
            bandwidth=safe_float(r["capacity_mbps"]),
            link_delay=safe_float(r["delay_ms"]),
            delay=safe_float(r["delay_ms"]),         # Geriye dönük uyumluluk
            link_rel=safe_float(r["r_link"]),
            reliability=safe_float(r["r_link"])      # Geriye dönük uyumluluk
        )

    return G

# =================================================================================================
# TALEP (DEMAND) YÜKLEME
# =================================================================================================
def load_demands(csv_file):
    """Test senaryolarını içeren Demand dosyasını okur."""
    df = pd.read_csv(csv_file)
    demands = []

    for _, r in df.iterrows():
        demands.append({
            "source": safe_int(r["src"]),
            "target": safe_int(r["dst"]),
            "bandwidth": safe_float(r["demand_mbps"])
        })

    return demands

# =================================================================================================
# YOL DOĞRULAMA VE KISIT KONTROLLERİ
# =================================================================================================
def is_valid_path(G, path):
    """Verilen yolun graf üzerinde fiziksel olarak mümkün olup olmadığını kontrol eder."""
    if not path or len(path) < 2:
        return False
    for u, v in zip(path, path[1:]):
        if not G.has_edge(u, v):
            return False
    return True

def check_bandwidth(G, path, bw):
    """Yol üzerindeki TÜM bağlantıların istenen bant genişliğini sağlayıp sağlamadığını kontrol eder."""
    if not is_valid_path(G, path):
        return False
    # Yol üzerindeki darboğazı (en düşük kapasiteli linki) bul ve karşılaştır
    return min(G[u][v]["bandwidth"] for u, v in zip(path, path[1:])) >= bw

# =================================================================================================
# QoS MALİYET (Fitness/Score) HESAPLAMA
# =================================================================================================
def weighted_cost(G, path, w1, w2, w3):
    """
    Bir yolun toplam QoS maliyetini hesaplar.
    Formül: w1*Gecikme + w2*Güvenilirlik + w3*KaynakKullanımı
    """
    # 1. Gecikme: Linklerdeki iletim süresi + Düğümlerdeki işlem süresi
    delay = sum(G[u][v].get("link_delay", G[u][v].get("delay", 0)) for u, v in zip(path, path[1:]))
    delay += sum(G.nodes[n].get("proc_delay", G.nodes[n].get("processing_delay", 0)) for n in path[1:-1])

    # 2. Güvenilirlik: Olasılıkların çarpımı -> Logaritmik toplama dönüşümü
    # Çarpımsal güvenilirliği toplamsal maliyete çevirmek için -log kullanılır.
    reliability = 0.0
    for u, v in zip(path, path[1:]):
        # 1e-12 math.log(0) hatasını önlemek içindir
        reliability += -math.log(max(G[u][v].get("link_rel", G[u][v].get("reliability", 0.99)), 1e-12))
    for n in path:
        reliability += -math.log(max(G.nodes[n].get("node_rel", G.nodes[n].get("reliability", 0.99)), 1e-12))

    # 3. Kaynak Kullanımı: Yüksek bant genişliği = Düşük maliyet (Ters orantı)
    resource = sum(1000.0 / G[u][v]["bandwidth"] for u, v in zip(path, path[1:]))

    return w1 * delay + w2 * reliability + w3 * resource

# =================================================================================================
# GENETİK ALGORİTMA (CORE)
# =================================================================================================
def genetic_algorithm(G, source, target, bw, w1, w2, w3,
                      pop_size=60, generations=120, mutation_rate=0.2):
    """
    Genetik Algoritma ile en iyi yolu arar.
    
    Parametreler:
    - G: Graf
    - source, target: Kaynak ve Hedef
    - bw: İstenen Minimum Bant Genişliği
    - w1, w2, w3: Gecikme, Güvenilirlik ve Kaynak Ağırlıkları
    - pop_size: Popülasyon büyüklüğü (aynı anda kaç yol denenecek)
    - generations: Kaç nesil boyunca evrimleşecek
    """

    # Ağırlıkları normalize et (Toplamı 1 olsun)
    s = w1 + w2 + w3
    w1, w2, w3 = w1/s, w2/s, w3/s

    # --- Yardımcı Fonksiyon: Rastgele Yol Üretme ---
    def random_path(max_steps=60):
        """Rastgele yürüyüş (random walk) ile kaynaktan hedefe bir yol bulmaya çalışır."""
        path = [source]
        current = source

        for _ in range(max_steps):
            # Gittiğimiz yere geri dönmemek için (döngü engelleme) visited kontrolü yapıyoruz
            nbrs = [n for n in G.neighbors(current) if n not in path]
            
            if not nbrs:
                return None # Çıkmaz sokak
            
            if target in nbrs:
                return path + [target] # Hedefe ulaştık!
            
            current = random.choice(nbrs)
            path.append(current)

        return None # Hedefe ulaşamadan adım sayısı bitti

    # 1. ADIM: BAŞLANGIÇ POPÜLASYONU (INITIALIZATION)
    # Rastgele yollar üreterek havuzu dolduruyoruz.
    population = []
    max_attempts = pop_size * 20  # Sonsuz döngüye girmemek için limit
    attempts = 0
    
    print(f"🔍 Popülasyon oluşturuluyor (hedef: {pop_size} birey)...")
    
    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1
        p = random_path()
        # Yol bulunduysa VE bant genişliğini sağlıyorsa havuza ekle
        if p and check_bandwidth(G, p, bw):
            population.append(p)
            if len(population) % 10 == 0:
                print(f"  ✓ {len(population)} birey oluşturuldu...")
    
    print(f"📊 Popülasyon tamamlandı: {len(population)}/{pop_size} birey ({attempts} deneme)")
    
    # Yeterli çeşitlilik (birey) yoksa algoritma çalışamaz
    min_required = max(3, pop_size // 20)  
    if len(population) < min_required:
        print(f"❌ Yetersiz popülasyon! En az {min_required} birey gerekli, sadece {len(population)} oluşturuldu")
        print(f"💡 İpucu: Bandwidth kısıtı çok yüksek olabilir (şu an: {bw} Mbps)")
        return None, float("inf")

    best_path = None
    best_cost = float("inf")

    # 2. ADIM: EVRİM DÖNGÜSÜ (EVOLUTION LOOP)
    for gen in range(generations):
        # Her bireyin skorunu hesapla
        scored = []
        for p in population:
            if check_bandwidth(G, p, bw):
                cost = weighted_cost(G, p, w1, w2, w3)
                scored.append((p, cost))

        if not scored:
            break

        # Skora göre sırala (En düşük maliyet en iyi)
        scored.sort(key=lambda x: x[1])

        # En iyiyi güncelle (Global Best)
        if scored[0][1] < best_cost:
            best_cost = scored[0][1]
            best_path = scored[0][0]

        # ELITIZM: En iyi bireyleri doğrudan sonraki nesile aktar
        # Popülasyonun %10'u "Elite" olarak saklanır.
        elite = [p for p, _ in scored[:max(1, pop_size // 10)]]
        population = elite[:] # Yeni popülasyonu elitlerle başlat

        # Popülasyon dolana kadar elitlerden türet (Basit Kopyalama/Mutasyon)
        # Not: Tam bir crossover yerine burada elitlerden rastgele seçim (selection) kullanılıyor.
        while len(population) < pop_size:
            population.append(random.choice(elite))

    return best_path, best_cost

# =================================================================================================
# MODÜL TEST KODU (Bu dosya doğrudan çalıştırılırsa burası devreye girer)
# =================================================================================================
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
