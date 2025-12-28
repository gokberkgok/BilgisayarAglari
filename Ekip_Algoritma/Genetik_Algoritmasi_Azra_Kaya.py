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

import pandas as pd  # CSV dosyalarını okumak için kullanılır
import networkx as nx  # Graf yapısını oluşturmak ve yönetmek için kullanılır
import os, math, random  # Dosya yolları, matematiksel işlemler ve rastgele sayı üretimi için

# =================================================================================================
# DOSYA YOLLARI VE YAPILANDIRMA
# =================================================================================================
# Scriptin bulunduğu dizini temel alarak CSV dosyalarının yerini belirler.
# Bu Python dosyasının bulunduğu klasörün tam yolunu al
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Düğüm (Node) verilerini içeren CSV dosyasının tam yolu
NODE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
# Kenar (Edge/Link) verilerini içeren CSV dosyasının tam yolu
EDGE_FILE   = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
# Test senaryolarını (Demand) içeren CSV dosyasının tam yolu
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

# =================================================================================================
# YARDIMCI FONKSİYONLAR (GÜVENLİ VERİ DÖNÜŞÜMÜ)
# =================================================================================================
def safe_float(x, default=0.0):
    """CSV'den okunan string veriyi float'a çevirir, virgül/nokta hatasını düzeltir."""
    try:
        # Veriyi önce string'e çevir, virgülleri noktaya dönüştür (Türkçe sayı formatı için)
        return float(str(x).replace(",", "."))
    except:
        # Hata durumunda varsayılan değeri döndür (veri bozuksa veya boşsa)
        return default

def safe_int(x, default=0):
    """CSV'den okunan string veriyi int'e çevirir."""
    try:
        # Önce float'a çevir (ondalıklı sayıları işlemek için), sonra int'e yuvarla
        return int(float(str(x).replace(",", ".")))
    except:
        # Hata durumunda varsayılan değeri döndür
        return default

# =================================================================================================
# GRAF YÜKLEME (CSV -> NetworkX)
# =================================================================================================
def load_graph(node_csv, edge_csv):
    """
    Düğüm ve Kenar CSV dosyalarını okuyup NetworkX graf nesnesi oluşturur.
    Her düğüm ve kenara Gecikme, Güvenilirlik ve Bant Genişliği bilgilerini ekler.
    """
    # Düğüm (node) CSV dosyasını pandas DataFrame olarak oku
    nd = pd.read_csv(node_csv)
    # Kenar (edge) CSV dosyasını pandas DataFrame olarak oku
    ed = pd.read_csv(edge_csv)

    # Yönsüz (undirected) bir graf nesnesi oluştur
    G = nx.Graph()

    # Düğümleri (Nodes) Ekleme
    # DataFrame'deki her satır için döngü (her satır bir düğümü temsil eder)
    for _, r in nd.iterrows():
        # Grafa yeni bir düğüm ekle
        G.add_node(
            safe_int(r["node_id"]),  # Düğüm ID'si (0-249 arası)
            # Farklı CSV formatlarına uyum sağlamak için alternatif anahtarlar:
            proc_delay=safe_float(r["s_ms"]),  # İşlem gecikmesi (processing delay) - milisaniye
            processing_delay=safe_float(r["s_ms"]),  # Geriye dönük uyumluluk için alternatif anahtar
            node_rel=safe_float(r["r_node"]),  # Düğüm güvenilirliği (0-1 arası olasılık)
            reliability=safe_float(r["r_node"])  # Geriye dönük uyumluluk için alternatif anahtar
        )

    # Kenarları (Edges/Links) Ekleme
    # DataFrame'deki her satır için döngü (her satır bir bağlantıyı temsil eder)
    for _, r in ed.iterrows():
        # Grafa yeni bir kenar (bağlantı) ekle
        G.add_edge(
            safe_int(r["src"]),  # Kaynak düğüm ID'si
            safe_int(r["dst"]),  # Hedef düğüm ID'si
            bandwidth=safe_float(r["capacity_mbps"]),  # Bant genişliği kapasitesi (Mbps)
            link_delay=safe_float(r["delay_ms"]),  # Link gecikmesi (milisaniye)
            delay=safe_float(r["delay_ms"]),  # Geriye dönük uyumluluk için alternatif anahtar
            link_rel=safe_float(r["r_link"]),  # Link güvenilirliği (0-1 arası olasılık)
            reliability=safe_float(r["r_link"])  # Geriye dönük uyumluluk için alternatif anahtar
        )

    return G

# =================================================================================================
# TALEP (DEMAND) YÜKLEME
# =================================================================================================
def load_demands(csv_file):
    """Test senaryolarını içeren Demand dosyasını okur."""
    # Demand CSV dosyasını oku
    df = pd.read_csv(csv_file)
    # Talepleri saklamak için boş liste oluştur
    demands = []

    # Her satır bir test senaryosunu temsil eder
    for _, r in df.iterrows():
        # Her senaryoyu dictionary olarak listeye ekle
        demands.append({
            "source": safe_int(r["src"]),  # Kaynak düğüm
            "target": safe_int(r["dst"]),  # Hedef düğüm
            "bandwidth": safe_float(r["demand_mbps"])  # İstenen minimum bant genişliği
        })

    # Tüm talepleri içeren listeyi döndür
    return demands

# =================================================================================================
# YOL DOĞRULAMA VE KISIT KONTROLLERİ
# =================================================================================================
def is_valid_path(G, path):
    """Verilen yolun graf üzerinde fiziksel olarak mümkün olup olmadığını kontrol eder."""
    # Yol boşsa veya tek düğümden oluşuyorsa geçersiz
    if not path or len(path) < 2:
        return False
    # Yoldaki ardışık her düğüm çifti için kontrol et
    for u, v in zip(path, path[1:]):
        # Eğer u ve v arasında bağlantı yoksa yol geçersiz
        if not G.has_edge(u, v):
            return False
    # Tüm kontroller başarılıysa yol geçerli
    return True

def check_bandwidth(G, path, bw):
    """Yol üzerindeki TÜM bağlantıların istenen bant genişliğini sağlayıp sağlamadığını kontrol eder."""
    # Önce yolun geçerli olup olmadığını kontrol et
    if not is_valid_path(G, path):
        return False
    # Yol üzerindeki darboğazı (en düşük kapasiteli linki) bul ve karşılaştır
    # Eğer en dar boğaz bile istenen bant genişliğinden büyükse, yol uygun demektir
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
    # Yoldaki tüm linklerin gecikmelerini topla
    delay = sum(G[u][v].get("link_delay", G[u][v].get("delay", 0)) for u, v in zip(path, path[1:]))
    # Ara düğümlerin (kaynak ve hedef hariç) işlem gecikmelerini ekle
    delay += sum(G.nodes[n].get("proc_delay", G.nodes[n].get("processing_delay", 0)) for n in path[1:-1])

    # 2. Güvenilirlik: Olasılıkların çarpımı -> Logaritmik toplama dönüşümü
    # Çarpımsal güvenilirliği toplamsal maliyete çevirmek için -log kullanılır.
    # Formül: P_total = P1 * P2 * P3 => -log(P_total) = -log(P1) + -log(P2) + -log(P3)
    reliability = 0.0
    # Tüm linklerin güvenilirlik maliyetini topla
    for u, v in zip(path, path[1:]):
        # 1e-12: math.log(0) hatasını önlemek için minimum değer (sıfıra çok yakın)
        reliability += -math.log(max(G[u][v].get("link_rel", G[u][v].get("reliability", 0.99)), 1e-12))
    # Tüm düğümlerin güvenilirlik maliyetini topla
    for n in path:
        reliability += -math.log(max(G.nodes[n].get("node_rel", G.nodes[n].get("reliability", 0.99)), 1e-12))

    # 3. Kaynak Kullanımı: Yüksek bant genişliği = Düşük maliyet (Ters orantı)
    # Bant genişliği ne kadar yüksekse, kaynak maliyeti o kadar düşük olur
    # 1000.0 sabiti: Ölçeklendirme faktörü (diğer metriklerle dengelemek için)
    resource = sum(1000.0 / G[u][v]["bandwidth"] for u, v in zip(path, path[1:]))

    # Ağırlıklı toplam maliyeti hesapla ve döndür
    # w1: Gecikme ağırlığı, w2: Güvenilirlik ağırlığı, w3: Kaynak ağırlığı
    return w1 * delay + w2 * reliability + w3 * resource

# =================================================================================================
# GENETİK ALGORİTMA (CORE)
# =================================================================================================
def genetic_algorithm(G, source, target, bw, w1, w2, w3,
                      pop_size=60, generations=120, mutation_rate=0.2, seed=None):
    """
    Genetik Algoritma ile en iyi yolu arar.
    
    Parametreler:
    - G: Graf
    - source, target: Kaynak ve Hedef
    - bw: İstenen Minimum Bant Genişliği
    - w1, w2, w3: Gecikme, Güvenilirlik ve Kaynak Ağırlıkları
    - pop_size: Popülasyon büyüklüğü (aynı anda kaç yol denenecek)
    - generations: Kaç nesil boyunca evrimleşecek
    - seed: Tekrarlanabilirlik için seed
    """
    # Eğer seed değeri verilmişse, rastgele sayı üretecini başlat
    # Bu, aynı seed ile her zaman aynı sonuçları almayı sağlar (tekrarlanabilirlik)
    if seed is not None:
        random.seed(seed)

    # Ağırlıkları normalize et (Toplamı 1 olsun)
    # Bu, farklı büyüklükteki ağırlıkların adil karşılaştırılmasını sağlar
    s = w1 + w2 + w3  # Toplam ağırlık
    w1, w2, w3 = w1/s, w2/s, w3/s  # Her ağırlığı toplama böl

    # --- Yardımcı Fonksiyon: Rastgele Yol Üretme ---
    def random_path(max_steps=60):
        """Rastgele yürüyüş (random walk) ile kaynaktan hedefe bir yol bulmaya çalışır."""
        # Yolu kaynak düğümle başlat
        path = [source]
        current = source  # Şu anki konum

        # Maksimum adım sayısı kadar dene
        for _ in range(max_steps):
            # Gittiğimiz yere geri dönmemek için (döngü engelleme) visited kontrolü yapıyoruz
            # Komşuları al, ama sadece daha önce ziyaret edilmemiş olanları
            nbrs = [n for n in G.neighbors(current) if n not in path]
            
            # Eğer hiç komşu yoksa (çıkmaz sokak)
            if not nbrs:
                return None  # Bu yol başarısız, None döndür
            
            # Eğer hedef komşular arasındaysa
            if target in nbrs:
                return path + [target]  # Hedefe ulaştık! Yolu tamamla ve döndür
            
            # Rastgele bir komşu seç ve oraya git
            current = random.choice(nbrs)
            path.append(current)  # Yola ekle

        # Maksimum adım sayısına ulaştık ama hedefe ulaşamadık
        return None

    # 1. ADIM: BAŞLANGIÇ POPÜLASYONU (INITIALIZATION)
    # Rastgele yollar üreterek havuzu dolduruyoruz.
    population = []  # Popülasyonu (bireyleri/yolları) saklamak için boş liste
    max_attempts = pop_size * 20  # Sonsuz döngüye girmemek için maksimum deneme sayısı
    attempts = 0  # Şu ana kadar yapılan deneme sayısı
    
    print(f"🔍 Popülasyon oluşturuluyor (hedef: {pop_size} birey)...")
    
    # Popülasyon hedef büyüklüğe ulaşana kadar veya maksimum deneme sayısına ulaşana kadar devam et
    while len(population) < pop_size and attempts < max_attempts:
        attempts += 1  # Deneme sayacını artır
        p = random_path()  # Rastgele bir yol üret
        # Yol bulunduysa VE bant genişliğini sağlıyorsa havuza ekle
        if p and check_bandwidth(G, p, bw):
            population.append(p)  # Geçerli yolu popülasyona ekle
            # Her 10 bireyden birinde ilerleme raporu ver
            if len(population) % 10 == 0:
                print(f"  ✓ {len(population)} birey oluşturuldu...")
    
    print(f"📊 Popülasyon tamamlandı: {len(population)}/{pop_size} birey ({attempts} deneme)")
    
    # Yeterli çeşitlilik (birey) yoksa algoritma çalışamaz
    # Minimum gerekli birey sayısı: En az 3 veya popülasyon hedefinin %5'i
    min_required = max(3, pop_size // 20)  
    if len(population) < min_required:
        # Yetersiz popülasyon uyarısı
        print(f"❌ Yetersiz popülasyon! En az {min_required} birey gerekli, sadece {len(population)} oluşturuldu")
        print(f"💡 İpucu: Bandwidth kısıtı çok yüksek olabilir (şu an: {bw} Mbps)")
        # Başarısız sonuç döndür (yol yok, maliyet sonsuz)
        return None, float("inf")

    # Şu ana kadar bulunan en iyi yol ve maliyeti sakla
    best_path = None  # Henüz yol bulunamadı
    best_cost = float("inf")  # Başlangıç maliyeti sonsuz (en kötü durum)

    # 2. ADIM: EVRİM DÖNGÜSÜ (EVOLUTION LOOP)
    # Belirtilen nesil sayısı kadar evrim sürecini tekrarla
    for gen in range(generations):
        # Her bireyin skorunu hesapla
        scored = []  # (yol, maliyet) çiftlerini saklamak için liste
        # Popülasyondaki her birey için
        for p in population:
            # Bant genişliği kısıtını hala sağlıyorsa
            if check_bandwidth(G, p, bw):
                # QoS maliyetini hesapla
                cost = weighted_cost(G, p, w1, w2, w3)
                # Yol ve maliyetini listeye ekle
                scored.append((p, cost))

        # Eğer hiç geçerli birey kalmadıysa döngüyü kır
        if not scored:
            break

        # Skora göre sırala (En düşük maliyet en iyi)
        # lambda x: x[1] => Her (yol, maliyet) çiftinin maliyet kısmına göre sırala
        scored.sort(key=lambda x: x[1])

        # En iyiyi güncelle (Global Best)
        # Eğer bu nesildeki en iyi birey, şimdiye kadarki en iyiden daha iyiyse
        if scored[0][1] < best_cost:
            best_cost = scored[0][1]  # En iyi maliyeti güncelle
            best_path = scored[0][0]  # En iyi yolu güncelle

        # ELITIZM: En iyi bireyleri doğrudan sonraki nesile aktar
        # Popülasyonun %10'u "Elite" olarak saklanır (en az 1 birey)
        # Bu, en iyi çözümlerin kaybolmamasını garantiler
        elite = [p for p, _ in scored[:max(1, pop_size // 10)]]
        population = elite[:]  # Yeni popülasyonu elitlerle başlat (kopyala)

        # Popülasyon dolana kadar elitlerden türet (Basit Kopyalama/Mutasyon)
        # Not: Tam bir crossover yerine burada elitlerden rastgele seçim (selection) kullanılıyor.
        # Bu basitleştirilmiş bir yaklaşımdır; klasik GA'da crossover ve mutasyon olur
        while len(population) < pop_size:
            # Rastgele bir elit bireyi seç ve popülasyona ekle
            population.append(random.choice(elite))

    # Tüm nesiller tamamlandı, en iyi yolu ve maliyetini döndür
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

    # Tekrarlanabilir sonuçlar için seed=42 kullan
    path, cost = genetic_algorithm(G, source, target, bw, w1, w2, w3, seed=42)

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
            # Tekrarlanabilir sonuçlar için seed=42 kullan
            p, c = genetic_algorithm(
                G, d["source"], d["target"], d["bandwidth"],
                w1, w2, w3,
                seed=42
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
