# =================================================================================================
# QoS Tabanlı Yol Bulma – Q-Learning Algoritması
# =================================================================================================
# Bu modül, Reinforcement Learning (Pekiştirmeli Öğrenme) yöntemlerinden biri olan 
# Q-Learning algoritmasını kullanarak ağ optimizasyonu yapar.
#
# TEMEL MANTIK:
# Bir "Agent" (Ajan) ağ üzerinde rastgele dolaşarak (keşif) hangi yolların daha iyi olduğunu öğrenir.
# Her adımda bir ödül (reward) veya ceza alır.
# Q-Tablosu (State-Action Matrix), ajanın deneyimlerini saklar.
#
# Q-LEARNING FORMÜLÜ (BELLMAN DENKLEMİ):
# Q(s, a) = Q(s, a) + alpha * [ Reward + gamma * max(Q(s', a')) - Q(s, a) ]
# - s: Mevcut durum (düğüm)
# - a: Aksiyon (gittiği komşu düğüm)
# - alpha: Öğrenme hızı (eski bilgi ile yeni bilgi arasındaki denge)
# - gamma: Gelecek odaklılık (gelecekteki ödüllerin şimdiki değeri)
# =================================================================================================

import random
import math
import networkx as nx
import pandas as pd
import os
import sys

# =================================================================================================
# GLOBAL PARAMETRELER VE YAPILANDIRMA
# =================================================================================================
# Ağ Parametreleri (Rastgele oluşturulursa kullanılır)
NODE_COUNT = 250
EDGE_PROBABILITY = 0.4

# Link Özellikleri (Random fallback değerleri)
BANDWIDTH_MIN = 100
BANDWIDTH_MAX = 1000
LINK_DELAY_MIN = 3
LINK_DELAY_MAX = 15
LINK_RELIABILITY_MIN = 0.95
LINK_RELIABILITY_MAX = 0.999

# Q-Learning Hiperparametreleri
# Bu değerler algoritmanın öğrenme performansını doğrudan etkiler.
ALPHA = 0.1          # Öğrenme Oranı (Learning Rate): Ajansın yeni bilgilere ne kadar hızlı adapte olacağı.
GAMMA = 0.99         # İndirim Faktörü (Discount Factor): Gelecekteki ödüllerin önemi (0-1 arası).
EPSILON = 0.2        # Keşif Oranı (Exploration Rate): Rastgele hareket etme olasılığı.
EPISODES = 300       # Bölüm Sayısı: Ajanın kaç kez baştan sona gidip geleceği.
MAX_STEPS = 250      # Maksimum Adım: Bir bölümde sonsuz döngüye girmemek için limit.

# Maliyet Ağırlıkları (Kullanıcı Arayüzünden de gelebilir)
W_DELAY = 0.4        # Gecikme ağırlığı
W_RELIABILITY = 0.4  # Güvenilirlik ağırlığı
W_RESOURCE = 0.2     # Kaynak kullanım ağırlığı

# Test için varsayılan Kaynak ve Hedef
SOURCE = 2
DESTINATION = 8


# =================================================================================================
# MALİYET (COST) FONKSİYONLARI
# =================================================================================================
def path_total_delay(G, path):
    """
    Yol üzerindeki toplam gecikmeyi (ms) hesaplar.
    Gecikme = Kenar Gecikmeleri + Düğüm İşlem Gecikmeleri
    """
    delay = 0
    # Kenar gecikmeleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        delay += G.edges[u, v]['link_delay']
    # Düğüm gecikmeleri (Başlangıç ve bitiş hariç ara düğümler)
    for k in path[1:-1]:
        delay += G.nodes[k]['proc_delay']
    return delay

def path_reliability_cost(G, path):
    """
    Yolun güvenilirlik maliyetini hesaplar.
    Güvenilirlik çarpımsal olduğu için (R_total = R1 * R2 ...), 
    toplamsal maliyete çevirmek için logaritma kullanıyoruz: Cost = -log(R)
    """
    cost = 0
    # Kenar güvenilirliği
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        # Log(0) hatasını önlemek için güvenilirlik çok küçükse belli bir sınır konabilir ama şimdilik doğrudan alıyoruz
        val = G.edges[u, v]['link_rel']
        if val <= 0: cost += float('inf')
        else: cost += -math.log(val)
    
    # Düğüm güvenilirliği
    for k in path:
        val = G.nodes[k]['node_rel']
        if val <= 0: cost += float('inf')
        else: cost += -math.log(val)
        
    return cost

def path_resource_cost(G, path):
    """
    Bant genişliğine dayalı kaynak kullanım maliyeti.
    Daha yüksek bant genişliği = Daha düşük maliyet (1/BW mantığı).
    """
    cost = 0
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        bw = G.edges[u, v]['bandwidth']
        if bw <= 0: cost += float('inf')
        else: cost += (1000.0 / bw)
    return cost


def total_cost(G, path, w_delay, w_rel, w_res):
    """
    Verilen ağırlıklara göre normalize edilmiş toplam maliyet skoru.
    Bu skor ne kadar düşükse, yol o kadar iyidir.
    """
    return (w_delay * path_total_delay(G, path) +
            w_rel   * path_reliability_cost(G, path) +
            w_res   * path_resource_cost(G, path))


# =================================================================================================
# Q-LEARNING AGENT SINIFI
# =================================================================================================
class QLearning:
    def __init__(self, G, alpha, gamma, epsilon):
        self.G = G
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-Tablosunun Başlatılması
        # Her düğüm (state) için komşularına (action) giden kenarların değeri 0 ile başlar.
        # Yapı: {Düğüm_ID: {Komşu_1: 0.0, Komşu_2: 0.0, ...}}
        self.Q = {n: {nb: 0.0 for nb in G.neighbors(n)} for n in G.nodes()}

    def choose(self, s):
        """
        EPSILON-GREEDY yaklaşımı ile bir sonraki adımı (aksiyonu) seçer.
        - %Epsilon ihtimalle: Rastgele bir komşuya git (KEŞİF / EXPLORATION).
        - %(1-Epsilon) ihtimalle: Q değeri en yüksek olan komşuya git (SÖMÜRÜ / EXPLOITATION).
        """
        neighbors = list(self.Q[s].keys())
        if not neighbors:
            return None # Çıkmaz sokak
        
        # Rastgele keşif (Exploration)
        if random.random() < self.epsilon:
            return random.choice(neighbors)
            
        # En iyi bilinen yolu seç (Exploitation)
        max_q = max(self.Q[s].values())
        # Birden fazla en iyi varsa, aralarından rastgele seç
        best = [a for a, q in self.Q[s].items() if q == max_q]
        return random.choice(best)

    def update(self, s, a, r, s_next):
        """
        BELLMAN DENKLEMİ ile Q değerini günceller.
        Q(s,a) = Q(s,a) + alpha * (Reward + gamma * max(Q(s',all)) - Q(s,a))
        
        Args:
            s (int): Mevcut düğüm (Current State)
            a (int): Gidilen komşu düğüm (Action)
            r (float): Alınan ödül (Reward)
            s_next (int): Bir sonraki durum (Next State). Hedefe varıldıysa None olabilir.
        """
        max_next = 0
        
        # Bir sonraki adımdaki en iyi Q değerini bul (Gelecek tahmini)
        if s_next is not None and s_next in self.Q and len(self.Q[s_next]) > 0:
            max_next = max(self.Q[s_next].values())
            
        # Hedeflenen yeni değer (Target)
        td = r + self.gamma * max_next
        
        # Mevcut değeri güncelle
        self.Q[s][a] += self.alpha * (td - self.Q[s][a])


# =================================================================================================
# GRAF OLUŞTURMA YARDIMCISI
# =================================================================================================
def generate_graph(N, p):
    """
    Test amaçlı rastgele graf oluşturur veya CSV'den veri okumayı dener.
    Önce NodeData.csv dosyasını okumaya çalışır, başaramazsa rastgele özellikler atar.
    """
    
    print(f"{'='*60}")
    print(f"GRAF OLUŞTURULUYOR: {N} düğüm, Bağlantı Olasılığı {p}")
    print(f"{'='*60}")
    
    # 1. CSV Okuma Denemesi
    try:
        cwd = os.getcwd()
        fpath = os.path.join(cwd, "BSM307_317_Guz2025_TermProject_NodeData.csv")
        try:
             df = pd.read_csv(fpath, sep=";", decimal=",")
        except:
             df = pd.read_csv(fpath, sep=",", decimal=".")
        
        # Kolon isimlerini standartlaştır
        # Beklenen: node_id, s_ms (processing delay), r_node (reliability)
        # Ancak burada manuel atama yapılmış, bu kısmı CSV formatına göre esnekleştirmek gerekebilir.
        df.columns = ["node_id", "processing_delay", "reliability"]
        
        if len(df) < N:
            print(f"⚠️  UYARI: CSV'de sadece {len(df)} düğüm var, N={N} olarak güncellendi.")
            N = len(df)
            
    except Exception as e:
        print(f"❌ HATA: NodeData.csv okunamadı! Rastgele değerler kullanılacak. ({str(e)})")
        # Programı durdurmak yerine devam edelim ama hatayı belirtelim.
        # exit(1) -> Arayüzde hataya sebep olmaması için kaldırdım.

    # 2. Topoloji Oluşturma (Erdős-Rényi Rastgele Graf Modeli)
    G = nx.erdos_renyi_graph(N, p)

    # 3. Grafın Bağlı Olmasını Garanti Et
    # Parçalı bulutlu (bağlantısız) graf olursa tüm düğümlere erişilemez.
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            # Her bir bileşenden rastgele bir düğüm seçip birbirine bağla
            a = random.choice(list(comps[i]))
            b = random.choice(list(comps[i + 1]))
            G.add_edge(a, b)
        print("⚠️  Graf bağlantısızdı, ek kenarlar ile bağlandı.")


    # 4. Düğüm ve Kenar Özelliklerini Atama
    
    # Node Attributes (CSV'den veya Rastgele)
    for n in G.nodes():
        if 'df' in locals() and n < len(df):
            G.nodes[n]['proc_delay'] = float(df.iloc[n]["processing_delay"])
            G.nodes[n]['node_rel'] = float(df.iloc[n]["reliability"])
        else:
            G.nodes[n]['proc_delay'] = 1.0 # Varsayılan ms
            G.nodes[n]['node_rel'] = 0.95  # Varsayılan %95
            
    # Edge Attributes (Rastgele)
    # Not: Gerçek uygulamada EdgeData.csv okunmalıdır, burada simülasyon yapılıyor.
    for u, v in G.edges():
        G.edges[u, v]['bandwidth'] = random.uniform(BANDWIDTH_MIN, BANDWIDTH_MAX)
        G.edges[u, v]['link_delay'] = random.uniform(LINK_DELAY_MIN, LINK_DELAY_MAX)
        G.edges[u, v]['link_rel'] = random.uniform(LINK_RELIABILITY_MIN, LINK_RELIABILITY_MAX)

    print(f"✅ Graf hazır: {len(G.nodes)} düğüm, {len(G.edges)} kenar")
    return G


# =================================================================================================
# Q-LEARNING EĞİTİM LOOP (Training Loop)
# =================================================================================================
def train_q_learning(G, source, destination, alpha, gamma, epsilon, episodes, max_steps, w_delay, w_rel, w_res):
    """
    Q-Learning ajanını eğiterek en iyi rotayı bulmasını sağlar.
    
    Döngü:
    1. Her episode (bölüm) için baştan başla (Kaynak düğüm).
    2. Hedefe varana kadar veya max adıma kadar yürü.
    3. Her adımda Q tablosunu güncelle.
    4. Hedefe varınca büyük bir ödül ver ve en iyi yolu kaydet.
    """
    
    print(f"\n🎓 EĞİTİM PARAMETRELERİ:")
    print(f"  Kaynak->Hedef: {source} -> {destination}")
    print(f"  Hiperparametreler: Alpha={alpha}, Gamma={gamma}, Epsilon={epsilon}")
    print(f"  Ağırlıklar: Delay={w_delay}, Rel={w_rel}, Res={w_res}")

    # Ağırlık Normalizasyonu
    total_w = w_delay + w_rel + w_res
    if total_w > 0:
        w_delay /= total_w
        w_rel /= total_w
        w_res /= total_w

    # Ajanı (Agent) Başlat
    agent = QLearning(G, alpha, gamma, epsilon)

    best_path = None
    best_cost = float("inf")

    # --- EPISODE DÖNGÜSÜ ---
    for ep in range(episodes):
        s = source
        path = [s] # Mevcut epizodun izlediği yol

        # --- STEP DÖNGÜSÜ ---
        for step in range(max_steps):
            # 1. Aksiyon Seç
            a = agent.choose(s)
            
            # Eğer gidecek yer yoksa (çıkmaz sokak) epizodu bitir
            if a is None:
                break

            path.append(a)

            # 2. Hedef Kontrolü ve Ödül
            # Eğer hedefe ulaştıysak;
            if a == destination:
                # Yolun toplam maliyetini hesapla
                cost = total_cost(G, path, w_delay, w_rel, w_res)
                
                # Ödül fonksiyonu: Maliyet ne kadar düşükse ödül o kadar büyük olmalı.
                # Örnek: Cost 10 ise Reward 1000, Cost 100 ise Reward 100.
                if cost > 0:
                    reward = 10000 / cost
                else:
                    reward = 10000 # Maliyet 0 ise (imkansız ama) sabit büyük ödül
                
                # Q Değerini güncelle (s -> a hamlesi mükemmeldi!)
                agent.update(s, a, reward, None) # Next state None çünkü bitti

                # Global En İyiyi Güncelle
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path) # Kopyasını al

                break # Epizot bitti, yenisine geç
            
            # 3. Ara Adım Güncellemesi
            # Hedefe varmadık, yola devam ediyoruz.
            # Ceza (-1) vererek ajanı kısa yolları bulmaya teşvik ediyoruz (daha az adım = daha az ceza).
            # VEYA maliyete dayalı anlık ceza verilebilir.
            agent.update(s, a, -1, a)
            
            # Konumu güncelle
            s = a

        # İlerleme Logu (Her 100 epizodda bir)
        if (ep + 1) % 100 == 0:
            print(f"📊 Episode {ep + 1}/{episodes} tamamlandı... (Şu ana kadarki en iyi maliyet: {best_cost:.2f})")

    print(f"✅ Eğitim tamamlandı!\n")
    return best_path, best_cost


# =================================================================================================
# SONUÇ GÖSTERİMİ
# =================================================================================================
def print_results(G, path, cost):
    """Bulunan yolun detaylarını ve metriklerini ekrana basar."""
    
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

    # Metrikleri ayrı ayrı hesapla
    delay = path_total_delay(G, path)
    rel = path_reliability_cost(G, path)
    res = path_resource_cost(G, path)

    print(f"\n📊 METRİKLER (Ayrıştırılmış):")
    print(f"  ⏱️  Gecikme (Delay): {delay:.3f} ms")
    print(f"  🔒 Güvenilirlik Skoru: {rel:.4f}")
    print(f"  📊 Kaynak Skoru: {res:.4f}")
    
    print(f"\n{'─'*60}")
    print("📍 DÜĞÜM DETAYLARI:")
    print(f"{'─'*60}")
    print(f"{'Node ID':<10} {'Delay (ms)':<15} {'Reliability':<15}")
    print(f"{'─'*60}")
    for node in path:
        proc_delay = G.nodes[node]['proc_delay']
        node_rel = G.nodes[node]['node_rel']
        print(f"{node:<10} {proc_delay:<15.3f} {node_rel:<15.6f}")
    print(f"{'='*60}\n")


# =================================================================================================
# ANA PROGRAM (DEBUG / TEST)
# =================================================================================================
def main():
    """Modül tek başına çalıştırıldığında burası devreye girer."""
    
    # 1. Graf Kur
    G = generate_graph(NODE_COUNT, EDGE_PROBABILITY)
    
    # 2. Geçerlilik Kontrolleri
    if SOURCE >= len(G.nodes) or DESTINATION >= len(G.nodes):
        print(f"❌ HATA: Kaynak ({SOURCE}) veya Hedef ({DESTINATION}) graf sınırları dışında!")
        sys.exit(1)
    
    if SOURCE == DESTINATION:
        print(f"❌ HATA: Kaynak ve hedef aynı olamaz!")
        sys.exit(1)
    
    if not nx.has_path(G, SOURCE, DESTINATION):
        print(f"❌ HATA: {SOURCE} ile {DESTINATION} arasında fiziksel bir yol yok!")
        sys.exit(1)
    
    # 3. Eğitimi Başlat
    best_path, best_cost = train_q_learning(
        G, SOURCE, DESTINATION,
        ALPHA, GAMMA, EPSILON,
        EPISODES, MAX_STEPS,
        W_DELAY, W_RELIABILITY, W_RESOURCE
    )
    
    # 4. Sonucu Göster
    print_results(G, best_path, best_cost)


if __name__ == "__main__":
    main()