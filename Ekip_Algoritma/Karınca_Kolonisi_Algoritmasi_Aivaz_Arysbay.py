# =================================================================================================
# QoS Tabanlı Yol Bulma – Karınca Kolonisi (ACO) ve Genetik Algoritma (GA)
# =================================================================================================
# Bu dosya, iki farklı sezgisel (heuristic) algoritmayı içerir:
# 1. Ant Colony Optimization (ACO): Karıncaların feromon izini takip ederek yol bulması.
# 2. Genetic Algorithm (GA): Evrimsel süreçle en iyi yolun bulunması.
#
# Ayrıca PyQt6 tabanlı bir arayüz ile bu iki algoritmanın karşılaştırmalı testine olanak tanır.
# =================================================================================================

import sys
import networkx as nx
import numpy as np
import random
import math
import time
import csv
import os
from collections import defaultdict

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QTextEdit, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, 
    QProgressBar, QMessageBox, QComboBox
)

# ==========================================
# 1. VERİ YÜKLEME İŞLEMLERİ
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")

def create_graph_from_csv():
    G = nx.Graph()
    
    try:
        with open(NODE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader:
                if len(row) < 3: continue
                try:
                    node_id = int(row[0])
                    proc_delay = float(row[1].replace(',', '.'))
                    reliability = float(row[2].replace(',', '.'))
                    G.add_node(node_id, processing_delay=proc_delay, reliability=reliability)
                except ValueError: continue
    except FileNotFoundError: print("Hata: Node dosyası bulunamadı.")

    
    try:
        with open(EDGE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            next(reader)
            for row in reader:
                if len(row) < 5: continue
                try:
                    u, v = int(row[0]), int(row[1])
                    bw = float(row[2].replace(',', '.'))
                    delay = float(row[3].replace(',', '.'))
                    rel = float(row[4].replace(',', '.'))
                    G.add_edge(u, v, bandwidth=bw, delay=delay, reliability=rel)
                except ValueError: continue
    except FileNotFoundError: print("Hata: Edge dosyası bulunamadı.")
    return G

def compute_metrics(G, path):
    """
    Verilen bir yol için QoS metriklerini ve ham maliyet bileşenlerini hesaplar.
    
    Args:
        G (nx.Graph): Ağ grafı
        path (list): Düğüm ID'lerinden oluşan yol listesi (Örn: [0, 5, 10])
        
    Returns:
        tuple: (Toplam Gecikme, Güvenilirlik Log Toplamı, Kaynak Maliyeti, Gerçek Güvenilirlik Çarpımı)
    """
    total_delay = 0
    rel_log_sum = 0
    res_cost_sum = 0
    true_rel = 1.0 # Gerçek kümülatif güvenilirlik (Çarpım)

    if not path: return 0, 0, 0, 0

    # 1. Hatta (Link) Ait Metriklerin Hesaplanması
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge = G[u][v]
        d = edge.get("delay", 10)
        r = edge.get("reliability", 0.9)
        bw = edge.get("bandwidth", 100)

        # Gecikme (Toplamsal)
        total_delay += d
        
        # Güvenilirlik (Çarpımsal -> Logaritmik Toplama Dönüşümü)
        # log(a*b) = log(a) + log(b). Maliyet minimizasyonu için -log(r) kullanılır.
        if r <= 0: r = 0.0001
        rel_log_sum += -math.log(r)
        
        # Kaynak Maliyeti (Bant genişliği ile ters orantılı)
        # Yüksek hız = Düşük maliyet.
        if bw <= 0: bw = 0.1
        res_cost_sum += (1000.0 / bw)
        
        true_rel *= r

    # 2. Düğüm (Node) Üzerindeki İşlemci Gecikmesi ve Güvenilirliği
    for i, node in enumerate(path):
        node_data = G.nodes[node]
        r = node_data.get("reliability", 0.99)
        proc_delay = node_data.get("processing_delay", 0)
        
        if r <= 0: r = 0.0001
        rel_log_sum += -math.log(r)
        
        true_rel *= r
        total_delay += proc_delay
            
    return total_delay, rel_log_sum, res_cost_sum, true_rel

def calculate_total_cost(G, path, weights):
    """Toplam ağırlıklı maliyet hesaplar"""
    if not path: return float('inf')
    d, r_cost, res_cost, _ = compute_metrics(G, path)
    w_d, w_r, w_res = weights
    return (w_d * d) + (w_r * r_cost) + (w_res * res_cost)

# =================================================================================================
# 2. ACO ÇÖZÜCÜ (KARINCA KOLONİSİ ALGORİTMASI)
# =================================================================================================

class ACOSolver:
    """
    Ant Colony Optimization (ACO) Algoritması
    Karıncalar, feromon izlerini ve sezgisel bilgiyi (visibility) kullanarak yol seçer.
    """
    @staticmethod
    def solve(graph, source, target, weights, min_bw, num_ants=20, num_iterations=30):
        # ----------------------------------------------------------------
        # 1. ACO PARAMETRELERİNİN TANIMLANMASI
        # ----------------------------------------------------------------
        # alpha: Feromon miktarının (iz) karınca üzerindeki etkisi (Güdü katsayısı).
        alpha = 1.0           
        # beta: Sezgisel bilginin (uzaklık/maliyet) karınca kararındaki etkisi.
        beta = 2.0            
        # evaporation_rate: Feromonun her turda buharlaşma oranı (0.1 = %10 azalır).
        # Bu, eski yolların zamanla unutulmasını ve yeni yolların keşfini sağlar.
        evaporation_rate = 0.1 
        # Q: Bir karıncanın bıraktığı toplam feromon miktarı sabiti.
        Q = 100.0             
        # tau_min: Bir kenardaki minimum feromon miktarı (Sıfıra inmemesi için).
        tau_min = 0.1         
        # tau_max: Bir kenardaki maksimum feromon miktarı (Doygunluk sınırı).
        tau_max = 10.0        

        # ----------------------------------------------------------------
        # 2. FEROMON HARİTASININ BAŞLATILMASI
        # ----------------------------------------------------------------
        # Feromon değerlerini tutacak sözlük yapısı tanımlanır.
        pheromones = {}
        # Graftaki tüm kenarlar üzerinde döngü başlatılır.
        for u, v in graph.edges():
            # (u, v) yönü için başlangıç feromonu atanır (1.0).
            pheromones[(u, v)] = 1.0
            # (v, u) yönü için başlangıç feromonu atanır (Simetrik).
            pheromones[(v, u)] = 1.0

        # Global en iyi yol değişkeni (Başlangıçta yok).
        global_best_path = None
        # Global en iyi maliyet değişkeni (Başlangıçta sonsuz).
        global_best_cost = float('inf')

        # Algoritma başlangıç zamanı kaydedilir.
        start_time = time.time()

        # ----------------------------------------------------------------
        # 3. İTERASYON DÖNGÜSÜ (EĞİTİM)
        # ----------------------------------------------------------------
        # Belirlenen iterasyon sayısı kadar döngü çalıştırılır.
        for iteration in range(num_iterations):
            # Bu iterasyonda bulunan tüm yolları ve maliyetlerini tutacak liste.
            paths_in_iteration = []

            # ------------------------------------------------------------
            # 4. KARINCA KOLONİSİ DÖNGÜSÜ
            # ------------------------------------------------------------
            # Her iterasyonda 'num_ants' kadar karınca yola çıkarılır.
            for ant in range(num_ants):
                # Karınca, kaynaktan hedefe bir yol bulmak için _ant_walk fonksiyonunu çağırır.
                path = ACOSolver._ant_walk(graph, source, target, pheromones, alpha, beta, min_bw, weights)
                
                # Eğer karınca başarılı bir şekilde hedefe ulaştıysa (yol boş değilse):
                if path:
                    # Bulunan yolun toplam QoS maliyeti hesaplanır.
                    cost = calculate_total_cost(graph, path, weights)
                    # Yol ve maliyet, bu iterasyonun listesine eklenir.
                    paths_in_iteration.append((path, cost))
                    
                    # Eğer bulunan maliyet, şu ana kadarki en iyi maliyetten düşükse:
                    if cost < global_best_cost:
                        # Global en iyi maliyet güncellenir.
                        global_best_cost = cost
                        # Global en iyi yol güncellenir (Listenin kopyası alınır).
                        global_best_path = list(path)

            # ------------------------------------------------------------
            # 5. FEROMON BUHARLAŞMASI (EVAPORATION)
            # ------------------------------------------------------------
            # Mevcut tüm feromon yolları (kenarları) üzerinde döngü.
            for key in pheromones:
                # Mevcut feromon miktarı, buharlaşma oranı kadar azaltılır.
                pheromones[key] *= (1.0 - evaporation_rate)
                
                # Eğer feromon miktarı minimum sınırın altına düştüyse:
                if pheromones[key] < tau_min: 
                    # Minimum sınıra (tau_min) eşitlenir.
                    pheromones[key] = tau_min

            # ------------------------------------------------------------
            # 6. FEROMON GÜNCELLEMESİ (DEPOSIT - YERELEL)
            # ------------------------------------------------------------
            # Bu iterasyonda bulunan başarılı yollar üzerinde döngü.
            for path, cost in paths_in_iteration:
                # Bırakılacak feromon miktarı hesaplanır (Maliyet ne kadar azsa, feromon o kadar çok).
                # Eğer maliyet 0 veya negatifse (teorik), sabit Q kullanılır.
                deposit = Q / cost if cost > 0 else Q
                
                # Yol üzerindeki her bir kenar (bağlantı) için döngü.
                for i in range(len(path) - 1):
                    # Kenarın başlangıç (u) ve bitiş (v) düğümleri alınır.
                    u, v = path[i], path[i+1]
                    
                    # (u, v) yönündeki feromona deposit miktarı eklenir.
                    # Maksimum sınır (tau_max) kontrolü yapılır.
                    pheromones[(u, v)] = min(tau_max, pheromones[(u, v)] + deposit) 
                    
                    # (v, u) yönündeki feromona da aynı miktar eklenir (Yönsüz graf varsayımı).
                    pheromones[(v, u)] = min(tau_max, pheromones[(v, u)] + deposit)

            # ------------------------------------------------------------
            # 7. ELİTİST FEROMON GÜNCELLEMESİ (GLOBAL BEST)
            # ------------------------------------------------------------
            # Eğer şimdiye kadar bulunmuş en iyi bir yol varsa:
            if global_best_path:
                # En iyi yol için ekstra ödül feromonu hesaplanır (2 kat etkili).
                deposit = (Q / global_best_cost) * 2.0 
                
                # En iyi yolun kenarları üzerinde döngü.
                for i in range(len(global_best_path) - 1):
                    u, v = global_best_path[i], global_best_path[i+1]
                    # Kenarlara ekstra feromon eklenir ve sınır kontrolü yapılır.
                    pheromones[(u, v)] = min(tau_max, pheromones[(u, v)] + deposit)
                    pheromones[(v, u)] = min(tau_max, pheromones[(v, u)] + deposit)

        # Toplam geçen süre milisaniye cinsinden hesaplanır.
        elapsed = (time.time() - start_time) * 1000
        # En iyi yol, en iyi maliyet ve geçen süre döndürülür.
        return global_best_path, global_best_cost, elapsed

    @staticmethod
    def _ant_walk(graph, start_node, end_node, pheromones, alpha, beta, min_bw, weights):
        """Tek bir karıncanın kaynaktan hedefe yürüyüşü."""
        # Karıncanın şu anki konumu başlangıç düğümüne atanır.
        current_node = start_node
        # Karıncanın izlediği yol listesi başlatılır.
        path = [current_node]
        # Ziyaret edilen düğümler kümesi oluşturulur (Döngüleri önlemek için).
        visited = set(path)
        # Ağırlıklar (Gecikme, Güvenilirlik, Kaynak) değişkenlere atanır.
        w_d, w_r, w_res = weights

        # Hedefe ulaşılmadığı sürece döngü devam eder.
        while current_node != end_node:
            # Mevcut düğümün tüm komşuları alınır.
            neighbors = list(graph.neighbors(current_node))
            # Geçerli (gidilebilir) komşuları tutacak liste.
            valid_neighbors = []
            
            # Tüm komşular kontrol edilir.
            for n in neighbors:
                # Eğer komşu daha önce ziyaret edildiyse atla (Döngü önleme).
                if n in visited: continue
                # Kenarın bant genişliği değeri alınır.
                edge_bw = graph[current_node][n].get('bandwidth', 0)
                # Eğer bant genişliği minimum gereksinimi karşılıyorsa:
                if edge_bw >= min_bw:
                    # Komşuyu geçerli listesine ekle.
                    valid_neighbors.append(n)

            # ----------------------------------------------------------------
            # ÇIKMAZ SOKAK (DEAD END) KONTROLÜ
            # ----------------------------------------------------------------
            # Eğer gidilecek hiçbir geçerli komşu yoksa:
            if not valid_neighbors:
                # Başarısızlık (None) döndür ve işlemi bitir.
                return None 

            # ----------------------------------------------------------------
            # SEÇİM OLASILIKLARININ HESAPLANMASI
            # ----------------------------------------------------------------
            # Her komşu için seçim olasılığını tutacak liste.
            probabilities = []
            # Olasılıkların toplamı (Payda).
            denominator = 0.0

            # Her geçerli komşu için olasılık hesabı yapılır.
            for neighbor in valid_neighbors:
                # Tau: Feromon miktarı (Geçmiş tecrübe).
                # Eğer kenarda feromon yoksa varsayılan 1.0 alınır.
                tau = pheromones.get((current_node, neighbor), 1.0)
                
                # Kenar verileri graf'tan çekilir.
                edge_data = graph[current_node][neighbor]
                d = edge_data.get('delay', 1.0)        # Gecikme
                r = edge_data.get('reliability', 0.99) # Güvenilirlik
                bw = edge_data.get('bandwidth', 100)   # Bant Genişliği
                
                # Eta: Sezgisel çekicilik (Maliyetin tersi - Görünürlük).
                # Güvenilirlik logaritmik maliyete çevrilir.
                if r <= 0: r = 0.0001
                r_cost = -math.log(r)
                # Kaynak maliyeti hesaplanır (1000/BW).
                res_cost = 1000.0/bw if bw > 0 else 1000.0
                
                # Yerel maliyet (Local Cost) hesaplanır.
                local_cost = (w_d * d) + (w_r * r_cost) + (w_res * res_cost)
                # Eta = 1 / Maliyet (Maliyet ne kadar azsa çekicilik o kadar fazla).
                eta = 1.0 / local_cost if local_cost > 0 else 1.0
                
                # Olasılık Formülü: P = (tau^alpha) * (eta^beta)
                # alpha: Feromonun etkisi, beta: Sezgisel bilginin etkisi.
                prob = (tau ** alpha) * (eta ** beta)
                
                # Hesaplanan olasılık listeye eklenir.
                probabilities.append(prob)
                # Toplam olasılığa eklenir.
                denominator += prob

            # Eğer toplam olasılık 0 ise (Matematiksel hata veya imkansız durum):
            if denominator == 0: return None
            
            # ----------------------------------------------------------------
            # ROULETTE WHEEL SELECTION (BİR SONRAKİ DÜĞÜMÜ SEÇME)
            # ----------------------------------------------------------------
            # Olasılıklar normalize edilir (Toplamları 1 olacak şekilde).
            probabilities = [p / denominator for p in probabilities]
            
            # random.choices ile ağırlıklı rastgele seçim yapılır.
            # Seçilen komşu 'next_node' olur.
            next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
            
            # Seçilen düğüm yola eklenir.
            path.append(next_node)
            # Seçilen düğüm ziyaret edilenler kümesine eklenir.
            visited.add(next_node)
            # Karıncanın konumu güncellenir.
            current_node = next_node
            
            # Sonsuz döngü koruması (Çok uzun yolları engellemek için).
            if len(path) > 250: return None 

        # Hedefe ulaşıldığında oluşturulan yol döndürülür.
        return path


# =================================================================================================
# 3. GA ÇÖZÜCÜ (GENETİK ALGORİTMA)
# =================================================================================================
class GASolver:
    """
    Genetic Algorithm (GA) Algoritması
    Popülasyon tabanlı evrimsel yaklaşım.
    """
    @staticmethod
    def solve(graph, source, target, weights, min_bw, population_size=40, generations=30):
        # Algoritma başlangıç zamanı kaydedilir.
        start_time = time.time()
        
        # 1. BAŞLANGIÇ POPÜLASYONU ÜRETİMİ
        # Popülasyonu tutacak liste oluşturulur.
        population = []
        attempts = 0 # Sonsuz döngüden kaçınmak için deneme sayacı.
        
        # Hedeflenen popülasyon boyutuna ulaşana kadar rastgele yollar üretilir.
        # Maksimum deneme sayısı: Popülasyon boyutu * 5
        while len(population) < population_size and attempts < population_size * 5:
            # Rastgele bir yol üretmek için yardımcı fonksiyon çağrılır.
            path = GASolver._random_path(graph, source, target, min_bw)
            
            # Eğer geçerli bir yol bulunursa:
            if path:
                # Yolun maliyeti hesaplanır.
                cost = calculate_total_cost(graph, path, weights)
                # Yol ve maliyeti popülasyona eklenir.
                population.append((path, cost))
            attempts += 1
            
        # Eğer hiç başlangıç yolu bulunamazsa (Popülasyon boşsa):
        if not population:
            # Başarısızlık döndürülür.
            return None, float('inf'), (time.time() - start_time) * 1000

        # Global en iyi yol ve maliyet başlatılır.
        global_best_path = None
        global_best_cost = float('inf')

        # 2. EVRİM DÖNGÜSÜ (GENERATIONS)
        for gen in range(generations):
            # Popülasyonu maliyete göre (küçükten büyüğe) sırala.
            # En iyi (en düşük maliyetli) bireyler listenin başında olur.
            population.sort(key=lambda x: x[1])
            
            # En iyi birey (popülasyonun birincisi) kontrol edilir.
            if population[0][1] < global_best_cost:
                # Global en iyi güncellenir.
                global_best_path = population[0][0]
                global_best_cost = population[0][1]

            # ELİTİZM (Seçkincilik):
            # En iyi performansı gösteren %10'luk dilim, hiçbir değişikliğe uğramadan
            # bir sonraki nesile doğrudan aktarılır. Bu, iyi çözümlerin kaybolmasını önler.
            new_population = population[:int(population_size * 0.1)]

            # Yeni nesil popülasyon boyutu tamamlanana kadar döngü devam eder.
            while len(new_population) < population_size:
                # SEÇİM (Selection): Turnuva yöntemiyle iki ebeveyn seçilir.
                parent1 = GASolver._tournament_selection(population)
                parent2 = GASolver._tournament_selection(population)
                
                # ÇAPRAZLAMA (Crossover):
                # Ebeveynlerin genleri (yol parçaları) birleştirilerek çocuk oluşturulur.
                child_path = GASolver._crossover(parent1[0], parent2[0])
                
                # MUTASYON (Mutation):
                # Çeşitliliği korumak için %20 ihtimalle rastgele değişim uygulanır.
                if random.random() < 0.2: 
                    child_path = GASolver._mutate(graph, child_path, min_bw)
                
                # Oluşturulan çocuk geçerli ise:
                if child_path:
                    # Çocuğun maliyeti hesaplanır.
                    cost = calculate_total_cost(graph, child_path, weights)
                    # Yeni popülasyona eklenir.
                    new_population.append((child_path, cost))
            
            # Eski popülasyon, yeni nesil ile değiştirilir.
            population = new_population

        # Toplam geçen süre hesaplanır.
        elapsed = (time.time() - start_time) * 1000
        # En iyi çözüm ve süre döndürülür.
        return global_best_path, global_best_cost, elapsed

    @staticmethod
    def _random_path(graph, source, target, min_bw):
        """Kaynaktan hedefe rastgele geçerli bir yol oluşturur."""
        path = [source]
        visited = set([source])
        curr = source
        
        while curr != target:
            # Geçerli komşuları bul:
            # 1. Ziyaret edilmemiş olmalı (path içinde olmamalı)
            # 2. Bant genişliği gereksinimini karşılamalı
            neighbors = [n for n in graph.neighbors(curr) 
                         if n not in visited and graph[curr][n].get('bandwidth', 0) >= min_bw]
            
            # Eğer geçerli komşu yoksa (çıkmaz sokak):
            if not neighbors: return None
            
            # Rastgele bir komşu seç.
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            curr = next_node
            
            # Çok uzun yolları engellemek için sınır.
            if len(path) > 250: return None
            
        return path

    @staticmethod
    def _tournament_selection(population):
        """Turnuva seçimi: Rastgele k birey seçilir, en iyisi döndürülür."""
        k = 3 # Turnuva boyutu
        # Popülasyondan rastgele k aday seç.
        candidates = random.sample(population, k)
        # Maliyeti en düşük (en iyi) olanı döndür.
        return min(candidates, key=lambda x: x[1])

    @staticmethod
    def _crossover(parent1, parent2):
        """İki ebeveyn yolu birleştirerek yeni bir yol (çocuk) oluşturur."""
        # İki yol arasındaki ortak düğümleri bul (Başlangıç ve bitiş hariç).
        # Ortak düğümler, yolları kesip birleştirebileceğimiz kavşak noktalarıdır.
        common_nodes = list(set(parent1[1:-1]) & set(parent2[1:-1]))
        
        # Eğer ortak ara düğüm yoksa, crossover yapılamaz.
        # Rastgele biri (parent1) olduğu gibi döndürülür.
        if not common_nodes:
            return parent1 

        # Ortak düğümlerden rastgele bir kesim noktası seçilir.
        cut_node = random.choice(common_nodes)
        
        # Kesim noktasının her iki ebeveyndeki indeksleri bulunur.
        idx1 = parent1.index(cut_node)
        idx2 = parent2.index(cut_node)
        
        # Parent1'in başı ile Parent2'nin sonu birleştirilir.
        # Bu, genetik çeşitliliği sağlayan yeni bir rota oluşturur.
        new_path = parent1[:idx1] + parent2[idx2:]
        
        # Geçerlilik Kontrolü:
        # Oluşan yeni yolda tekrar eden düğüm var mı? (Döngü kontrolü)
        if len(new_path) != len(set(new_path)):
            return parent1 # Geçersizse ebeveyni döndür.
            
        return new_path

    @staticmethod
    def _mutate(graph, path, min_bw):
        """Bir yolda rastgele değişiklik (mutasyon) yapar."""
        # Çok kısa yollarda mutasyon yapılamaz.
        if len(path) < 3: return path
        
        # Yol üzerinde rastgele bir kopma noktası seçilir.
        idx = random.randint(1, len(path)-2)
        # Mutasyon noktasına kadar olan kısım alınır.
        partial_path = path[:idx+1]
        
        # Hedef düğüm alınır.
        target = path[-1]
        
        # Kopma noktasından itibaren hedefe giden YENİ rastgele bir yol aranır.
        remaining = GASolver._random_path_from_partial(graph, partial_path, target, min_bw)
        
        # Eğer geçerli bir yol bulunursa döndürülür.
        if remaining:
            return remaining
        # Bulunamazsa orijinal yol korunur.
        return path

    @staticmethod
    def _random_path_from_partial(graph, current_path, target, min_bw):
        """Kısmi bir yoldan başlayıp hedefe giden rastgele yol tamamlar."""
        path = list(current_path)
        visited = set(path)
        curr = path[-1]
        
        while curr != target:
            # Geçerli komşuları bul (Ziyaret edilmemiş ve BW yeterli).
            neighbors = [n for n in graph.neighbors(curr) 
                         if n not in visited and graph[curr][n].get('bandwidth', 0) >= min_bw]
            
            if not neighbors: return None
            
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            curr = next_node
            
            if len(path) > 250: return None
            
        return path

# =================================================================================================
# 4. ARAYÜZ (GUI) - PyQt6
# =================================================================================================
# Ana uygulama penceresi ve sekmelerin yönetimi.
# Bu sınıf, kullanıcı arayüzünü oluşturur, grafikleri çizer ve algoritmaları tetikler.

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - QoS Routing Project (ACO & GA)")
        self.resize(1300, 850)

        # Başlangıçta grafiği bir kez yükle
        self.G = create_graph_from_csv()
        self.node_count = self.G.number_of_nodes()
        
        # Düğümlerin konumlarını belirle (Görselleştirme için)
        if self.node_count > 0:
            # Spring layout, düğümleri dengeli bir şekilde dağıtır
            self.pos = nx.spring_layout(self.G, seed=42) 
        else:
            self.pos = {}

        # Sekmeli Yapı (TabWidget)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Tekli Analiz (Kullanıcı manuel parametre girer)
        self.tab1 = QWidget()
        self.init_single_run_tab()
        self.tabs.addTab(self.tab1, "🔍 Analiz (Tekli Çalıştırma)")

        # Tab 2: Toplu Test (Dosyadan okuyup istatistik çıkarır)
        self.tab2 = QWidget()
        self.init_batch_test_tab()
        self.tabs.addTab(self.tab2, "📊 Toplu Test (Kıyaslama)")

    def init_single_run_tab(self):
        """Tekli çalıştırma sekmesinin arayüz elemanlarını oluşturur."""
        layout = QHBoxLayout(self.tab1)
        
        # --- Sol Panel (Ayarlar) ---
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        l_layout = QVBoxLayout(left_panel)

        l_layout.addWidget(QLabel("<h2>Algoritma Ayarları</h2>"))
        
        l_layout.addWidget(QLabel("Algoritma Seç:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["ACO - Karınca Kolonisi", "GA - Genetik Algoritma"])
        l_layout.addWidget(self.combo_algo)

        l_layout.addWidget(QLabel("Kaynak (Source):"))
        self.spin_s = QSpinBox(); self.spin_s.setRange(0, 500); self.spin_s.setValue(0)
        l_layout.addWidget(self.spin_s)

        l_layout.addWidget(QLabel("Hedef (Target):"))
        self.spin_d = QSpinBox(); self.spin_d.setRange(0, 500); self.spin_d.setValue(10)
        l_layout.addWidget(self.spin_d)

        l_layout.addWidget(QLabel("Min Bant Genişliği:"))
        self.spin_bw = QSpinBox(); self.spin_bw.setRange(0, 10000); self.spin_bw.setValue(50)
        l_layout.addWidget(self.spin_bw)

        l_layout.addWidget(QLabel("<h3>Ağırlıklar (Weights)</h3>"))
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setValue(0.33); self.spin_wd.setSingleStep(0.1)
        l_layout.addWidget(QLabel("Gecikme (Delay):")); l_layout.addWidget(self.spin_wd)
        
        self.spin_wr = QDoubleSpinBox(); self.spin_wr.setValue(0.33); self.spin_wr.setSingleStep(0.1)
        l_layout.addWidget(QLabel("Güvenilirlik (Reliability):")); l_layout.addWidget(self.spin_wr)

        self.spin_wres = QDoubleSpinBox(); self.spin_wres.setValue(0.34); self.spin_wres.setSingleStep(0.1)
        l_layout.addWidget(QLabel("Kaynak (Resource):")); l_layout.addWidget(self.spin_wres)

        self.btn_run = QPushButton("🚀 Hesapla")
        self.btn_run.clicked.connect(self.run_single)
        l_layout.addWidget(self.btn_run)

        # Sonuçların yazılacağı metin kutusu
        self.txt_output = QTextEdit(); self.txt_output.setReadOnly(True)
        l_layout.addWidget(self.txt_output)
        
        layout.addWidget(left_panel)

        # --- Sağ Panel (Grafik) ---
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        if self.node_count > 0: self.plot_graph([], 0, 0)

    def run_single(self):
        """Tekli analiz butonuna tıklandığında çalışır."""
        # Arayüzden kullanıcı parametrelerini al
        S = self.spin_s.value()  # Kaynak Düğüm
        D = self.spin_d.value()  # Hedef Düğüm
        B = self.spin_bw.value() # Minimum Bant Genişliği
        
        # Ağırlıklar (Gecikme, Güvenilirlik, Kaynak)
        weights = (self.spin_wd.value(), self.spin_wr.value(), self.spin_wres.value())
        
        # Seçilen algoritmayı belirle
        algo_choice = self.combo_algo.currentText()

        # Kullanıcıya bilgi ver (Arayüz donmasını önlemek için update)
        self.txt_output.setText(f"{algo_choice} Çalışıyor...")
        QApplication.processEvents() # Arayüzü tazelemeye zorla

        # Seçime göre ilgili algoritmayı çalıştır
        if "ACO" in algo_choice:
            path, cost, time_ms = ACOSolver.solve(self.G, S, D, weights, min_bw=B)
        else:
            path, cost, time_ms = GASolver.solve(self.G, S, D, weights, min_bw=B)

        # Eğer başarılı bir yol bulunduysa:
        if path:
            # Bulunan yol için ayrıntılı QoS metriklerini hesapla
            delay, rel_sum, res_sum, true_rel = compute_metrics(self.G, path)
            
            # Sonuç mesajını oluştur
            msg = (f"✅ {algo_choice} Sonuç:\n"
                   f"Süre: {time_ms:.2f} ms\n"
                   f"Maliyet (Fitness): {cost:.4f}\n"
                   f"----------------------\n"
                   f"Yol Uzunluğu: {len(path)} düğüm\n"
                   f"Yol: {path}\n"
                   f"----------------------\n"
                   f"Toplam Gecikme: {delay:.2f} ms\n"
                   f"Toplam Güvenilirlik: {true_rel:.4f}")
            
            # Mesajı ekrana yazdır
            self.txt_output.setText(msg)
            # Yolu grafik üzerinde çiz
            self.plot_graph(path, S, D)
        else:
            # Başarısızlık durumunda bilgi ver
            self.txt_output.setText("❌ Yol Bulunamadı (Geçersiz parametreler veya izole düğüm)")
            self.plot_graph(None, S, D)

    def plot_graph(self, path, S, D):
        """Grafiği ve (varsa) bulunan yolu çizer."""
        self.figure.clear()
        
        # Matplotlib ekseni oluştur
        ax = self.figure.add_subplot(111)
        
        if path:
            # Yol kenarlarını oluştur (Zip ile ardışık düğümleri eşleştir)
            path_edges = list(zip(path, path[1:]))
           
            # 1. Tüm düğümleri çiz (Arkaplan - Gri)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_size=20, node_color='#e0e0e0', alpha=0.3)
            # 2. Tüm kenarları çiz (Arkaplan - Gri)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.1, edge_color='#cccccc')
            
            # 3. Bulunan yolu vurgula
            # Yol üzerindeki düğümler (Turuncu)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=path, node_color='orange', node_size=80)
            # Yol üzerindeki kenarlar (Kırmızı ve Kalın)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=path_edges, edge_color='red', width=2)
            
            # 4. Kaynak ve Hedef düğümleri belirginleştir
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[S], node_color='green', node_size=150, label='Source')
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[D], node_color='blue', node_size=150, label='Dest')
            
            # Başlık ekle
            ax.set_title(f"Rota: {S} -> {D}")
        else:
            # Yol yoksa sadece basit grafiği çiz
            nx.draw(self.G, self.pos, ax=ax, node_size=30, node_color='lightblue', with_labels=False, alpha=0.5)
            ax.set_title("Ağ Topolojisi")
            
        # Eksenleri kapat (Daha temiz görünüm için)
        ax.axis('off')
        
        # Çizimi güncelle
        self.canvas.draw()

    def init_batch_test_tab(self):
        """Toplu test sekmesinin arayüz elemanlarını oluşturur."""
        layout = QVBoxLayout(self.tab2)
        
        # Üst buton paneli
        top = QHBoxLayout()
        self.btn_batch = QPushButton("🧪 Toplu Testi Başlat (ACO vs GA)"); 
        self.btn_batch.clicked.connect(self.run_batch)
        top.addWidget(self.btn_batch)
        
        self.btn_export = QPushButton("💾 CSV Kaydet"); 
        self.btn_export.clicked.connect(self.export_csv)
        top.addWidget(self.btn_export)
        layout.addLayout(top)
        
        # İlerleme çubuğu
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Sonuç tablosu
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["S->D (Talep)", "Algoritma", "Başarı %", "Ort. Maliyet", "Ort. Süre", "En İyi", "En Kötü"])
        # Sütunları pencereye sığacak şekilde genişlet
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

    def run_batch(self):
        """Toplu testi başlatır. CSV'deki tüm senaryoları çalıştırır."""
        demands = []
        try:
            # Talep dosyasını oku
            with open(DEMAND_FILE, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader) # Başlığı atla
                for row in reader:
                    if len(row) >= 3:
                        # (Kaynak, Hedef, Bant Genişliği)
                        demands.append((int(row[0]), int(row[1]), float(row[2].replace(',','.'))))
        except: 
            QMessageBox.warning(self, "Hata", "DemandData.csv okunamadı!")
            return

        # Tabloyu temizle
        self.table.setRowCount(0)
        # İlerleme çubuğunu ayarla (Her talep için 2 algoritma çalışacak)
        self.progress.setMaximum(len(demands) * 2) 
        
        weights = (0.33, 0.33, 0.34) # Sabit ağırlıklar
        repeats = 5  # Her senaryo için tekrar sayısı (İstatistiksel güvenilirlik için)

        prog_val = 0
        # Tüm talepler üzerinde döngü
        for S, D, B in demands:
            # Her iki algoritmayı da dene
            for algo_name in ["ACO", "GA"]:
                costs = []
                times = []
                success_count = 0
                
                # İstatistik toplamak için 'repeats' kadar çalıştır
                for _ in range(repeats):
                    if algo_name == "ACO":
                        # Daha hızlı sonuç için iterasyon/karınca sayısı düşürüldü
                        path, cost, t = ACOSolver.solve(self.G, S, D, weights, min_bw=B, num_ants=15, num_iterations=15)
                    else:
                        # Daha hızlı sonuç için popülasyon/jenerasyon düşürüldü
                        path, cost, t = GASolver.solve(self.G, S, D, weights, min_bw=B, population_size=20, generations=20)
                    
                    if path:
                        success_count += 1
                        costs.append(cost)
                        times.append(t)
                
                # Sonuçları tabloya ekle
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(f"{S}->{D} ({B})"))
                self.table.setItem(row, 1, QTableWidgetItem(algo_name))
                
                # Başarı oranını hesapla
                succ_rate = (success_count / repeats) * 100
                self.table.setItem(row, 2, QTableWidgetItem(f"%{succ_rate:.0f}"))
                
                if costs:
                    # İstatistiksel metrikleri hesapla
                    avg_cost = sum(costs) / len(costs)
                    avg_time = sum(times) / len(times)
                    best_c = min(costs)
                    worst_c = max(costs)
                    
                    self.table.setItem(row, 3, QTableWidgetItem(f"{avg_cost:.2f}"))
                    self.table.setItem(row, 4, QTableWidgetItem(f"{avg_time:.1f}"))
                    self.table.setItem(row, 5, QTableWidgetItem(f"{best_c:.2f}"))
                    self.table.setItem(row, 6, QTableWidgetItem(f"{worst_c:.2f}"))
                else:
                    # Sonuç yoksa tire koy
                    for c in range(3, 7): self.table.setItem(row, c, QTableWidgetItem("-"))
                
                # Arayüzü güncelle
                prog_val += 1
                self.progress.setValue(prog_val)
                QApplication.processEvents()

    def export_csv(self):
        """Sonuç tablosunu CSV dosyasına aktarır."""
        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "", "CSV(*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                # Başlıkları yaz
                headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
                writer.writerow(headers)
                # Satırları yaz
                for r in range(self.table.rowCount()):
                    writer.writerow([self.table.item(r,c).text() for c in range(self.table.columnCount())])

if __name__ == "__main__":
    # PyQt uygulamasını başlat
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # Olay döngüsünü başlat
    sys.exit(app.exec())