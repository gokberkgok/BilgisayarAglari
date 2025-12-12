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
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QProgressBar
)

# ==========================================
# 1. ARKA PLAN MANTIĞI (NETWORK & SARSA)
# ==========================================

# Script ile aynı dizindeki CSV dosyalarını bul (taşınabilirlik için)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NODE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_NodeData.csv")
EDGE_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_EdgeData.csv")
DEMAND_FILE = os.path.join(BASE_DIR, "BSM307_317_Guz2025_TermProject_DemandData.csv")


def create_graph_from_csv():
    """CSV Dosyalarından Graf Oluşturur"""
    G = nx.Graph()
    
    # 1. Düğümleri Yükle
    # node_id, s_ms, r_node
    try:
        with open(NODE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Başlığı atla
            
            for row in reader:
                if not row or len(row) < 3: continue
                # Boşlukları temizle ve dönüştür
                row = [x.strip() for x in row]
                
                try:
                    node_id = int(row[0])
                    processing_delay = float(row[1])
                    reliability = float(row[2])
                    
                    G.add_node(node_id, processing_delay=processing_delay, reliability=reliability)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"HATA: {NODE_FILE} bulunamadı! Lütfen dosyayı kontrol edin.")
        raise  # CSV yoksa program dursun, sessizce random graf oluşturma

    # 2. Kenarları (Linkleri) Yükle
    # src, dst, capacity_mbps, delay_ms, r_link
    try:
        with open(EDGE_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Başlığı atla
            
            for row in reader:
                if not row or len(row) < 5: continue
                row = [x.strip() for x in row]
                
                try:
                    u = int(row[0])
                    v = int(row[1])
                    bandwidth = float(row[2])
                    delay = float(row[3])
                    reliability = float(row[4])
                    
                    # NetworkX graph'a ekle
                    G.add_edge(u, v, bandwidth=bandwidth, delay=delay, reliability=reliability)
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"HATA: {EDGE_FILE} bulunamadı!")
    
    # Bağlantılılık Kontrolü (Rapor gereksinimi: graf bağlı olmalı)
    if G.number_of_nodes() > 0:
        if not nx.is_connected(G):
            print("⚠ UYARI: CSV'den yüklenen graf bağlı değil!")
            print("  En büyük bağlı bileşen kullanılıyor...")
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            print(f"  ✓ Yeni graf boyutu: {G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar")
        else:
            print(f"✓ Graf bağlı: {G.number_of_nodes()} düğüm, {G.number_of_edges()} kenar")
        
    return G

def compute_metrics(G, path):
    """Rapor Bölüm 3: Metrik Hesaplamaları"""
    delay = 0
    rel_cost = 0
    res_cost = 0
    true_rel = 1.0

    if not path:
        return 0, 0, 0, 0

    # Kenar maliyetleri
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        
        # Eğer kenar yoksa (hata durumu)
        if not G.has_edge(u, v):
            return float('inf'), float('inf'), float('inf'), 0

        edge_data = G[u][v]
        d = edge_data.get("delay", 10)
        r = edge_data.get("reliability", 0.9)
        bw = edge_data.get("bandwidth", 100)

        delay += d
        # log(0) hatasını önlemek için r'yi kontrol et
        if r <= 0: r = 0.0001
        rel_cost += -math.log(r)
        
        if bw <= 0: bw = 0.1
        res_cost += (1000 / bw)
        true_rel *= r

    # Düğüm maliyetleri
    for i, node in enumerate(path):
        node_data = G.nodes[node]
        r = node_data.get("reliability", 0.99)
        proc_delay = node_data.get("processing_delay", 0)

        if r <= 0: r = 0.0001
        rel_cost += -math.log(r)
        true_rel *= r
        
        # Uç düğümler hariç işlem gecikmesi ekle
        if i != 0 and i != len(path) - 1: 
            delay += proc_delay
            
    return delay, rel_cost, res_cost, true_rel

def get_valid_neighbors(G, node, min_bw):
    """Bant Genişliği (B) Filtresi"""
    valid = []
    # Graph yönlendirmesiz ise G.neighbors(node) yeterli
    for nbr in G.neighbors(node):
        bw = G[node][nbr].get("bandwidth", 0)
        if bw >= min_bw:
            valid.append(nbr)
    return valid


def sarsa_route(G, S, D, Wd, Wr, Wres, min_bw, episodes=2000):
    """SARSA Algoritması"""
    start_time = time.time()
    
    # Edge Case: Kaynak == Hedef
    if S == D:
        elapsed = (time.time() - start_time) * 1000
        return [S], 0.0, elapsed
    
    Q = defaultdict(lambda: 0.0) # (state, action) -> value
    
    # Q tablosu için helper
    def get_Q(s, a):
        return Q[(s, a)]
    
    def set_Q(s, a, val):
        Q[(s, a)] = val

    best_path = None
    best_cost = float("inf")
    
    # Hiperparametreler
    epsilon_start = 0.3
    epsilon_end = 0.05
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    alpha = 0.1
    gamma = 0.95
    max_steps = 100 # Sonsuz döngü önleyici

    for episode in range(episodes):
        # Epsilon decay (zamanla exploration azalır, exploitation artar)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        state = S
        path = [state]
        
        valid_neighbors = get_valid_neighbors(G, state, min_bw)
        if not valid_neighbors: continue  # Sonraki episode'a geç
        
        # İlk eylem seçimi (Epsilon-Greedy)
        if random.random() < epsilon:
            action = random.choice(valid_neighbors)
        else:
            # Greedy: Max Q değerine sahip komşu
            # Başlangıçta hepsi 0 olduğu için yine randoma döner, sorun yok
            action = max(valid_neighbors, key=lambda n: get_Q(state, n))

        steps = 0
        while state != D and steps < max_steps:
            steps += 1
            next_state = action
            
            # Action node D ise döngü biter
            if next_state == D:
                path.append(next_state)
                # Ödül hesapla
                d, r_cost, res, _ = compute_metrics(G, path)
                total_cost = (Wd * d) + (Wr * r_cost) + (Wres * res)
                
                reward = 1000.0 / total_cost if total_cost > 0 else 1000.0
                
                # Q Update (Terminal)
                # Q(S, A) = Q(S, A) + alpha * (R - Q(S, A))
                current_q = get_Q(state, action)
                set_Q(state, action, current_q + alpha * (reward - current_q))

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = list(path)
                
                # Yeni episoda geç
                break

            # Ara düğümdeyiz
            next_valid_neighbors = get_valid_neighbors(G, next_state, min_bw)
            
            if not next_valid_neighbors:
                # Dead end (çıkmaz sokak)
                reward = -100
                current_q = get_Q(state, action)
                set_Q(state, action, current_q + alpha * (reward - current_q))
                break

            # Sonraki eylemi seç (SARSA -> On-policy)
            if random.random() < epsilon:
                next_action = random.choice(next_valid_neighbors)
            else:
                next_action = max(next_valid_neighbors, key=lambda n: get_Q(next_state, n))

            # Q Update (Non-terminal)
            # Q(S, A) = Q(S, A) + alpha * (R + gamma * Q(S', A') - Q(S, A))
            reward = -1 # Hop cezası
            
            q_sa = get_Q(state, action)
            q_next = get_Q(next_state, next_action)
            
            new_val = q_sa + alpha * (reward + gamma * q_next - q_sa)
            set_Q(state, action, new_val)

            state = next_state
            action = next_action
            path.append(state)
    
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time) * 1000
    
    # Best path varsa ve cost mantıklıysa
    if best_cost == float("inf"):
        return None, 0, elapsed_time_ms
        
    return best_path, best_cost, elapsed_time_ms

# ==========================================
# 2. ARAYÜZ (PYQT6) - GELİŞTİRİLMİŞ TAB YAPISI
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - QoS SARSA Projesi (CSV Verileri)")
        self.resize(1300, 900)

        # CSV Dosyalarından Grafi Yükle
        self.G = create_graph_from_csv()
        self.node_count = self.G.number_of_nodes()
        
        # Konumlandırma (Spring layout biraz yavaş olabilir, ama görsel için gerekli)
        # Sabit seed ile her seferinde aynı şekil
        self.pos = nx.spring_layout(self.G, seed=42)

        # Ana Tab Widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Tekli Çalıştırma & Görselleştirme
        self.tab1 = QWidget()
        self.init_single_run_tab()
        self.tabs.addTab(self.tab1, "🔍 Tekli Analiz & Görselleştirme")

        # Tab 2: Toplu Deney & İstatistik
        self.tab2 = QWidget()
        self.init_batch_test_tab()
        self.tabs.addTab(self.tab2, "📊 Toplu Deney (DemandData.csv)")

    # ---------------------------------------------------
    # TAB 1: TEKLİ ÇALIŞTIRMA
    # ---------------------------------------------------
    def init_single_run_tab(self):
        layout = QHBoxLayout(self.tab1)

        # Sol Panel
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_panel.setStyleSheet("background-color: #f8f9fa; border-right: 1px solid #ddd;")
        l_layout = QVBoxLayout(left_panel)

        l_layout.addWidget(QLabel("<h2>Ayarlar</h2>"))
        l_layout.addWidget(QLabel(f"Toplam Düğüm Sayısı: {self.node_count}"))
        
        l_layout.addWidget(QLabel("Kaynak (S):"))
        self.spin_s = QSpinBox(); self.spin_s.setRange(0, self.node_count * 2); self.spin_s.setValue(0)
        l_layout.addWidget(self.spin_s)

        l_layout.addWidget(QLabel("Hedef (D):"))
        self.spin_d = QSpinBox(); self.spin_d.setRange(0, self.node_count * 2); self.spin_d.setValue(10)
        l_layout.addWidget(self.spin_d)

        l_layout.addWidget(QLabel("Min. Bant Genişliği (B):"))
        self.spin_bw = QSpinBox(); self.spin_bw.setRange(0, 10000); self.spin_bw.setSuffix(" Mbps")
        l_layout.addWidget(self.spin_bw)

        l_layout.addWidget(QLabel("<h3>Ağırlıklar</h3>"))
        self.spin_wd = QDoubleSpinBox(); self.spin_wd.setValue(0.33); self.spin_wd.setSingleStep(0.1)
        l_layout.addWidget(QLabel("W_Delay:")); l_layout.addWidget(self.spin_wd)
        
        self.spin_wr = QDoubleSpinBox(); self.spin_wr.setValue(0.33); self.spin_wr.setSingleStep(0.1)
        l_layout.addWidget(QLabel("W_Rel:")); l_layout.addWidget(self.spin_wr)

        self.spin_wres = QDoubleSpinBox(); self.spin_wres.setValue(0.34); self.spin_wres.setSingleStep(0.1)
        l_layout.addWidget(QLabel("W_Res:")); l_layout.addWidget(self.spin_wres)

        l_layout.addWidget(QLabel("<h3>SARSA Parametreleri</h3>"))
        l_layout.addWidget(QLabel("Episode Sayısı:"))
        self.spin_episodes = QSpinBox(); self.spin_episodes.setValue(2000); self.spin_episodes.setRange(100, 10000)
        l_layout.addWidget(self.spin_episodes)

        self.btn_run = QPushButton("🚀 Çalıştır")
        self.btn_run.setStyleSheet("background-color: #007bff; color: white; padding: 10px; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_single)
        l_layout.addWidget(self.btn_run)

        self.txt_output = QTextEdit(); self.txt_output.setReadOnly(True)
        l_layout.addWidget(self.txt_output)
        
        layout.addWidget(left_panel)

        # Sağ Panel (Matplotlib)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # İlk çizim
        if self.node_count > 0:
             # İlk 2 node'u default çizdirelim
             s_def = list(self.G.nodes())[0]
             d_def = list(self.G.nodes())[-1]
             self.plot_graph(None, s_def, d_def)

    def run_single(self):
        S = self.spin_s.value()
        D = self.spin_d.value()
        B = self.spin_bw.value()
        Wd, Wr, Wres = self.spin_wd.value(), self.spin_wr.value(), self.spin_wres.value()

        if S not in self.G or D not in self.G:
            self.txt_output.setText(f"HATA: {S} veya {D} düğümü grafikte yok.")
            return

        self.txt_output.setText("Hesaplanıyor...")
        QApplication.processEvents()

        episodes = self.spin_episodes.value()
        path, cost, time_ms = sarsa_route(self.G, S, D, Wd, Wr, Wres, min_bw=B, episodes=episodes)


        if path:
            delay, rel_cost, res_cost, true_rel = compute_metrics(self.G, path)
            
            # Bandwidth kısıtının sağlandığını doğrula
            bw_violations = []
            for i in range(len(path) - 1):
                edge_bw = self.G[path[i]][path[i+1]].get('bandwidth', 0)
                if edge_bw < B:
                    bw_violations.append(f"{path[i]}-{path[i+1]} (BW:{edge_bw:.0f}<{B})")
            
            msg = (f"✅ SONUÇ:\nSüre: {time_ms:.2f} ms\nMaliyet: {cost:.4f}\nHop: {len(path)-1}\n"
                   f"Path: {path}\n"
                   f"Toplam Gecikme: {delay:.2f} ms\nGenel Güvenilirlik: {true_rel:.4f}")
            
            if bw_violations:
                msg += f"\n\n⚠ UYARI: Bandwidth kısıtı ihlalleri:\n" + "\n".join(bw_violations)
            
            self.txt_output.setText(msg)
            self.plot_graph(path, S, D)
        else:
            self.txt_output.setText("❌ Yol Bulunamadı!\nBu bant genişliği veya topoloji kısıtlarından kaynaklanabilir.")
            self.plot_graph(None, S, D)

    def plot_graph(self, path, S, D):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Yol bulunduysa diğer node'ları sol, bulunamadıysa normal çiz
        if path:
            # Yoldaki node'lar dışındakiler çok soluk
            other_nodes = [n for n in self.G.nodes() if n not in path and n != S and n != D]
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=other_nodes, 
                                   node_size=15, node_color='lightgray', alpha=0.2)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.05, edge_color='gray')
        else:
            # Yol yoksa normal görünürlük
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_size=20, node_color='lightgray', alpha=0.4)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.1, edge_color='gray')
        
        # Kaynak ve Hedef Vurgula
        if S in self.G and D in self.G:
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[S, D], 
                                   node_color=['blue', 'red'], node_size=120)
            # S ve D için sadece ID göster
            labels_sd = {S: str(S), D: str(D)}
            nx.draw_networkx_labels(self.G, self.pos, labels_sd, ax=ax, 
                                   font_size=6, font_color='white', font_weight='bold')
        
        if path:
            # Yolu çiz
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=edges, 
                                   edge_color='green', width=3, alpha=0.8)
            
            # Yoldaki ara düğümler (S ve D hariç)
            path_nodes_mid = [n for n in path if n != S and n != D]
            if path_nodes_mid:
                nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=path_nodes_mid, 
                                       node_size=80, node_color='lime', alpha=0.9)
                
                # Ara düğümler için sadece ID göster
                labels_path = {node: str(node) for node in path_nodes_mid}
                nx.draw_networkx_labels(self.G, self.pos, labels_path, ax=ax, 
                                       font_size=5, font_color='black')
        
        ax.set_title(f"Topoloji (S={S} -> D={D})")
        ax.axis('off')
        self.canvas.draw()

    # ---------------------------------------------------
    # TAB 2: TOPLU DENEY MODÜLÜ
    # ---------------------------------------------------
    def init_batch_test_tab(self):
        layout = QVBoxLayout(self.tab2)

        # Üst Kontrol Paneli
        top_panel = QFrame()
        top_panel.setStyleSheet("background-color: #e9ecef; border-radius: 5px;")
        h_layout = QHBoxLayout(top_panel)

        h_layout.addWidget(QLabel("<b>DemandData.csv Kullanılarak Test Yapılacak</b>"))
        
        h_layout.addWidget(QLabel("Tekrar Sayısı (Her Talep İçin):"))
        self.spin_repeat_count = QSpinBox(); self.spin_repeat_count.setValue(5)
        h_layout.addWidget(self.spin_repeat_count)

        self.btn_start_batch = QPushButton("🧪 CSV İLE DENEYİ BAŞLAT")
        self.btn_start_batch.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.btn_start_batch.clicked.connect(self.run_batch_experiment)
        h_layout.addWidget(self.btn_start_batch)
        
        self.btn_export = QPushButton("💾 Sonuçları Kaydet")
        self.btn_export.clicked.connect(self.export_csv)
        h_layout.addWidget(self.btn_export)

        layout.addWidget(top_panel)

        # İlerleme Çubuğu
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Tablo
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Senaryo (ID)", "S -> D", "Talep (BW)", "Başarı Oranı", 
            "Ort. Maliyet", "Std. Sapma", "En İyi Cost", "En Kötü Cost", "Ort. Süre (ms)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

    def load_demands(self):
        """CSV'den talepleri oku"""
        demands = []
        try:
            with open(DEMAND_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # Header
                # src, dst, demand_mbps
                for row in reader:
                    if not row or len(row) < 3: continue
                    row = [x.strip() for x in row]
                    try:
                        s = int(row[0])
                        d = int(row[1])
                        bw = float(row[2])
                        demands.append((s, d, bw))
                    except:
                        continue
        except FileNotFoundError:
            print(f"Uyarı: {DEMAND_FILE} bulunamadı.")
            
        return demands

    def run_batch_experiment(self):
        demands = self.load_demands()
        
        if not demands:
            # Eğer dosya boşsa ya da okunamazsa basit bir uyarı mesajbox'ı olmadığı için print ya da status'a yazabiliriz
            # Burada tabloya tek satır uyarı ekleyelim
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem("DEMAND DATA ERROR"))
            return

        repeats = self.spin_repeat_count.value()
        Wd, Wr, Wres = 0.33, 0.33, 0.34 # Varsayılan ağırlıklar

        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(demands))

        # Deneyleri Çalıştır
        for i, (S, D, B) in enumerate(demands):
            costs = []
            times = []
            success_count = 0
            
            # Bu talep için 'repeats' kadar çalıştır (SARSA stokastik olduğu için sonuç değişebilir)
            for _ in range(repeats):
                path, cost, t = sarsa_route(self.G, S, D, Wd, Wr, Wres, B, episodes=1000)
                if path:
                    costs.append(cost)
                    times.append(t)
                    success_count += 1
            
            # İstatistik Hesapla
            if costs:
                avg_cost = np.mean(costs)
                std_dev = np.std(costs)
                best_cost = np.min(costs)
                worst_cost = np.max(costs)
                avg_time = np.mean(times)
            else:
                avg_cost = std_dev = best_cost = worst_cost = avg_time = 0

            success_rate = (success_count / repeats) * 100

            # Tabloya Ekle
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(i+1)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{S} -> {D}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{B:.0f} Mbps"))
            self.table.setItem(row, 3, QTableWidgetItem(f"%{success_rate:.0f}"))
            
            if success_count > 0:
                self.table.setItem(row, 4, QTableWidgetItem(f"{avg_cost:.4f}"))
                self.table.setItem(row, 5, QTableWidgetItem(f"{std_dev:.4f}"))
                self.table.setItem(row, 6, QTableWidgetItem(f"{best_cost:.4f}"))
                self.table.setItem(row, 7, QTableWidgetItem(f"{worst_cost:.4f}"))
                self.table.setItem(row, 8, QTableWidgetItem(f"{avg_time:.2f}"))
            else:
                self.table.setItem(row, 4, QTableWidgetItem("BAŞARISIZ"))
                for c in range(5, 9): self.table.setItem(row, c, QTableWidgetItem("-"))

            self.progress_bar.setValue(i + 1)
            QApplication.processEvents()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSV Kaydet", "", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
                writer.writerow(headers)
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
