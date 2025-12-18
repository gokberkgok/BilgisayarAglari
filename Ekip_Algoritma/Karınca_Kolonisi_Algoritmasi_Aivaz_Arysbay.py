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
    """Yol maliyeti hesaplama fonksiyonu"""
    total_delay = 0
    rel_log_sum = 0
    res_cost_sum = 0
    true_rel = 1.0

    if not path: return 0, 0, 0, 0

   
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
        true_rel *= r

    
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

# ==========================================
# 2. ACOSolver (GELİŞTİRİLMİŞ KARINCA KOLONİSİ)
# ==========================================

class ACOSolver:
    @staticmethod
    def solve(graph, source, target, weights, min_bw, num_ants=20, num_iterations=30):
        
        alpha = 1.0
        beta = 2.0
        evaporation_rate = 0.1
        Q = 100.0
        tau_min = 0.1  
        tau_max = 10.0 

        pheromones = {}
        for u, v in graph.edges():
            pheromones[(u, v)] = 1.0
            pheromones[(v, u)] = 1.0

        global_best_path = None
        global_best_cost = float('inf')

        start_time = time.time()

        for iteration in range(num_iterations):
            paths_in_iteration = []

            for ant in range(num_ants):
                path = ACOSolver._ant_walk(graph, source, target, pheromones, alpha, beta, min_bw, weights)
                
                if path:
                    cost = calculate_total_cost(graph, path, weights)
                    paths_in_iteration.append((path, cost))
                    
                    if cost < global_best_cost:
                        global_best_cost = cost
                        global_best_path = list(path)

           
            for key in pheromones:
                pheromones[key] *= (1.0 - evaporation_rate)
                
                if pheromones[key] < tau_min: pheromones[key] = tau_min

            
            for path, cost in paths_in_iteration:
                deposit = Q / cost if cost > 0 else Q
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    pheromones[(u, v)] = min(tau_max, pheromones[(u, v)] + deposit) 
                    pheromones[(v, u)] = min(tau_max, pheromones[(v, u)] + deposit)

            
            if global_best_path:
                deposit = (Q / global_best_cost) * 2.0 
                for i in range(len(global_best_path) - 1):
                    u, v = global_best_path[i], global_best_path[i+1]
                    pheromones[(u, v)] = min(tau_max, pheromones[(u, v)] + deposit)
                    pheromones[(v, u)] = min(tau_max, pheromones[(v, u)] + deposit)

        elapsed = (time.time() - start_time) * 1000
        return global_best_path, global_best_cost, elapsed

    @staticmethod
    def _ant_walk(graph, start_node, end_node, pheromones, alpha, beta, min_bw, weights):
        current_node = start_node
        path = [current_node]
        visited = set(path)
        w_d, w_r, w_res = weights

        while current_node != end_node:
            neighbors = list(graph.neighbors(current_node))
            valid_neighbors = []
            
            for n in neighbors:
                if n in visited: continue
                edge_bw = graph[current_node][n].get('bandwidth', 0)
                if edge_bw >= min_bw:
                    valid_neighbors.append(n)

            if not valid_neighbors:
                return None 

            probabilities = []
            denominator = 0.0

            for neighbor in valid_neighbors:
                tau = pheromones.get((current_node, neighbor), 1.0)
                
                
                edge_data = graph[current_node][neighbor]
                d = edge_data.get('delay', 1.0)
                r = edge_data.get('reliability', 0.99)
                bw = edge_data.get('bandwidth', 100)
                
                
                if r <= 0: r = 0.0001
                r_cost = -math.log(r)
                res_cost = 1000.0/bw if bw > 0 else 1000.0
                
                local_cost = (w_d * d) + (w_r * r_cost) + (w_res * res_cost)
                eta = 1.0 / local_cost if local_cost > 0 else 1.0
                
                prob = (tau ** alpha) * (eta ** beta)
                probabilities.append(prob)
                denominator += prob

            if denominator == 0: return None
            
            probabilities = [p / denominator for p in probabilities]
            
            
            next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
            
            if len(path) > 250: return None 

        return path

# ==========================================
# 3. GASolver (GENETİK ALGORİTMA) - Yeni [cite: 80]
# ==========================================

class GASolver:
    @staticmethod
    def solve(graph, source, target, weights, min_bw, population_size=40, generations=30):
        start_time = time.time()
        
        
        population = []
        attempts = 0
        while len(population) < population_size and attempts < population_size * 5:
            path = GASolver._random_path(graph, source, target, min_bw)
            if path:
                cost = calculate_total_cost(graph, path, weights)
                population.append((path, cost))
            attempts += 1
            
        if not population:
            return None, float('inf'), (time.time() - start_time) * 1000

        global_best_path = None
        global_best_cost = float('inf')

        for gen in range(generations):
            
            population.sort(key=lambda x: x[1])
            
            
            if population[0][1] < global_best_cost:
                global_best_path = population[0][0]
                global_best_cost = population[0][1]

           
            new_population = population[:int(population_size * 0.1)]

           
            while len(new_population) < population_size:
                parent1 = GASolver._tournament_selection(population)
                parent2 = GASolver._tournament_selection(population)
                
                
                child_path = GASolver._crossover(parent1[0], parent2[0])
                
                
                if random.random() < 0.2: # %20 Mutasyon şansı
                    child_path = GASolver._mutate(graph, child_path, min_bw)
                
                if child_path:
                    cost = calculate_total_cost(graph, child_path, weights)
                    new_population.append((child_path, cost))
            
            population = new_population

        elapsed = (time.time() - start_time) * 1000
        return global_best_path, global_best_cost, elapsed

    @staticmethod
    def _random_path(graph, source, target, min_bw):
        
        path = [source]
        visited = set([source])
        curr = source
        while curr != target:
            neighbors = [n for n in graph.neighbors(curr) 
                         if n not in visited and graph[curr][n].get('bandwidth', 0) >= min_bw]
            if not neighbors: return None
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            curr = next_node
            if len(path) > 250: return None
        return path

    @staticmethod
    def _tournament_selection(population):
        k = 3
        candidates = random.sample(population, k)
        return min(candidates, key=lambda x: x[1])

    @staticmethod
    def _crossover(parent1, parent2):
        
        common_nodes = list(set(parent1[1:-1]) & set(parent2[1:-1]))
        if not common_nodes:
            return parent1 

        cut_node = random.choice(common_nodes)
        
        
        idx1 = parent1.index(cut_node)
        idx2 = parent2.index(cut_node)
        
        new_path = parent1[:idx1] + parent2[idx2:]
        
        
        if len(new_path) != len(set(new_path)):
            return parent1 
            
        return new_path

    @staticmethod
    def _mutate(graph, path, min_bw):
        if len(path) < 3: return path
        
        idx = random.randint(1, len(path)-2)
        mutation_point = path[idx]
        target = path[-1]
        
        
        partial_path = path[:idx+1]
        remaining = GASolver._random_path_from_partial(graph, partial_path, target, min_bw)
        
        if remaining:
            return remaining
        return path

    @staticmethod
    def _random_path_from_partial(graph, current_path, target, min_bw):
        
        path = list(current_path)
        visited = set(path)
        curr = path[-1]
        
        while curr != target:
            neighbors = [n for n in graph.neighbors(curr) 
                         if n not in visited and graph[curr][n].get('bandwidth', 0) >= min_bw]
            if not neighbors: return None
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            curr = next_node
            if len(path) > 250: return None
        return path

# ==========================================
# 4. ARAYÜZ (GUI)
# ==========================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - QoS Routing Project (ACO & GA)")
        self.resize(1300, 850)

        self.G = create_graph_from_csv()
        self.node_count = self.G.number_of_nodes()
        if self.node_count > 0:
            self.pos = nx.spring_layout(self.G, seed=42) 
        else:
            self.pos = {}

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab1 = QWidget()
        self.init_single_run_tab()
        self.tabs.addTab(self.tab1, "🔍 Analiz (Tekli Çalıştırma)")

        self.tab2 = QWidget()
        self.init_batch_test_tab()
        self.tabs.addTab(self.tab2, "📊 Toplu Test (Kıyaslama)")

    def init_single_run_tab(self):
        layout = QHBoxLayout(self.tab1)
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

        self.txt_output = QTextEdit(); self.txt_output.setReadOnly(True)
        l_layout.addWidget(self.txt_output)
        
        layout.addWidget(left_panel)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        if self.node_count > 0: self.plot_graph([], 0, 0)

    def run_single(self):
        S = self.spin_s.value()
        D = self.spin_d.value()
        B = self.spin_bw.value()
        weights = (self.spin_wd.value(), self.spin_wr.value(), self.spin_wres.value())
        algo_choice = self.combo_algo.currentText()

        self.txt_output.setText(f"{algo_choice} Çalışıyor...")
        QApplication.processEvents()

        if "ACO" in algo_choice:
            path, cost, time_ms = ACOSolver.solve(self.G, S, D, weights, min_bw=B)
        else:
            path, cost, time_ms = GASolver.solve(self.G, S, D, weights, min_bw=B)

        if path:
            delay, rel_sum, res_sum, true_rel = compute_metrics(self.G, path)
            msg = (f"✅ {algo_choice} Sonuç:\n"
                   f"Süre: {time_ms:.2f} ms\n"
                   f"Maliyet (Fitness): {cost:.4f}\n"
                   f"----------------------\n"
                   f"Yol Uzunluğu: {len(path)} düğüm\n"
                   f"Yol: {path}\n"
                   f"----------------------\n"
                   f"Toplam Gecikme: {delay:.2f} ms\n"
                   f"Toplam Güvenilirlik: {true_rel:.4f}")
            self.txt_output.setText(msg)
            self.plot_graph(path, S, D)
        else:
            self.txt_output.setText("❌ Yol Bulunamadı (Geçersiz parametreler veya izole düğüm)")
            self.plot_graph(None, S, D)

    def plot_graph(self, path, S, D):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if path:
            path_edges = list(zip(path, path[1:]))
           
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_size=20, node_color='#e0e0e0', alpha=0.3)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, alpha=0.1, edge_color='#cccccc')
            
            
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=path, node_color='orange', node_size=80)
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=path_edges, edge_color='red', width=2)
            
            
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[S], node_color='green', node_size=150, label='Source')
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[D], node_color='blue', node_size=150, label='Dest')
            
            ax.set_title(f"Rota: {S} -> {D}")
        else:
            nx.draw(self.G, self.pos, ax=ax, node_size=30, node_color='lightblue', with_labels=False, alpha=0.5)
            ax.set_title("Ağ Topolojisi")
            
        ax.axis('off')
        self.canvas.draw()

    def init_batch_test_tab(self):
        layout = QVBoxLayout(self.tab2)
        top = QHBoxLayout()
        self.btn_batch = QPushButton("🧪 Toplu Testi Başlat (ACO vs GA)"); 
        self.btn_batch.clicked.connect(self.run_batch)
        top.addWidget(self.btn_batch)
        self.btn_export = QPushButton("💾 CSV Kaydet"); 
        self.btn_export.clicked.connect(self.export_csv)
        top.addWidget(self.btn_export)
        layout.addLayout(top)
        
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["S->D (Talep)", "Algoritma", "Başarı %", "Ort. Maliyet", "Ort. Süre", "En İyi", "En Kötü"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

    def run_batch(self):
        demands = []
        try:
            with open(DEMAND_FILE, 'r') as f:
                reader = csv.reader(f, delimiter=';')
                next(reader)
                for row in reader:
                    if len(row) >= 3:
                        demands.append((int(row[0]), int(row[1]), float(row[2].replace(',','.'))))
        except: 
            QMessageBox.warning(self, "Hata", "DemandData.csv okunamadı!")
            return

        self.table.setRowCount(0)
        self.progress.setMaximum(len(demands) * 2) 
        weights = (0.33, 0.33, 0.34)
        repeats = 5 

        prog_val = 0
        for S, D, B in demands:
            for algo_name in ["ACO", "GA"]:
                costs = []
                times = []
                success_count = 0
                
                for _ in range(repeats):
                    if algo_name == "ACO":
                        path, cost, t = ACOSolver.solve(self.G, S, D, weights, min_bw=B, num_ants=15, num_iterations=15)
                    else:
                        path, cost, t = GASolver.solve(self.G, S, D, weights, min_bw=B, population_size=20, generations=20)
                    
                    if path:
                        success_count += 1
                        costs.append(cost)
                        times.append(t)
                
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(f"{S}->{D} ({B})"))
                self.table.setItem(row, 1, QTableWidgetItem(algo_name))
                
                succ_rate = (success_count / repeats) * 100
                self.table.setItem(row, 2, QTableWidgetItem(f"%{succ_rate:.0f}"))
                
                if costs:
                    avg_cost = sum(costs) / len(costs)
                    avg_time = sum(times) / len(times)
                    best_c = min(costs)
                    worst_c = max(costs)
                    
                    self.table.setItem(row, 3, QTableWidgetItem(f"{avg_cost:.2f}"))
                    self.table.setItem(row, 4, QTableWidgetItem(f"{avg_time:.1f}"))
                    self.table.setItem(row, 5, QTableWidgetItem(f"{best_c:.2f}"))
                    self.table.setItem(row, 6, QTableWidgetItem(f"{worst_c:.2f}"))
                else:
                    for c in range(3, 7): self.table.setItem(row, c, QTableWidgetItem("-"))
                
                prog_val += 1
                self.progress.setValue(prog_val)
                QApplication.processEvents()

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Kaydet", "", "CSV(*.csv)")
        if path:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                headers = [self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())]
                writer.writerow(headers)
                for r in range(self.table.rowCount()):
                    writer.writerow([self.table.item(r,c).text() for c in range(self.table.columnCount())])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())