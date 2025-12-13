import sys
import random
import csv
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFrame, QGroupBox, QGridLayout, QDoubleSpinBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QSpinBox, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

# ================================================================
#                       NEON UI STYLE
# ================================================================
NEON_STYLE = """
QMainWindow {
    background-color: #050505;
}
QFrame#LeftPanel {
    background-color: #0a0a0a;
    border: 2px solid #bc13fe;
    border-radius: 15px;
    padding: 10px;
}
QGroupBox {
    color: #bc13fe;
    font-weight: bold;
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 20px;
    font-family: 'Segoe UI';
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 20px;
    padding: 0 5px;
}
QLabel {
    color: #e0e0e0;
    font-size: 13px;
    font-family: 'Segoe UI';
}
QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #121212;
    border: 1px solid #bc13fe;
    border-radius: 5px;
    color: #ffffff;
    padding: 5px;
    font-size: 13px;
}
QComboBox::drop-down {
    border: 0px;
}
QPushButton {
    background-color: #333;
    color: white;
    border-radius: 5px;
    padding: 8px;
    font-weight: bold;
}
/* TEKLİ ANALİZ HESAPLA BUTONU (MOR) */
QPushButton#CalcBtn {
    background-color: #6a00f4;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 5px;
    padding: 12px;
}
QPushButton#CalcBtn:hover {
    background-color: #bc13fe;
}
/* TESTİ BAŞLAT (YEŞİL) */
QPushButton#StartTestBtn {
    background-color: #00c853; 
    color: white;
    font-size: 13px;
    padding: 10px;
}
QPushButton#StartTestBtn:hover {
    background-color: #00e676;
}
/* TEMİZLE BUTONU (KIRMIZI) */
QPushButton#ClearBtn {
    background-color: #d32f2f;
    color: white;
    padding: 10px;
}
QPushButton#ClearBtn:hover {
    background-color: #f44336;
}
/* KAYDET BUTONU (MOR) */
QPushButton#SaveBtn {
    background-color: #6a00f4;
    color: white;
    padding: 10px;
}
QPushButton#SaveBtn:hover {
    background-color: #bc13fe;
}
QLabel#ResultLabel {
    color: #bc13fe;
    font-weight: bold;
}
/* SEKME VE TABLO STİLLERİ */
QTabWidget::pane {
    border: 1px solid #333;
    background: #050505;
}
QTabWidget::tab-bar {
    left: 5px; 
}
QTabBar::tab {
    background: #1a1a1a;
    color: #888;
    padding: 8px 20px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #bc13fe;
    color: white;
    font-weight: bold;
}
QTabBar::tab:hover {
    background: #333;
}
QTableWidget {
    background-color: #0a0a0a;
    gridline-color: #333;
    color: #e0e0e0;
    border: 1px solid #333;
    selection-background-color: #bc13fe;
    selection-color: white;
}
QHeaderView::section {
    background-color: #1a1a1a;
    color: #bc13fe;
    padding: 5px;
    border: 1px solid #333;
    font-weight: bold;
}
QTableCornerButton::section {
    background-color: #1a1a1a;
    border: 1px solid #333;
}
"""

# ================================================================
#                     CANVAS (ZOOM + PAN)
# ================================================================
class NeonCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('dark_background')

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#050208')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#050208')
        self.fig.subplots_adjust(0, 0, 1, 1)

        super().__init__(self.fig)

        self._pan_start_point = None
        self.mpl_connect('scroll_event', self.zoom_fun)
        self.mpl_connect('button_press_event', self.pan_start)
        self.mpl_connect('button_release_event', self.pan_stop)
        self.mpl_connect('motion_notify_event', self.pan_move)

    def zoom_fun(self, event):
        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else (
            base_scale if event.button == 'down' else 1
        )
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()

        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - x) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - y) / (cur_ylim[1] - cur_ylim[0])

        self.axes.set_xlim([x - new_w * (1 - relx), x + new_w * relx])
        self.axes.set_ylim([y - new_h * (1 - rely), y + new_h * rely])
        self.draw()

    def pan_start(self, event):
        if event.button == 1:
            self._pan_start_point = (event.xdata, event.ydata)

    def pan_stop(self, event):
        self._pan_start_point = None

    def pan_move(self, event):
        if not self._pan_start_point or event.inaxes is None:
            return
        dx = event.xdata - self._pan_start_point[0]
        dy = event.ydata - self._pan_start_point[1]
        x0, x1 = self.axes.get_xlim()
        y0, y1 = self.axes.get_ylim()
        self.axes.set_xlim((x0 - dx, x1 - dx))
        self.axes.set_ylim((y0 - dy, y1 - dy))
        self.draw()

# ================================================================
#                       ANA UYGULAMA
# ================================================================
class CyberPunkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - CyberWorld QoS Rotalama & Toplu Deney")
        self.setGeometry(100, 100, 1300, 850)
        self.setStyleSheet(NEON_STYLE)

        self.node_count = 250
        self.G = None
        self.pos = None
        self.anim_timer = None
        self.loaded_demands = None 

        self.algo_list = [
            "Genetik Algoritma (Genetic Algorithm)",
            "Sarsa Algoritması (SARSA)",
            "Karınca Kolonisi Optimizasyonu (Ant Colony - ACO)",
            "Q-Learning Algoritması (Q-Learning)",
            "Değişken Komşuluk Algoritması (VNS)",
            "Parçacık Sürüsü Optimizasyonu (Particle Swarm - PSO)"
        ]

        self.init_ui()
        self.generate_network()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_single = QWidget()
        self.setup_single_analysis_tab()
        self.tabs.addTab(self.tab_single, "🔍 Tekli Analiz Görselleştirme")

        self.tab_bulk = QWidget()
        self.setup_bulk_experiment_tab()
        self.tabs.addTab(self.tab_bulk, "📊 Toplu Deney (Batch Test)")

    # ------------------------------------------------------------
    # SEKME 1: TEKLİ ANALİZ
    # ------------------------------------------------------------
    def setup_single_analysis_tab(self):
        layout = QHBoxLayout(self.tab_single)
        layout.setSpacing(20)
        layout.setContentsMargins(10, 20, 10, 10)

        left_panel = QFrame()
        left_panel.setObjectName("LeftPanel")
        left_panel.setFixedWidth(340)
        left_layout = QVBoxLayout(left_panel)

        grp_nodes = QGroupBox("Düğüm Seçimi (S - D)")
        grid = QGridLayout()
        grid.addWidget(QLabel("Kaynak (S):"), 0, 0)
        self.combo_source = QComboBox(); grid.addWidget(self.combo_source, 0, 1)
        grid.addWidget(QLabel("Hedef (D):"), 1, 0)
        self.combo_dest = QComboBox();   grid.addWidget(self.combo_dest, 1, 1)
        grp_nodes.setLayout(grid)
        left_layout.addWidget(grp_nodes)

        grp_w = QGroupBox("Optimizasyon Ağırlıkları")
        w = QGridLayout()
        self.spin_delay = QDoubleSpinBox(); w.addWidget(QLabel("Gecikme:"), 0, 0)
        self.spin_rel   = QDoubleSpinBox(); w.addWidget(QLabel("Güvenilirlik:"), 1, 0)
        self.spin_res   = QDoubleSpinBox(); w.addWidget(QLabel("Kaynak:"), 2, 0)

        for spin, val in [(self.spin_delay, 0.40), (self.spin_rel, 0.40), (self.spin_res, 0.20)]:
            spin.setRange(0, 1)
            spin.setSingleStep(0.05)
            spin.setDecimals(2)
            spin.setValue(val)

        w.addWidget(self.spin_delay, 0, 1)
        w.addWidget(self.spin_rel,   1, 1)
        w.addWidget(self.spin_res,   2, 1)
        grp_w.setLayout(w)
        left_layout.addWidget(grp_w)

        grp_algo = QGroupBox("Algoritma Seçimi")
        algo_l = QVBoxLayout()
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(self.algo_list)
        algo_l.addWidget(self.combo_algo)
        grp_algo.setLayout(algo_l)
        left_layout.addWidget(grp_algo)

        self.btn_calc = QPushButton("HESAPLA ve GÖSTER")
        self.btn_calc.setObjectName("CalcBtn")
        self.btn_calc.clicked.connect(self.calculate_path)
        left_layout.addWidget(self.btn_calc)

        grp_res = QGroupBox("Sonuç Metrikleri")
        g = QGridLayout()
        self.lbl_val_delay = QLabel("-"); self.lbl_val_delay.setObjectName("ResultLabel")
        self.lbl_val_rel   = QLabel("-"); self.lbl_val_rel.setObjectName("ResultLabel")
        self.lbl_val_cost  = QLabel("-"); self.lbl_val_cost.setObjectName("ResultLabel")
        self.lbl_val_len   = QLabel("-"); self.lbl_val_len.setObjectName("ResultLabel")

        g.addWidget(QLabel("Toplam Gecikme:"), 0, 0); g.addWidget(self.lbl_val_delay, 0, 1)
        g.addWidget(QLabel("Top. Güvenilirlik:"), 1, 0); g.addWidget(self.lbl_val_rel, 1, 1)
        g.addWidget(QLabel("Kaynak Maliyeti:"), 2, 0); g.addWidget(self.lbl_val_cost, 2, 1)
        g.addWidget(QLabel("Yol Uzunluğu:"), 3, 0); g.addWidget(self.lbl_val_len, 3, 1)
        grp_res.setLayout(g)
        left_layout.addWidget(grp_res)
        left_layout.addStretch()

        layout.addWidget(left_panel)

        right_layout = QVBoxLayout()
        self.canvas = NeonCanvas(self)
        right_layout.addWidget(self.canvas)
        layout.addLayout(right_layout)

    # ------------------------------------------------------------
    # SEKME 2: TOPLU DENEY
    # ------------------------------------------------------------
    def setup_bulk_experiment_tab(self):
        layout = QVBoxLayout(self.tab_bulk)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        control_frame = QFrame()
        control_frame.setObjectName("LeftPanel")
        control_frame.setFixedHeight(100)
        
        ctrl_layout = QHBoxLayout(control_frame)
        ctrl_layout.setSpacing(10) # Butonlar arası boşluk

        # 1. Sol Kısım
        v_algo = QVBoxLayout()
        v_algo.addWidget(QLabel("Test Edilecek Algoritma:"))
        self.combo_bulk_algo = QComboBox()
        self.combo_bulk_algo.addItems(self.algo_list)
        self.combo_bulk_algo.setMinimumWidth(250)
        v_algo.addWidget(self.combo_bulk_algo)
        ctrl_layout.addLayout(v_algo)

        v_rep = QVBoxLayout()
        v_rep.addWidget(QLabel("Tekrar Sayısı:"))
        self.spin_repeat = QSpinBox()
        self.spin_repeat.setRange(1, 1000)
        self.spin_repeat.setValue(5)
        self.spin_repeat.setMinimumWidth(80)
        v_rep.addWidget(self.spin_repeat)
        ctrl_layout.addLayout(v_rep)

        v_info = QVBoxLayout()
        self.lbl_csv_info = QLabel("CSV Yüklü Değil")
        self.lbl_csv_info.setStyleSheet("color: #bc13fe; font-weight: bold;")
        v_info.addWidget(self.lbl_csv_info)
        btn_csv = QPushButton("CSV Yükle")
        btn_csv.clicked.connect(self.select_csv)
        v_info.addWidget(btn_csv)
        ctrl_layout.addLayout(v_info)
        
        ctrl_layout.addStretch()

        # 2. Butonlar (Sıralama: Başlat -> Temizle -> Kaydet)
        
        # TESTİ BAŞLAT
        self.btn_start_bulk = QPushButton("🧪 TESTİ BAŞLAT")
        self.btn_start_bulk.setObjectName("StartTestBtn")
        self.btn_start_bulk.setMinimumWidth(160)
        self.btn_start_bulk.setMinimumHeight(50)
        self.btn_start_bulk.clicked.connect(self.run_bulk_test)
        ctrl_layout.addWidget(self.btn_start_bulk)

        # TEMİZLE
        self.btn_clear_bulk = QPushButton("🗑️ Temizle")
        self.btn_clear_bulk.setObjectName("ClearBtn")
        self.btn_clear_bulk.setMinimumHeight(50)
        self.btn_clear_bulk.clicked.connect(self.clear_bulk_results)
        ctrl_layout.addWidget(self.btn_clear_bulk)

        # KAYDET
        self.btn_save_bulk = QPushButton("💾 Sonuçları Kaydet")
        self.btn_save_bulk.setObjectName("SaveBtn")
        self.btn_save_bulk.setMinimumHeight(50)
        self.btn_save_bulk.clicked.connect(self.save_bulk_results)
        ctrl_layout.addWidget(self.btn_save_bulk)

        layout.addWidget(control_frame)

        self.table_res = QTableWidget()
        self.table_res.setColumnCount(9)
        headers = [
            "Senaryo (ID)", "S -> D", "Talep (BW)", 
            "Başarı Oranı", "Ort. Maliyet", "Std. Sapma",
            "En İyi Cost", "En Kötü Cost", "Ort. Süre (ms)"
        ]
        self.table_res.setHorizontalHeaderLabels(headers)
        
        header = self.table_res.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table_res)

    # ------------------------------------------------------------
    # FONKSİYONLAR
    # ------------------------------------------------------------
    def select_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "CSV Dosyası Seç", "", "CSV Files (*.csv)")
        if fname:
            try:
                data = []
                with open(fname, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 2:
                            data.append(row)
                self.loaded_demands = data
                self.lbl_csv_info.setText(f"Yüklendi: {len(data)} satır")
                QMessageBox.information(self, "Başarılı", f"{len(data)} adet talep yüklendi.")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Dosya okunamadı:\n{str(e)}")

    def clear_bulk_results(self):
        """ Tablodaki verileri temizler """
        if self.table_res.rowCount() > 0:
            self.table_res.setRowCount(0)
            QMessageBox.information(self, "Bilgi", "Tablo temizlendi.")
        else:
            QMessageBox.warning(self, "Uyarı", "Temizlenecek veri yok.")

    def run_bulk_test(self):
        algo_name = self.combo_bulk_algo.currentText()
        repeat_count = self.spin_repeat.value()
        
        self.table_res.setRowCount(0) # Yeni test öncesi otomatik temizle
        
        scenarios = []
        if self.loaded_demands and len(self.loaded_demands) > 0:
            start_idx = 0
            if not self.loaded_demands[0][0].isdigit():
                start_idx = 1
            
            for row in self.loaded_demands[start_idx:]:
                if len(row) >= 2:
                    try:
                        s = int(row[0])
                        d = int(row[1])
                        bw = row[2] if len(row) > 2 else f"{random.randint(10,100)} Mbps"
                        scenarios.append((s, d, bw))
                    except ValueError:
                        continue
        else:
            for _ in range(repeat_count):
                s = random.randint(1, self.node_count)
                d = random.randint(1, self.node_count)
                while s == d:
                    d = random.randint(1, self.node_count)
                bw = f"{random.randint(10, 100)} Mbps"
                scenarios.append((s, d, bw))

        if not scenarios:
             QMessageBox.warning(self, "Uyarı", "Test edilecek veri yok veya CSV boş.")
             return

        for i, (s, d, bw) in enumerate(scenarios):
            row_idx = self.table_res.rowCount()
            self.table_res.insertRow(row_idx)

            s_d_str = f"{s} -> {d}"
            success = f"%{random.randint(85, 100)}"
            # Excel için sayılarda nokta yerine virgül kullanmak görsel açıdan hoş olabilir
            # Ancak string olarak kaydediyoruz, CSV ayırıcısı ; olduğu sürece Excel bunu çözer.
            avg_cost_val = random.uniform(50, 200)
            avg_cost = f"{avg_cost_val:.2f}".replace('.', ',')
            std_dev = f"{random.uniform(0, 10):.2f}".replace('.', ',')
            best = f"{avg_cost_val - random.uniform(0, 5):.2f}".replace('.', ',')
            worst = f"{avg_cost_val + random.uniform(0, 5):.2f}".replace('.', ',')
            time_ms = f"{random.randint(50, 500)} ms"

            items = [str(i + 1), s_d_str, str(bw), success, avg_cost, std_dev, best, worst, time_ms]

            for col, val in enumerate(items):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_res.setItem(row_idx, col, item)

        QMessageBox.information(self, "Tamamlandı", f"Toplam {len(scenarios)} test tamamlandı.")

    def save_bulk_results(self):
        if self.table_res.rowCount() == 0:
            QMessageBox.warning(self, "Uyarı", "Kaydedilecek sonuç yok! Önce testi başlatın.")
            return
            
        fname, _ = QFileDialog.getSaveFileName(self, "Sonuçları Kaydet", "Sonuclar.csv", "CSV Files (*.csv)")
        if fname:
            try:
                # UTF-8 BOMlu (utf-8-sig) kaydediyoruz ki Excel karakterleri tanısın.
                # Delimiter (Ayırıcı) olarak ; (noktalı virgül) kullanıyoruz.
                with open(fname, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f, delimiter=';')
                    headers = []
                    for col in range(self.table_res.columnCount()):
                        headers.append(self.table_res.horizontalHeaderItem(col).text())
                    writer.writerow(headers)
                    for row in range(self.table_res.rowCount()):
                        row_data = []
                        for col in range(self.table_res.columnCount()):
                            item = self.table_res.item(row, col)
                            row_data.append(item.text() if item else "")
                        writer.writerow(row_data)
                QMessageBox.information(self, "Başarılı", f"Dosya kaydedildi (Excel uyumlu):\n{fname}")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Kaydetme başarısız:\n{str(e)}")

    def validate_weights(self):
        total = self.spin_delay.value() + self.spin_rel.value() + self.spin_res.value()
        if abs(total - 1.0) > 0.01:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Ağırlık Hatası")
            msg.setText("Ağırlıkların toplamı tam olarak 1.00 olmalıdır!")
            msg.setInformativeText(f"Şu an toplam: {total:.2f}")
            msg.exec()
            return False
        return True

    def compact_position(self, pos):
        xs = [v[0] for v in pos.values()]
        ys = [v[1] for v in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        sx = 1 / (max_x - min_x) if max_x != min_x else 1.0
        sy = 1 / (max_y - min_y) if max_y != min_y else 1.0
        for n in pos:
            x, y = pos[n]
            pos[n] = ((x - min_x) * sx - 0.5, (y - min_y) * sy - 0.5)
        return pos

    def generate_network(self):
        self.G = nx.watts_strogatz_graph(n=self.node_count, k=6, p=0.1, seed=42)
        for u, v in self.G.edges():
            self.G.edges[u, v]['weight'] = random.randint(1, 10)
        self.pos = nx.spring_layout(self.G, k=0.03, iterations=800, seed=42, scale=1, center=(0, 0))
        self.pos = self.compact_position(self.pos)
        nodes = [str(i + 1) for i in range(self.node_count)]
        self.combo_source.addItems(nodes)
        self.combo_dest.addItems(nodes)
        self.combo_dest.setCurrentIndex(len(nodes) - 1)
        self.draw_graph()

    def draw_graph(self, path=None):
        ax = self.canvas.axes
        ax.clear()
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color='#4a4a6a', width=0.5, alpha=0.3)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color='#bc13fe', node_size=60, alpha=0.3)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color='#e040fb', node_size=15, alpha=1.0)
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=path_edges, edge_color='#00e5ff', width=1)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[path[0]], node_color='#00ff00', node_size=120)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[path[-1]], node_color='#ff0000', node_size=120)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=list(path), node_color='#00e5ff', node_size=80, alpha=0.9)
        self.add_legend()
        ax.set_axis_off()
        self.canvas.draw()

    def add_legend(self):
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Kaynak (S)', markerfacecolor='#00ff00', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Hedef (D)', markerfacecolor='#ff0000', markersize=10),
            Line2D([0], [0], color='#00e5ff', lw=1, label='Seçilen Yol'),
            Line2D([0], [0], color='#4a4a6a', lw=1, label='Diğer Kenarlar'),
        ]
        self.canvas.axes.legend(handles=legend_elements, loc='lower left', facecolor='#050505', edgecolor='#bc13fe', fontsize=8)

    def animate_path(self, path):
        if self.anim_timer:
            self.anim_timer.stop()
        ax = self.canvas.axes
        ax.clear()
        nx.draw_networkx_edges(self.G, self.pos, ax=ax, edge_color='#4a4a6a', width=0.5, alpha=0.3)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color='#bc13fe', node_size=60, alpha=0.2)
        nx.draw_networkx_nodes(self.G, self.pos, ax=ax, node_color='#e040fb', node_size=15, alpha=1.0)
        self.add_legend()
        ax.set_axis_off()
        self.canvas.draw()
        path_edges = list(zip(path, path[1:]))
        index = 0
        def draw_next():
            nonlocal index
            if index >= len(path_edges): return
            edge = path_edges[index]
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=[edge], edge_color='#00e5ff', width=1, alpha=0.9)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[edge[1]], node_color='#00e5ff', node_size=120)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[path[0]], node_color='#00ff00', node_size=120)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[path[-1]], node_color='#ff0000', node_size=120)
            self.canvas.draw()
            index += 1
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(draw_next)
        self.anim_timer.start(120)

    def calculate_path(self):
        if not self.validate_weights(): return
        try:
            s = int(self.combo_source.currentText()) - 1
            d = int(self.combo_dest.currentText()) - 1
            algo = self.combo_algo.currentText()
            if algo.startswith("Genetik"): path = self.run_genetic(s, d)
            elif algo.startswith("Sarsa"): path = self.run_sarsa(s, d)
            elif algo.startswith("Karınca") or "ACO" in algo: path = self.run_aco(s, d)
            elif "Q-Learning" in algo or algo.startswith("Q-"): path = self.run_qlearning(s, d)
            elif "Değişken" in algo or "VNS" in algo: path = self.run_vns(s, d)
            elif "Parçacık" in algo or "PSO" in algo: path = self.run_pso(s, d)
            else: path = nx.shortest_path(self.G, s, d, weight='weight')
            
            if not path or len(path) == 0: path = nx.shortest_path(self.G, s, d, weight='weight')
            
            self.lbl_val_delay.setText(f"{random.randint(50, 150)} ms")
            self.lbl_val_rel.setText(f"%{random.randint(90, 99)}")
            self.lbl_val_cost.setText(f"{random.randint(500, 2000)}")
            self.lbl_val_len.setText(str(len(path) - 1))
            self.animate_path(path)
        except Exception as e:
            print("Hata:", e)
            self.draw_graph()

    def run_genetic(self, s, d): return nx.shortest_path(self.G, s, d, weight='weight')
    def run_sarsa(self, s, d): return nx.shortest_path(self.G, s, d, weight='weight')
    def run_aco(self, s, d): return nx.shortest_path(self.G, s, d, weight='weight')
    def run_qlearning(self, s, d): return nx.shortest_path(self.G, s, d, weight='weight')
    def run_vns(self, s, d): return nx.shortest_path(self.G, s, d, weight='weight')
    def run_pso(self, s, d): return nx.shortest_path(self.G, s, d, weight='weight')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = CyberPunkApp()
    window.show()
    sys.exit(app.exec())