import sys
import random
import csv
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import pandas as pd

# =================================================================================================
# MODÜL TANITIMI
# =================================================================================================
# Bu dosya (Arayuz.py), projenin ANA ÇALIŞTIRILABİLİR (Main) dosyasıdır.
# PyQt6 kütüphanesi kullanılarak geliştirilmiş modern bir Grafik Arayüz (GUI) sunar.
#
# TEMEL GÖREVLERİ:
# 1. Ağı Görselleştirme: NetworkX ve Matplotlib kullanarak düğüm ve kenarları çizer.
# 2. Algoritmaları Yönetme: Sarsa, Q-Learning, Genetik, ACO, PSO gibi algoritmaları import eder ve çalıştırır.
# 3. Parametre Yönetimi: Kullanıcının ağırlık (Gecikme, Güvenilirlik vb.) ve hiperparametre girmesini sağlar.
# 4. Toplu Deney (Bulk Test): CSV dosyasından yüzlerce senaryoyu okuyup algoritmaları kıyaslar.
# =================================================================================================

# -------------------------------------------------------------------------------------------------
# 1. HARİCİ MODÜLLERİN İÇE AKTARILMASI (IMPORT)
# -------------------------------------------------------------------------------------------------
# QoS maliyet hesaplama modülü: Tüm algoritmaların ortak bir matematiksel dil kullanmasını sağlar.
# (Böylece örneğin Genetik Algoritma'nın hesapladığı maliyet ile Sarsa'nınki kıyaslanabilir olur)
from qos_maliyet import (
    compute_edge_cost,
    compute_path_cost,
    validate_path_bandwidth,
    compute_path_metrics
)

# Q-Learning modülünden gerekli fonksiyonları import et
from Q_Learning_Gokberk_Gok_ import (
    QLearning, 
    train_q_learning,
    path_total_delay,
    path_reliability_cost,
    path_resource_cost,
    total_cost
)

# -------------------------------------------------------------------------------------------------
# DİNAMİK MODÜL YÜKLEME (Dynamic Import)
# -------------------------------------------------------------------------------------------------
# Projedeki diğer algoritma dosyaları (team member'ların yazdığı kodlar) standart bir Python paketi 
# yapısında olmayabilir veya isimleri değişken olabilir. Bu yüzden `importlib` kullanarak 
# dosya yolundan (path) doğrudan yükleme yapıyoruz. Bu yöntem, dosya isimleri değişse bile 
# kodun kolayca adapte edilmesini sağlar.

# SARSA modülünden gerekli fonksiyonları import et
import importlib.util
import os
sarsa_path = os.path.join(os.path.dirname(__file__), "Sarsa_Algoritmasi_Oguzhan_Demirbas.py")
spec = importlib.util.spec_from_file_location("sarsa_module", sarsa_path)
sarsa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sarsa_module)
sarsa_route = sarsa_module.sarsa_route
compute_cost_sarsa = sarsa_module.compute_cost

# VNS modülünden gerekli sınıfları import et
vns_path = os.path.join(os.path.dirname(__file__), "VNS_Algorithm_Yigit_Emre.py")
spec_vns = importlib.util.spec_from_file_location("vns_module", vns_path)
vns_module = importlib.util.module_from_spec(spec_vns)
spec_vns.loader.exec_module(vns_module)
NetworkGraph = vns_module.NetworkGraph
VNS = vns_module.VNS

# PSO modülünden gerekli sınıfları import et
pso_path = os.path.join(os.path.dirname(__file__), "Parcacık_Surusu_Optimizasyonu_Salim_Caner.py")
spec_pso = importlib.util.spec_from_file_location("pso_module", pso_path)
pso_module = importlib.util.module_from_spec(spec_pso)
spec_pso.loader.exec_module(pso_module)
PSO = pso_module.PSO

# ACO modülünden gerekli sınıfları import et
aco_path = os.path.join(os.path.dirname(__file__), "Karınca_Kolonisi_Algoritmasi_Aivaz_Arysbay.py")
spec_aco = importlib.util.spec_from_file_location("aco_module", aco_path)
aco_module = importlib.util.module_from_spec(spec_aco)
spec_aco.loader.exec_module(aco_module)
ACOSolver = aco_module.ACOSolver

# Genetik Algoritma modülünden gerekli fonksiyonları import et
genetic_path = os.path.join(os.path.dirname(__file__), "Genetik_Algoritmasi_Azra_Kaya.py")
spec_genetic = importlib.util.spec_from_file_location("genetic_module", genetic_path)
genetic_module = importlib.util.module_from_spec(spec_genetic)
spec_genetic.loader.exec_module(genetic_module)
genetic_algorithm = genetic_module.genetic_algorithm

# PyQt6 Bileşenleri: Modern ve tepkisel bir arayüz oluşturmak için kullanılır.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QFrame, QGroupBox, QGridLayout, QDoubleSpinBox,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QSpinBox, QHeaderView, QFileDialog, QDialog, QTextEdit
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import time

# ================================================================
#                       NEON UI STYLE
# ================================================================
# Arayüzün estetik görünümü için CSS benzeri stil tanımlamaları.
# "Cyberpunk / Neon" teması seçilerek modern ve dikkat çekici bir görünüm hedeflendi.
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
QComboBox QAbstractItemView {
    background-color: #1a1a1a;
    color: #ffffff;
    selection-background-color: #bc13fe;
    selection-color: #ffffff;
    border: 1px solid #bc13fe;
    font-size: 13px;
}
/* SpinBox Ok Butonları - Gizli */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    width: 0px;
    border: none;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 0px;
    border: none;
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
/* DURDUR BUTONU (TURUNCU) */
QPushButton#PauseBtn {
    background-color: #ff6f00;
    color: white;
    padding: 10px;
}
QPushButton#PauseBtn:hover {
    background-color: #ff8f00;
}
/* DEVAM ET BUTONU (MAVİ) */
QPushButton#ResumeBtn {
    background-color: #0091ea;
    color: white;
    padding: 10px;
}
QPushButton#ResumeBtn:hover {
    background-color: #00b0ff;
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
#                Q-LEARNING PARAMETRE DIALOG
# ================================================================
class QLearningParamsDialog(QDialog):
    """Q-Learning hiperparametrelerini ayarlamak için dialog penceresi"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Q-Learning Parametreleri")
        self.setModal(True)
        self.setStyleSheet(NEON_STYLE)
        self.setFixedSize(400, 350)
        
        # Varsayılan değerler
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2
        self.episodes = 300
        self.max_steps = 250
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        title = QLabel("🎓 Q-Learning Hiperparametreleri")
        title.setStyleSheet("color: #bc13fe; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Açıklama
        desc = QLabel("Algoritma parametrelerini özelleştirin:")
        desc.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(desc)
        
        # Parametreler grubu
        params_group = QGroupBox("Parametreler")
        params_layout = QGridLayout()
        
        # Alpha (Öğrenme oranı)
        lbl_alpha = QLabel("Alpha (Öğrenme Oranı):")
        lbl_alpha.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_alpha, 0, 0)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 1.0)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.setValue(self.alpha)
        params_layout.addWidget(self.spin_alpha, 0, 1)
        
        # Gamma (İndirim faktörü)
        lbl_gamma = QLabel("Gamma (İndirim Faktörü):")
        lbl_gamma.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_gamma, 1, 0)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 0.999)
        self.spin_gamma.setSingleStep(0.01)
        self.spin_gamma.setDecimals(3)
        self.spin_gamma.setValue(self.gamma)
        params_layout.addWidget(self.spin_gamma, 1, 1)
        
        # Epsilon (Keşif oranı)
        lbl_epsilon = QLabel("Epsilon (Keşif Oranı):")
        lbl_epsilon.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_epsilon, 2, 0)
        self.spin_epsilon = QDoubleSpinBox()
        self.spin_epsilon.setRange(0.0, 1.0)
        self.spin_epsilon.setSingleStep(0.05)
        self.spin_epsilon.setDecimals(2)
        self.spin_epsilon.setValue(self.epsilon)
        params_layout.addWidget(self.spin_epsilon, 2, 1)
        
        # Episodes
        lbl_episodes = QLabel("Episodes (Eğitim Sayısı):")
        lbl_episodes.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_episodes, 3, 0)
        self.spin_episodes = QSpinBox()
        self.spin_episodes.setRange(10, 1000)
        self.spin_episodes.setSingleStep(10)
        self.spin_episodes.setValue(self.episodes)
        params_layout.addWidget(self.spin_episodes, 3, 1)
        
        # Max Steps
        lbl_max_steps = QLabel("Max Steps (Maks. Adım):")
        lbl_max_steps.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_max_steps, 4, 0)
        self.spin_max_steps = QSpinBox()
        self.spin_max_steps.setRange(50, 500)
        self.spin_max_steps.setSingleStep(10)
        self.spin_max_steps.setValue(self.max_steps)
        params_layout.addWidget(self.spin_max_steps, 4, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        # Varsayılan değerlere dön butonu
        btn_reset = QPushButton("🔄 Varsayılan Değerler")
        btn_reset.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(btn_reset)
        
        button_layout.addStretch()
        
        # Tamam butonu
        btn_ok = QPushButton("✅ Tamam")
        btn_ok.setObjectName("CalcBtn")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        # İptal butonu
        btn_cancel = QPushButton("❌ İptal")
        btn_cancel.setObjectName("ClearBtn")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def reset_to_defaults(self):
        """Varsayılan değerlere dön"""
        self.spin_alpha.setValue(0.1)
        self.spin_gamma.setValue(0.9)
        self.spin_epsilon.setValue(0.9)
        self.spin_episodes.setValue(300)
        self.spin_max_steps.setValue(250)
    
    def get_params(self):
        """Parametreleri döndür"""
        return {
            'alpha': self.spin_alpha.value(),
            'gamma': self.spin_gamma.value(),
            'epsilon': self.spin_epsilon.value(),
            'episodes': self.spin_episodes.value(),
            'max_steps': self.spin_max_steps.value()
        }

# ================================================================
#                SARSA PARAMETRE DIALOG
# ================================================================
class SARSAParamsDialog(QDialog):
    """SARSA hiperparametrelerini ayarlamak için dialog penceresi"""
    
    def __init__(self, parent=None, default_bw=100.0):
        super().__init__(parent)
        self.setWindowTitle("SARSA Parametreleri")
        self.setModal(True)
        self.setStyleSheet(NEON_STYLE)
        self.setFixedSize(400, 350)
        
        # Varsayılan değerler
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.3
        self.episodes = 2000
        self.min_bandwidth = default_bw
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        title = QLabel("🎯 SARSA Hiperparametreleri")
        title.setStyleSheet("color: #bc13fe; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Açıklama
        desc = QLabel("Algoritma parametrelerini özelleştirin:")
        desc.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(desc)
        
        # Parametreler grubu
        params_group = QGroupBox("Parametreler")
        params_layout = QGridLayout()
        
        # Alpha (Öğrenme oranı)
        lbl_alpha = QLabel("Alpha (Öğrenme Oranı):")
        lbl_alpha.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_alpha, 0, 0)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 1.0)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.setValue(self.alpha)
        params_layout.addWidget(self.spin_alpha, 0, 1)
        
        # Gamma (İndirim faktörü)
        lbl_gamma = QLabel("Gamma (İndirim Faktörü):")
        lbl_gamma.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_gamma, 1, 0)
        self.spin_gamma = QDoubleSpinBox()
        self.spin_gamma.setRange(0.1, 0.999)
        self.spin_gamma.setSingleStep(0.01)
        self.spin_gamma.setDecimals(3)
        self.spin_gamma.setValue(self.gamma)
        params_layout.addWidget(self.spin_gamma, 1, 1)
        
        # Epsilon (Keşif oranı)
        lbl_epsilon = QLabel("Epsilon (Keşif Oranı):")
        lbl_epsilon.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_epsilon, 2, 0)
        self.spin_epsilon = QDoubleSpinBox()
        self.spin_epsilon.setRange(0.0, 1.0)
        self.spin_epsilon.setSingleStep(0.05)
        self.spin_epsilon.setDecimals(2)
        self.spin_epsilon.setValue(self.epsilon)
        params_layout.addWidget(self.spin_epsilon, 2, 1)
        
        # Episodes
        lbl_episodes = QLabel("Episodes (Eğitim Sayısı):")
        lbl_episodes.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_episodes, 3, 0)
        self.spin_episodes = QSpinBox()
        self.spin_episodes.setRange(100, 5000)
        self.spin_episodes.setSingleStep(100)
        self.spin_episodes.setValue(self.episodes)
        params_layout.addWidget(self.spin_episodes, 3, 1)
        
        # Min Bandwidth
        lbl_min_bw = QLabel("Min Bandwidth (Mbps):")
        lbl_min_bw.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_min_bw, 4, 0)
        self.spin_min_bw = QDoubleSpinBox()
        self.spin_min_bw.setRange(0.1, 1000.0)
        self.spin_min_bw.setSingleStep(10.0)
        self.spin_min_bw.setDecimals(1)
        self.spin_min_bw.setValue(self.min_bandwidth)
        params_layout.addWidget(self.spin_min_bw, 4, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        # Varsayılan değerlere dön butonu
        btn_reset = QPushButton("🔄 Varsayılan Değerler")
        btn_reset.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(btn_reset)
        
        button_layout.addStretch()
        
        # Tamam butonu
        btn_ok = QPushButton("✅ Tamam")
        btn_ok.setObjectName("CalcBtn")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        # İptal butonu
        btn_cancel = QPushButton("❌ İptal")
        btn_cancel.setObjectName("ClearBtn")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def reset_to_defaults(self):
        """Varsayılan değerlere dön"""
        self.spin_alpha.setValue(0.1)
        self.spin_gamma.setValue(0.95)
        self.spin_epsilon.setValue(0.3)
        self.spin_episodes.setValue(2000)
        self.spin_min_bw.setValue(10.0)
    
    def get_params(self):
        """Parametreleri döndür"""
        return {
            'alpha': self.spin_alpha.value(),
            'gamma': self.spin_gamma.value(),
            'epsilon': self.spin_epsilon.value(),
            'episodes': self.spin_episodes.value(),
            'min_bandwidth': self.spin_min_bw.value()
        }

# ================================================================
#                VNS PARAMETRE DIALOG
# ================================================================
class VNSParamsDialog(QDialog):
    """VNS hiperparametrelerini ayarlamak için dialog penceresi"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VNS Parametreleri")
        self.setModal(True)
        self.setStyleSheet(NEON_STYLE)
        self.setFixedSize(400, 300)
        
        # Varsayılan değerler
        self.max_iterations = 20
        self.k_max = 3
        self.test_runs = 1  # GUI için tek çalıştırma yeterli
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        title = QLabel("🔍 VNS Hiperparametreleri")
        title.setStyleSheet("color: #bc13fe; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Açıklama
        desc = QLabel("Variable Neighborhood Search parametrelerini özelleştirin:")
        desc.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(desc)
        
        # Parametreler grubu
        params_group = QGroupBox("Parametreler")
        params_layout = QGridLayout()
        
        # Max Iterations
        lbl_max_iter = QLabel("Max Iterations:")
        lbl_max_iter.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_max_iter, 0, 0)
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(5, 100)
        self.spin_max_iter.setSingleStep(5)
        self.spin_max_iter.setValue(self.max_iterations)
        params_layout.addWidget(self.spin_max_iter, 0, 1)
        
        # K Max (Neighborhood size)
        lbl_k_max = QLabel("K Max (Komşuluk):")
        lbl_k_max.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_k_max, 1, 0)
        self.spin_k_max = QSpinBox()
        self.spin_k_max.setRange(1, 10)
        self.spin_k_max.setSingleStep(1)
        self.spin_k_max.setValue(self.k_max)
        params_layout.addWidget(self.spin_k_max, 1, 1)
        
        # Test Runs
        lbl_test_runs = QLabel("Test Runs:")
        lbl_test_runs.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_test_runs, 2, 0)
        self.spin_test_runs = QSpinBox()
        self.spin_test_runs.setRange(1, 10)
        self.spin_test_runs.setSingleStep(1)
        self.spin_test_runs.setValue(self.test_runs)
        params_layout.addWidget(self.spin_test_runs, 2, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        # Varsayılan değerlere dön butonu
        btn_reset = QPushButton("🔄 Varsayılan Değerler")
        btn_reset.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(btn_reset)
        
        button_layout.addStretch()
        
        # Tamam butonu
        btn_ok = QPushButton("✅ Tamam")
        btn_ok.setObjectName("CalcBtn")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        # İptal butonu
        btn_cancel = QPushButton("❌ İptal")
        btn_cancel.setObjectName("ClearBtn")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def reset_to_defaults(self):
        """Varsayılan değerlere dön"""
        self.spin_max_iter.setValue(20)
        self.spin_k_max.setValue(3)
        self.spin_test_runs.setValue(1)
    
    def get_params(self):
        """Parametreleri döndür"""
        return {
            'max_iterations': self.spin_max_iter.value(),
            'k_max': self.spin_k_max.value(),
            'test_runs': self.spin_test_runs.value()
        }

# ================================================================
#                PSO PARAMETRE DIALOG
# ================================================================
class PSOParamsDialog(QDialog):
    """PSO hiperparametrelerini ayarlamak için dialog penceresi"""
    
    def __init__(self, parent=None, default_bw=10.0):
        super().__init__(parent)
        self.setWindowTitle("PSO Parametreleri")
        self.setModal(True)
        self.setStyleSheet(NEON_STYLE)
        self.setFixedSize(400, 300)
        
        # Varsayılan değerler
        self.num_particles = 30
        self.iterations = 100
        self.min_bandwidth = default_bw
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        title = QLabel("🤖 PSO Hiperparametreleri")
        title.setStyleSheet("color: #bc13fe; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Açıklama
        desc = QLabel("Particle Swarm Optimization parametrelerini özelleştirin:")
        desc.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(desc)
        
        # Parametreler grubu
        params_group = QGroupBox("Parametreler")
        params_layout = QGridLayout()
        
        # Parçacık Sayısı
        lbl_particles = QLabel("Number of Particles:")
        lbl_particles.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_particles, 0, 0)
        self.spin_particles = QSpinBox()
        self.spin_particles.setRange(5, 100)
        self.spin_particles.setSingleStep(5)
        self.spin_particles.setValue(self.num_particles)
        params_layout.addWidget(self.spin_particles, 0, 1)
        
        # İterasyon Sayısı
        lbl_iterations = QLabel("Iterations:")
        lbl_iterations.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_iterations, 1, 0)
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 500)
        self.spin_iterations.setSingleStep(10)
        self.spin_iterations.setValue(self.iterations)
        params_layout.addWidget(self.spin_iterations, 1, 1)
        
        # Min Bandwidth Constraint
        lbl_bw = QLabel("Min Bandwidth (Mbps):")
        lbl_bw.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_bw, 2, 0)
        self.spin_bw = QDoubleSpinBox()
        self.spin_bw.setRange(0, 1000)
        self.spin_bw.setValue(self.min_bandwidth)
        params_layout.addWidget(self.spin_bw, 2, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        # Varsayılan değerlere dön butonu
        btn_reset = QPushButton("🔄 Varsayılan Değerler")
        btn_reset.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(btn_reset)
        
        button_layout.addStretch()
        
        # Tamam butonu
        btn_ok = QPushButton("✅ Tamam")
        btn_ok.setObjectName("CalcBtn")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        # İptal butonu
        btn_cancel = QPushButton("❌ İptal")
        btn_cancel.setObjectName("ClearBtn")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def reset_to_defaults(self):
        """Varsayılan değerlere dön"""
        self.spin_particles.setValue(30)
        self.spin_iterations.setValue(100)
        self.spin_bw.setValue(10.0)
    
    def get_params(self):
        """Parametreleri döndür"""
        return {
            'num_particles': self.spin_particles.value(),
            'iterations': self.spin_iterations.value(),
            'min_bandwidth': self.spin_bw.value()
        }

# ================================================================
#                ACO PARAMETRE DIALOG
# ================================================================
class ACOParamsDialog(QDialog):
    """ACO hiperparametrelerini ayarlamak için dialog penceresi"""
    
    def __init__(self, parent=None, default_bw=10.0):
        super().__init__(parent)
        self.setWindowTitle("ACO Parametreleri")
        self.setModal(True)
        self.setStyleSheet(NEON_STYLE)
        self.setFixedSize(400, 300)
        
        # Varsayılan değerler
        self.num_ants = 20
        self.num_iterations = 30
        self.min_bandwidth = default_bw
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Başlık
        title = QLabel("🐜 ACO Hiperparametreleri")
        title.setStyleSheet("color: #bc13fe; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Açıklama
        desc = QLabel("Ant Colony Optimization parametrelerini özelleştirin:")
        desc.setStyleSheet("color: #888; font-size: 12px;")
        layout.addWidget(desc)
        
        # Parametreler grubu
        params_group = QGroupBox("Parametreler")
        params_layout = QGridLayout()
        
        # Karınca Sayısı
        lbl_ants = QLabel("Karınca Sayısı (Ants):")
        lbl_ants.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_ants, 0, 0)
        self.spin_ants = QSpinBox()
        self.spin_ants.setRange(5, 200)
        self.spin_ants.setSingleStep(5)
        self.spin_ants.setValue(self.num_ants)
        params_layout.addWidget(self.spin_ants, 0, 1)
        
        # İterasyon Sayısı
        lbl_iterations = QLabel("İterasyon (Iterations):")
        lbl_iterations.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_iterations, 1, 0)
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 500)
        self.spin_iterations.setSingleStep(10)
        self.spin_iterations.setValue(self.num_iterations)
        params_layout.addWidget(self.spin_iterations, 1, 1)
        
        # Min Bandwidth Constraint
        lbl_bw = QLabel("Min Bandwidth (Mbps):")
        lbl_bw.setStyleSheet("color: #2a2a2a; font-weight: bold;")
        params_layout.addWidget(lbl_bw, 2, 0)
        self.spin_bw = QDoubleSpinBox()
        self.spin_bw.setRange(0, 1000)
        self.spin_bw.setValue(self.min_bandwidth)
        params_layout.addWidget(self.spin_bw, 2, 1)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Butonlar
        button_layout = QHBoxLayout()
        
        # Varsayılan değerlere dön butonu
        btn_reset = QPushButton("🔄 Varsayılan Değerler")
        btn_reset.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(btn_reset)
        
        button_layout.addStretch()
        
        # Tamam butonu
        btn_ok = QPushButton("✅ Tamam")
        btn_ok.setObjectName("CalcBtn")
        btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(btn_ok)
        
        # İptal butonu
        btn_cancel = QPushButton("❌ İptal")
        btn_cancel.setObjectName("ClearBtn")
        btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(btn_cancel)
        
        layout.addLayout(button_layout)
    
    def reset_to_defaults(self):
        """Varsayılan değerlere dön"""
        self.spin_ants.setValue(20)
        self.spin_iterations.setValue(30)
        self.spin_bw.setValue(10.0)
    
    def get_params(self):
        """Parametreleri döndür"""
        return {
            'num_ants': self.spin_ants.value(),
            'num_iterations': self.spin_iterations.value(),
            'min_bandwidth': self.spin_bw.value()
        }

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
# ================================================================
#                       ANA UYGULAMA SINIFI
# ================================================================
class CyberPunkApp(QMainWindow):
    """
    Uygulamanın ana penceresidir (Main Window).
    Tüm grafiksel bileşenleri, olay döngülerini ve algoritma çağrılarını yönetir.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSM307 - CyberWorld QoS Rotalama & Toplu Deney")
        self.setGeometry(100, 100, 1300, 850)
        
        # Grafik arayüz için özel CSS stilini uygula
        self.setStyleSheet(NEON_STYLE)

        # Temel değişkenler
        self.node_count = 250   # Proje gereksinimi: 250 düğüm
        self.G = None           # NetworkX graf nesnesi (Ağın matematiksel modeli)
        self.pos = None         # Düğümlerin ekrandaki koordinatları (layout)
        self.anim_timer = None  # Animasyon zamanlayıcısı
        self.loaded_demands = None # DemandData.csv'den okunan veriler
        self.test_paused = False   # Toplu test duraklatıldı mı?

        # Kullanıcının seçebileceği algoritmaların listesi
        self.algo_list = [
            "Genetik Algoritma (Genetic Algorithm)",
            "Sarsa Algoritması (SARSA)",
            "Karınca Kolonisi Optimizasyonu (Ant Colony - ACO)",
            "Q-Learning Algoritması (Q-Learning)",
            "Değişken Komşuluk Algoritması (VNS)",
            "Parçacık Sürüsü Optimizasyonu (Particle Swarm - PSO)"
        ]

        # Arayüzü kur
        self.init_ui()
        
        # Ağı oluştur (Nodes + Edges)
        self.generate_network()
        
        # Test senaryolarını yükle
        self.load_demand_data()

    def init_ui(self):
        """
        Ana pencere düzenini (Layout) oluşturur.
        İki ana sekme (Tab) içerir:
        1. Tekli Analiz: Grafiği gösterir, tek bir hesaplama yapar.
        2. Toplu Deney: Yüzlerce testi peş peşe çalıştırır.
        """
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Sekme kontrolcüsü
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Sekme 1: Tekli Analiz
        self.tab_single = QWidget()
        self.setup_single_analysis_tab()
        self.tabs.addTab(self.tab_single, "🔍 Tekli Analiz Görselleştirme")

        # Sekme 2: Toplu Deney
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
        
        # Min Bant Genişliği
        grid.addWidget(QLabel("Min BW (Mbps):"), 2, 0)
        self.spin_main_bw = QDoubleSpinBox()
        self.spin_main_bw.setRange(0, 10000)
        self.spin_main_bw.setValue(100.0)
        grid.addWidget(self.spin_main_bw, 2, 1)
        
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
        grp_res.setMaximumHeight(100)
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
        
        # Algoritma Log Alanı
        grp_log = QGroupBox("📋 Algoritma Logları")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #00ff00;
                border: 1px solid #333;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 5px;
            }
        """)
        self.log_text.setPlaceholderText("Algoritma çıktıları burada görünecek...")
        log_layout.addWidget(self.log_text)
        
        # Log temizleme butonu
        btn_clear_log = QPushButton("🗑️ Logları Temizle")
        btn_clear_log.setObjectName("ClearBtn")
        btn_clear_log.setMaximumHeight(30)
        btn_clear_log.clicked.connect(lambda: self.log_text.clear())
        log_layout.addWidget(btn_clear_log)
        
        grp_log.setLayout(log_layout)
        left_layout.addWidget(grp_log)

        layout.addWidget(left_panel)

        # ORTA: Grafik Canvas
        self.canvas = NeonCanvas(self)
        layout.addWidget(self.canvas, stretch=1)
        
        # SAĞ: Analiz Paneli (Dikey)
        right_panel = QFrame()
        right_panel.setObjectName("LeftPanel") # Aynı stili kullanmak için ID'yi koruyoruz
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(15, 20, 15, 20)
        right_layout.setSpacing(20)
        
        # Başlık
        analysis_title = QLabel("📊 Yol Analizi")
        analysis_title.setStyleSheet("color: #bc13fe; font-weight: bold; font-size: 18px; border-bottom: 2px solid #333; padding-bottom: 10px;")
        analysis_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(analysis_title)
        
        # 1. Algoritma
        lbl_algo_title = QLabel("Algoritma:")
        lbl_algo_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        right_layout.addWidget(lbl_algo_title)
        
        self.lbl_analysis_algo = QLabel("-")
        self.lbl_analysis_algo.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 16px;")
        self.lbl_analysis_algo.setWordWrap(True)
        right_layout.addWidget(self.lbl_analysis_algo)
        
        # Ayırıcı çizgi
        line1 = QFrame(); line1.setFrameShape(QFrame.Shape.HLine); line1.setStyleSheet("color: #333;")
        right_layout.addWidget(line1)
        
        # 2. Yol Uzunluğu
        lbl_path_title = QLabel("Yol Uzunluğu:")
        lbl_path_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        right_layout.addWidget(lbl_path_title)
        
        self.lbl_analysis_path_len = QLabel("-")
        self.lbl_analysis_path_len.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 16px;")
        right_layout.addWidget(self.lbl_analysis_path_len)
        
        # 3. Süre
        lbl_time_title = QLabel("Hesaplama Süresi:")
        lbl_time_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        right_layout.addWidget(lbl_time_title)
        
        self.lbl_analysis_time = QLabel("-")
        self.lbl_analysis_time.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 16px;")
        right_layout.addWidget(self.lbl_analysis_time)
        
        # Ayırıcı çizgi
        line2 = QFrame(); line2.setFrameShape(QFrame.Shape.HLine); line2.setStyleSheet("color: #333;")
        right_layout.addWidget(line2)
        
        # 4. Toplam Maliyet
        lbl_cost_title = QLabel("Toplam Maliyet:")
        lbl_cost_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        right_layout.addWidget(lbl_cost_title)
        
        self.lbl_analysis_cost = QLabel("-")
        self.lbl_analysis_cost.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 20px;")
        self.lbl_analysis_cost.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.lbl_analysis_cost)
        
        # 5. Durum
        lbl_status_title = QLabel("Durum:")
        lbl_status_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        right_layout.addWidget(lbl_status_title)
        
        self.lbl_analysis_status = QLabel("-")
        self.lbl_analysis_status.setStyleSheet("color: #ffaa00; font-style: italic; font-size: 14px;")
        self.lbl_analysis_status.setWordWrap(True)
        right_layout.addWidget(self.lbl_analysis_status)
        
        # 6. Yol (Path)
        lbl_path_route_title = QLabel("Yol:")
        lbl_path_route_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #aaa;")
        right_layout.addWidget(lbl_path_route_title)
        
        self.lbl_analysis_path = QLabel("-")
        self.lbl_analysis_path.setStyleSheet("color: #00e5ff; font-size: 12px;")
        self.lbl_analysis_path.setWordWrap(True)
        right_layout.addWidget(self.lbl_analysis_path)
        
        right_layout.addStretch()
        layout.addWidget(right_panel)

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
        
        # Algoritma seçimi ve Ayarlar butonu yan yana
        h_algo_layout = QHBoxLayout()
        self.combo_bulk_algo = QComboBox()
        self.combo_bulk_algo.addItems(self.algo_list)
        self.combo_bulk_algo.setMinimumWidth(250)
        h_algo_layout.addWidget(self.combo_bulk_algo)
        
        # Ayarlar butonu (icon: unicode gear ⚙️)
        self.btn_bulk_settings = QPushButton("⚙️")
        self.btn_bulk_settings.setFixedSize(40, 30) # Küçük kare buton
        self.btn_bulk_settings.setToolTip("Algoritma Parametrelerini Ayarla")
        self.btn_bulk_settings.clicked.connect(self.configure_bulk_algo)
        h_algo_layout.addWidget(self.btn_bulk_settings)
        
        v_algo.addLayout(h_algo_layout)
        ctrl_layout.addLayout(v_algo)

        ctrl_layout.addStretch()

        # 2. Butonlar (Sıralama: Başlat -> Temizle -> Kaydet)
        
        # TESTİ BAŞLAT
        self.btn_start_bulk = QPushButton("🧪 TESTİ BAŞLAT")
        self.btn_start_bulk.setObjectName("StartTestBtn")
        self.btn_start_bulk.setMinimumWidth(160)
        self.btn_start_bulk.setMinimumHeight(50)
        self.btn_start_bulk.clicked.connect(self.run_bulk_test)
        ctrl_layout.addWidget(self.btn_start_bulk)

        # TESTİ DURDUR/DEVAM ET
        self.btn_pause_bulk = QPushButton("⏸️ Testi Durdur")
        self.btn_pause_bulk.setObjectName("PauseBtn")
        self.btn_pause_bulk.setMinimumHeight(50)
        self.btn_pause_bulk.setEnabled(False)
        self.btn_pause_bulk.clicked.connect(self.toggle_pause_test)
        ctrl_layout.addWidget(self.btn_pause_bulk)

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
    def log(self, message):
        """Log mesajını log alanına ekle"""
        self.log_text.append(message)
        # Otomatik olarak en alta kaydır
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def load_demand_data(self):
        """DemandData.csv dosyasını otomatik yükle"""
        self.loaded_demands = []
        try:
             # Önce ; ayırıcı ile dene
            demand_csv = os.path.join(os.path.dirname(__file__), "BSM307_317_Guz2025_TermProject_DemandData.csv")
            try:
                df = pd.read_csv(demand_csv, sep=";", decimal=",")
                if df.shape[1] < 3: df = pd.read_csv(demand_csv, sep=",", decimal=".")
            except:
                df = pd.read_csv(demand_csv, sep=",", decimal=".")

            if len(df) > 0:
                # DataFrame to list of lists (S, D, BW)
                for _, row in df.iterrows():
                     self.loaded_demands.append([str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2])])
                
                self.log(f"✅ DemandData.csv yüklendi: {len(self.loaded_demands)} satır")
            else:
                self.log("⚠️ DemandData.csv boş")
        except Exception as e:
            self.log(f"⚠️ DemandData.csv yüklenemedi: {e}")

    def clear_bulk_results(self):
        """ Tablodaki verileri temizler """
        if self.table_res.rowCount() > 0:
            self.table_res.setRowCount(0)
            QMessageBox.information(self, "Bilgi", "Tablo temizlendi.")
        else:
            QMessageBox.warning(self, "Uyarı", "Temizlenecek veri yok.")

    def toggle_pause_test(self):
        """Test durumunu durdur/devam et arasında değiştir"""
        self.test_paused = not self.test_paused
        if self.test_paused:
            self.btn_pause_bulk.setText("▶️ Teste Devam Et")
            self.btn_pause_bulk.setObjectName("ResumeBtn")
        else:
            self.btn_pause_bulk.setText("⏸️ Testi Durdur")
            self.btn_pause_bulk.setObjectName("PauseBtn")
        # Stil değişikliğini uygula
        self.btn_pause_bulk.setStyleSheet("")
        self.btn_pause_bulk.setStyleSheet(NEON_STYLE)

    # Algoritma Parametrelerini Ayarla (Buton click)
    def configure_bulk_algo(self):
        algo_name = self.combo_bulk_algo.currentText()
        
        if not hasattr(self, 'bulk_test_params'):
            self.bulk_test_params = {}
            
        params = {}
        result = QDialog.DialogCode.Rejected
        
        if algo_name.startswith("Genetik"):
             # Styled message box
             msg = QMessageBox(self)
             msg.setWindowTitle("Bilgi")
             msg.setText("Genetik Algoritma için ayarlanabilir ekstra parametre yok. Ana ekrandaki ağırlıklar kullanılır.")
             msg.setIcon(QMessageBox.Icon.Information)
             msg.setStyleSheet("""
                 QMessageBox { background-color: #f0f0f0; }
                 QLabel { color: black; font-size: 14px; }
                 QPushButton { background-color: #0078d7; color: white; padding: 5px 15px; border-radius: 4px; }
                 QPushButton:hover { background-color: #005a9e; }
             """)
             msg.exec()
             return
        elif algo_name.startswith("Sarsa"):
            default_bw = self.spin_main_bw.value()
            dialog = SARSAParamsDialog(self, default_bw=default_bw)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                params = dialog.get_params()
                result = QDialog.DialogCode.Accepted
        elif "Q-Learning" in algo_name:
            dialog = QLearningParamsDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                params = dialog.get_params()
                result = QDialog.DialogCode.Accepted
        elif "VNS" in algo_name:
            dialog = VNSParamsDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                params = dialog.get_params()
                result = QDialog.DialogCode.Accepted
        elif "PSO" in algo_name:
            default_bw = self.spin_main_bw.value()
            dialog = PSOParamsDialog(self, default_bw=default_bw)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                params = dialog.get_params()
                result = QDialog.DialogCode.Accepted
        elif "ACO" in algo_name:
            default_bw = self.spin_main_bw.value()
            dialog = ACOParamsDialog(self, default_bw=default_bw)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                params = dialog.get_params()
                result = QDialog.DialogCode.Accepted
        
        if result == QDialog.DialogCode.Accepted:
            self.bulk_test_params[algo_name] = params
            QMessageBox.information(self, "Başarılı", f"{algo_name} parametreleri kaydedildi.")

    # =============================================================================================
    # TOPLU DENEY (BULK TEST) ÇALIŞTIRMA MANTIĞI
    # =============================================================================================
    def run_bulk_test(self):
        """
        DemandData.csv dosyasındaki tüm senaryoları sırayla işleyerek seçilen algoritmanın
        performansını ölçer.
        
        Süreç Adımları:
        1. Algoritma ve Kayıtlı Parametreleri Al: Kullanıcının "Ayarlar" butonunda belirlediği parametreleri çeker.
        2. UI Hazırlığı: Tabloyu temizler, "Durdur" butonunu aktif eder.
        3. Senaryo Listesi: CSV verilerini (S, D, BW) bir listeye dönüştürür.
        4. Döngü (Loop): Her senaryo için;
           - Pause Kontrolü: Kullanıcı durdurduysa bekler.
           - Algoritma Çağrısı: İlgili algoritmayı çalıştırır.
           - Metrik Hesabı: Bulunan yolun başarısını, maliyetini ve süresini kaydeder.
           - Tablo Güncelleme: Sonucu anlık olarak ekrana yansıtır.
        5. Raporlama: İşlem bitince kullanıcıya onay mesajı gösterir.
        """
        # 1. Algoritma Seçimi
        algo_name = self.combo_bulk_algo.currentText()
        
        # 2. Parametreleri Al (Kaydedilmiş varsa kullan, yoksa varsayılan)
        # Kullanıcı "Ayarlar" (Gear) butonuna basıp parametre girdiyse onları kullanırız.
        # Girmediyse, algoritmanın kendi içindeki varsayılanlar kullanılır.
        if not hasattr(self, 'bulk_test_params'):
            self.bulk_test_params = {}
            
        params = self.bulk_test_params.get(algo_name, {})
        
        # Eğer parametre yoksa ve algoritmanın dialogu varsa, kullanıcıyı uyarabiliriz 
        # veya varsayılanlarla devam edebiliriz. Kullanıcı butona basmadıysa varsayılanlar geçerli olsun.

        # 3. Hazırlık ve UI Güncelleme
        self.table_res.setRowCount(0)
        self.test_paused = False
        self.btn_pause_bulk.setEnabled(True)
        self.btn_pause_bulk.setText("⏸️ Testi Durdur")
        self.btn_pause_bulk.setObjectName("PauseBtn")
        self.btn_pause_bulk.setStyleSheet(NEON_STYLE)

        # Scenarios hazırla
        scenarios = []
        if self.loaded_demands and len(self.loaded_demands) > 0:
            start_idx = 0
            if not self.loaded_demands[0][0].isdigit(): start_idx = 1 # Header kontrolü
            
            for row in self.loaded_demands[start_idx:]:
                if len(row) >= 2:
                    try:
                        s = int(row[0])
                        d = int(row[1])
                        # BW formatı: "100" veya "100 Mbps"
                        bw_str = str(row[2]).lower().replace("mbps","").strip()
                        bw = float(bw_str) if bw_str else 10.0
                        scenarios.append((s, d, bw))
                    except ValueError: continue
        
        if not scenarios:
             QMessageBox.warning(self, "Uyarı", "Test edilecek veri yok veya CSV boş/hatalı.")
             self.btn_pause_bulk.setEnabled(False)
             return

        # Ağırlıkları al (Tüm algoritmalar için ortak)
        w_delay = self.spin_delay.value()
        w_rel = self.spin_rel.value()
        w_res = self.spin_res.value()
        weights_tuple = (w_delay, w_rel, w_res) # ACO vb için
        
        # Algoritma Ön Hazırlığı (Graf Dönüşümleri)
        # ----------------------------------------------------------------
        algo_graph = None 
        
        if "VNS" in algo_name:
            algo_graph = NetworkGraph()
            for node in self.G.nodes():
                algo_graph.nodes[node] = {
                    "s_ms": self.G.nodes[node].get('proc_delay', 5),
                    "r_node": self.G.nodes[node].get('node_rel', 0.99)
                }
                algo_graph.edges.setdefault(node, {})
            for u, v in self.G.edges():
                props = {
                    "bw": self.G.edges[u, v].get('bandwidth', 100),
                    "delay": self.G.edges[u, v].get('link_delay', 10),
                    "r_link": self.G.edges[u, v].get('link_rel', 0.99)
                }
                algo_graph.edges.setdefault(u, {})[v] = props
                algo_graph.edges.setdefault(v, {})[u] = props

        elif "PSO" in algo_name:
            algo_graph = nx.Graph()
            for n in self.G.nodes():
                algo_graph.add_node(n, processing_delay=self.G.nodes[n].get('proc_delay', 0), reliability=self.G.nodes[n].get('node_rel', 1.0))
            for u, v in self.G.edges():
                e = self.G[u][v]
                algo_graph.add_edge(u, v, bandwidth=e.get('bandwidth', 1000), delay=e.get('link_delay', 10), reliability=e.get('link_rel', 1.0))
        
        elif "ACO" in algo_name:
            algo_graph = nx.Graph() # PSO ile aynı yapı genelde
            for n in self.G.nodes():
                algo_graph.add_node(n, processing_delay=self.G.nodes[n].get('proc_delay', 0), reliability=self.G.nodes[n].get('node_rel', 1.0))
            for u, v in self.G.edges():
                e = self.G[u][v]
                algo_graph.add_edge(u, v, bandwidth=e.get('bandwidth', 1000), delay=e.get('link_delay', 10), reliability=e.get('link_rel', 1.0))

        # 4. Döngü
        # ----------------------------------------------------------------
        total_tests = len(scenarios)
        self.log(f"🧪 Toplu Test Başlıyor: {algo_name}, {total_tests} senaryo")
        
        for i, (s, d, bw_req) in enumerate(scenarios):
            # Pause Check
            while self.test_paused:
                QApplication.processEvents()
                time.sleep(0.1)
            
            row_idx = self.table_res.rowCount()
            self.table_res.insertRow(row_idx)
            
            # Tabloya ilk verileri yaz (Bekliyor...)
            self.table_res.setItem(row_idx, 0, QTableWidgetItem(str(i + 1)))
            self.table_res.setItem(row_idx, 1, QTableWidgetItem(f"{s} -> {d}"))
            self.table_res.setItem(row_idx, 2, QTableWidgetItem(f"{bw_req:.0f}"))
            self.table_res.setItem(row_idx, 3, QTableWidgetItem("...")) # Durum
            
            QApplication.processEvents() # UI güncellensin
            
            path = None
            start_time = time.time()
            cost_val = 0.0
            
            # --- ALGORİTMA ÇALIŞTIR ---
            try:
                if algo_name.startswith("Genetik"):
                     path, cost_val = genetic_algorithm(
                        self.G, s, d, bw_req, 
                        w_delay, w_rel, w_res,
                        pop_size=50, generations=50, mutation_rate=0.2, # Hız için biraz azalttım
                        seed=42
                    )
                
                elif algo_name.startswith("Sarsa"):
                    # episodes sayısını bulk testte çok yüksek tutmamak iyi olabilir
                    episodes_ = params.get('episodes', 500) 
                    path, cost_val = sarsa_route(self.G, s, d, bw_req, episodes_, seed=42)
                
                elif "Q-Learning" in algo_name:
                    path, cost_val = train_q_learning(
                        self.G, s, d,
                        params.get('alpha', 0.1), params.get('gamma', 0.99), params.get('epsilon', 0.1),
                        params.get('episodes', 200), params.get('max_steps', 200),
                        w_delay, w_rel, w_res,
                        seed=42
                    )
                
                elif "VNS" in algo_name:
                    # VNS Global ayarları güncelle
                    import VNS_Algorithm_Yigit_Emre as vns_mod
                    old_iter = vns_mod.MAX_VNS_ITER
                    vns_mod.MAX_VNS_ITER = params.get('max_iterations', 10)
                    vns_mod.K_MAX = params.get('k_max', 3)
                    
                    vns_solver = VNS(algo_graph)
                    # Tek run yapıp geçiyoruz
                    path, result = vns_solver.run(s, d, seed=42)
                    if result: cost_val = result[1]["Cost"]
                    
                    # Restore
                    vns_mod.MAX_VNS_ITER = old_iter
                
                elif "PSO" in algo_name:
                    pso_solver = PSO(algo_graph, s, d, bw_req, 
                                     num_particles=params.get('num_particles', 20), 
                                     iterations=params.get('iterations', 50),
                                     seed=42)
                    path, cost_val = pso_solver.run()
                
                elif "ACO" in algo_name:
                    path, cost_val, _ = ACOSolver.solve(
                        algo_graph, s, d, weights_tuple, bw_req,
                        num_ants=params.get('num_ants', 20), 
                        num_iterations=params.get('num_iterations', 30),
                        seed=42
                    )
                
            except Exception as e:
                self.log(f"Hata (Senaryo {i+1}): {e}")
                path = None

            elapsed = (time.time() - start_time) * 1000 # ms cinsinden
            
            # --- SONUÇLARI YAZ ---
            success_str = "0%"
            cost_str = "-"
            if path and len(path) > 0:
                success_str = "100%"
                cost_str = f"{cost_val:.2f}".replace('.', ',')
                # Eğer maliyet 0 geldiyse (bazı algoritmalar döndürmeyebilir), tekrar hesapla
                if cost_val == 0:
                     try:
                        c_info = compute_path_cost(self.G, path, {'delay': w_delay, 'reliability': w_rel, 'resource': w_res})
                        cost_str = f"{c_info['total_cost']:.2f}".replace('.', ',')
                     except: pass
            
            self.table_res.setItem(row_idx, 3, QTableWidgetItem(success_str))
            
            # Renklendirme
            item_success = self.table_res.item(row_idx, 3)
            if "100" in success_str:
                item_success.setForeground(Qt.GlobalColor.green)
            else:
                item_success.setForeground(Qt.GlobalColor.red)

            self.table_res.setItem(row_idx, 4, QTableWidgetItem(cost_str)) # Ort. Maliyet
            self.table_res.setItem(row_idx, 5, QTableWidgetItem("0"))      # Std Sapma (Tek run)
            self.table_res.setItem(row_idx, 6, QTableWidgetItem(cost_str)) # Best
            self.table_res.setItem(row_idx, 7, QTableWidgetItem(cost_str)) # Worst
            self.table_res.setItem(row_idx, 8, QTableWidgetItem(f"{elapsed:.0f}"))
            
            # Tabloyu kaydır
            self.table_res.scrollToBottom()

        self.btn_pause_bulk.setEnabled(False)
        # Custom styled message box for black text
        msg = QMessageBox(self)
        msg.setWindowTitle("Tamamlandı")
        msg.setText("Tüm senaryolar test edildi.")
        msg.setIcon(QMessageBox.Icon.Information)
        # Force light background and black text
        msg.setStyleSheet("""
            QMessageBox { background-color: #f0f0f0; }
            QLabel { color: black; font-size: 14px; }
            QPushButton { background-color: #0078d7; color: white; padding: 5px 15px; border-radius: 4px; }
            QPushButton:hover { background-color: #005a9e; }
        """)
        msg.exec()

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
        """Graf oluştur: CSV varsa oradan, yoksa rastgele (Watts-Strogatz)"""
        csv_success = False
        try:
            # 1. NODE DATA OKUMA
            node_csv = os.path.join(os.path.dirname(__file__), "BSM307_317_Guz2025_TermProject_NodeData.csv")
            try:
                df_nodes = pd.read_csv(node_csv, sep=";", decimal=",")
                if df_nodes.shape[1] < 3: 
                    df_nodes = pd.read_csv(node_csv, sep=",", decimal=".")
            except:
                df_nodes = pd.read_csv(node_csv, sep=",", decimal=".")

            # 2. EDGE DATA OKUMA
            edge_csv = os.path.join(os.path.dirname(__file__), "BSM307_317_Guz2025_TermProject_EdgeData.csv")
            try:
                df_edges = pd.read_csv(edge_csv, sep=";", decimal=",")
                if df_edges.shape[1] < 5: 
                    df_edges = pd.read_csv(edge_csv, sep=",", decimal=".")
            except:
                df_edges = pd.read_csv(edge_csv, sep=",", decimal=".")
            
            if df_nodes.shape[1] >= 3 and df_edges.shape[1] >= 5:
                # Grafı sıfırdan oluştur
                self.G = nx.Graph()
                
                # Nodes ekle
                for idx, row in df_nodes.iterrows():
                    try:
                        nid = int(row.iloc[0])
                        p_delay = float(row.iloc[1])
                        n_rel = float(row.iloc[2])
                        self.G.add_node(nid, proc_delay=p_delay, node_rel=n_rel)
                    except: continue

                # Edges ekle
                for idx, row in df_edges.iterrows():
                    try:
                        u = int(row.iloc[0])
                        v = int(row.iloc[1])
                        bw = float(row.iloc[2])
                        l_delay = float(row.iloc[3])
                        l_rel = float(row.iloc[4])
                        
                        self.G.add_edge(u, v, 
                            bandwidth=bw,
                            link_delay=l_delay,
                            link_rel=l_rel
                        )
                    except: continue
                
                # Tüm edge'lere QoS tabanlı weight ekle
                for u, v in self.G.edges():
                    qos_cost = compute_edge_cost(self.G, u, v, weights={'delay': 1.0, 'reliability': 1.0, 'resource': 1.0})
                    self.G[u][v]['weight'] = qos_cost
                
                self.node_count = self.G.number_of_nodes()
                if self.node_count > 0 and self.G.number_of_edges() > 0:
                    print(f"✅ Graf CSV dosyalarından başarıyla oluşturuldu: {self.node_count} düğüm, {self.G.number_of_edges()} kenar")
                    csv_success = True
                else: 
                     print("⚠️ CSV okundu ama graf boş.")

        except Exception as e:
            print(f"⚠️ CSV okuma hatası, rastgele graf oluşturulacak: {e}")

        if not csv_success:
            print("⚠️ Otomatik (Rastgele) Graf Moduna Geçiliyor...")
            self.G = nx.watts_strogatz_graph(n=self.node_count, k=6, p=0.1, seed=42)
            
            for n in self.G.nodes():
                self.G.nodes[n]['proc_delay'] = random.uniform(1.0, 5.0)
                self.G.nodes[n]['node_rel'] = random.uniform(0.95, 0.999)
            
            for u, v in self.G.edges():
                self.G.edges[u, v]['bandwidth'] = random.uniform(100, 1000)
                self.G.edges[u, v]['link_delay'] = random.uniform(3, 15)
                self.G.edges[u, v]['link_rel'] = random.uniform(0.95, 0.999)
            
            # Tüm edge'lere QoS tabanlı weight ekle
            for u, v in self.G.edges():
                qos_cost = compute_edge_cost(self.G, u, v, weights={'delay': 1.0, 'reliability': 1.0, 'resource': 1.0})
                self.G[u][v]['weight'] = qos_cost
        
        # Layout ve UI güncellemeleri
        self.pos = nx.spring_layout(self.G, k=0.03, iterations=800, seed=42, scale=1, center=(0, 0))
        self.pos = self.compact_position(self.pos)
        nodes = [str(i) for i in range(self.node_count)]
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
        
        # Düğüm numaralarını ekle (0-indexed)
        labels = {n: str(n) for n in self.G.nodes()}
        nx.draw_networkx_labels(self.G, self.pos, labels, ax=ax, font_size=6, font_color='white', font_weight='bold')
        
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(self.G, self.pos, ax=ax, edgelist=path_edges, edge_color='#00e5ff', width=1)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[path[0]], node_color='#00ff00', node_size=120)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=[path[-1]], node_color='#ff0000', node_size=120)
            nx.draw_networkx_nodes(self.G, self.pos, ax=ax, nodelist=list(path), node_color='#00e5ff', node_size=80, alpha=0.9)
            
            # Yol üzerindeki düğümlerin numaralarını daha belirgin göster
            path_labels = {n: str(n) for n in path}
            nx.draw_networkx_labels(self.G, self.pos, path_labels, ax=ax, font_size=8, font_color='white', font_weight='bold')
        
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
        
        # Düğüm numaralarını ekle
        labels = {n: str(n) for n in self.G.nodes()}
        nx.draw_networkx_labels(self.G, self.pos, labels, ax=ax, font_size=6, font_color='white', font_weight='bold')
        
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
            
            # Yol üzerindeki düğümlerin numaralarını daha belirgin göster
            path_labels = {n: str(n) for n in path[:index+2]}
            nx.draw_networkx_labels(self.G, self.pos, path_labels, ax=ax, font_size=8, font_color='white', font_weight='bold')
            
            self.canvas.draw()
            index += 1
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(draw_next)
        self.anim_timer.start(120)

    def calculate_path(self):
        if not self.validate_weights(): return
        import time
        try:
            s = int(self.combo_source.currentText())
            d = int(self.combo_dest.currentText())
            algo = self.combo_algo.currentText()
            
            # Analiz çubuğunu güncelle - Algoritma adı
            self.lbl_analysis_algo.setText(algo.split('(')[0].strip())
            
            # Süre ölçümü başlat
            start_time = time.time()
            
            if algo.startswith("Genetik"): path = self.run_genetic(s, d)
            elif algo.startswith("Sarsa"): path = self.run_sarsa(s, d)
            elif algo.startswith("Karınca") or "ACO" in algo: path = self.run_aco(s, d)
            elif "Q-Learning" in algo or algo.startswith("Q-"): path = self.run_qlearning(s, d)
            elif "Değişken" in algo or "VNS" in algo: path = self.run_vns(s, d)
            elif "Parçacık" in algo or "PSO" in algo: path = self.run_pso(s, d)
            else: 
                # Bilinmeyen algoritma - QoS tabanlı shortest path kullan
                path = nx.shortest_path(self.G, s, d, weight='weight')
            
            # Süre ölçümü bitir
            elapsed_time = time.time() - start_time
            self.lbl_analysis_time.setText(f"{elapsed_time:.2f}s")
            
            # Yol bulunamadıysa kullanıcıyı uyar
            if not path or len(path) == 0:
                self.log(f"⚠️ UYARI: {algo} algoritması yol bulamadı!")
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("Yol Bulunamadı")
                msg.setText(f"{algo} algoritması kaynak {s}'den hedef {d}'e giden bir yol bulamadı.\n\n"
                           f"Lütfen farklı kaynak/hedef veya farklı algoritma deneyin.")
                msg.setStyleSheet("""
                    QMessageBox { background-color: #f0f0f0; }
                    QLabel { color: black; font-size: 14px; }
                    QPushButton { background-color: #0078d7; color: white; padding: 5px 15px; border-radius: 4px; }
                    QPushButton:hover { background-color: #005a9e; }
                """)
                msg.exec()
                return
            
            # Analiz çubuğunu güncelle - Yol uzunluğu
            self.lbl_analysis_path_len.setText(f"{len(path) - 1} hop ({len(path)} düğüm)")
            
            # ✅ BANDWIDTH KISITINI KONTROL ET
            min_bw = self.spin_main_bw.value()
            is_valid, invalid_edges = validate_path_bandwidth(self.G, path, min_bw)
            
            if not is_valid:
                self.log(f"⚠️ UYARI: Yol bandwidth kısıtını ihlal ediyor!")
                self.log(f"  Minimum gerekli: {min_bw} Mbps")
                self.log(f"  Geçersiz edge'ler:")
                for u, v, bw in invalid_edges:
                    self.log(f"    {u} → {v}: {bw:.2f} Mbps < {min_bw} Mbps")
                
                QMessageBox.warning(self, "Bandwidth Kısıtı İhlali",
                                  f"Bulunan yol bandwidth kısıtını sağlamıyor!\n\n"
                                  f"Minimum gerekli: {min_bw} Mbps\n"
                                  f"Geçersiz edge sayısı: {len(invalid_edges)}\n\n"
                                  f"Yol çizilebilir ancak geçerli değildir.")
            
            # ✅ TÜM ALGORİTMALAR İÇİN GERÇEK METRİKLERİ HESAPLA
            try:
                # qos_maliyet modülünden gerçek metrikleri hesapla
                metrics = compute_path_metrics(self.G, path)
                
                delay = metrics['delay']
                reliability = metrics['reliability']
                resource_cost = metrics['resource_cost']
                hop_count = metrics['hop_count']
                
                # Ağırlıklarla toplam maliyet hesapla
                w_delay = self.spin_delay.value()
                w_rel = self.spin_rel.value()
                w_res = self.spin_res.value()
                
                weights = {'delay': w_delay, 'reliability': w_rel, 'resource': w_res}
                cost_info = compute_path_cost(self.G, path, weights)
                total_cost_val = cost_info['total_cost']
                
                # GUI'ye gerçek değerleri yazdır
                self.lbl_val_delay.setText(f"{delay:.2f} ms")
                self.lbl_val_rel.setText(f"{reliability*100:.2f}%")  # Güvenilirlik yüzde olarak
                self.lbl_val_cost.setText(f"{resource_cost:.2f}")
                self.lbl_val_len.setText(str(hop_count))
                
                # Analiz çubuğunu güncelle
                self.lbl_analysis_cost.setText(f"{total_cost_val:.4f}")
                self.lbl_analysis_status.setText("✅ Başarılı")
                
                # Yolu göster (0-indexed)
                path_str = " → ".join(str(node) for node in path)
                self.lbl_analysis_path.setText(path_str)
                
                self.log(f"📊 Yol Metrikleri:")
                self.log(f"  Toplam Gecikme: {delay:.2f} ms")
                self.log(f"  Güvenilirlik: {reliability*100:.2f}%")
                self.log(f"  Kaynak Maliyeti: {resource_cost:.2f}")
                self.log(f"  Hop Sayısı: {hop_count}")
                self.log(f"  Toplam QoS Maliyeti: {total_cost_val:.4f}")
                self.log(f"  Yol: {path_str}")
                
            except Exception as e:
                self.log(f"❌ Metrik hesaplama hatası: {e}")
                import traceback
                traceback.print_exc()
                
                # Hata durumunda bile temel bilgileri göster
                self.lbl_val_len.setText(str(len(path) - 1))
                self.lbl_analysis_cost.setText("Hesaplanamadı")
                self.lbl_analysis_status.setText("⚠️ Metrik Hatası")
                self.lbl_analysis_path.setText("-")
            
            self.animate_path(path)
        except Exception as e:
            print("Hata:", e)
            self.draw_graph()

    def run_genetic(self, s, d):
        """Genetik Algoritma ile en iyi yolu bul"""
        try:
            # Parametre al
            min_bw = self.spin_main_bw.value()
            w_delay = self.spin_delay.value()
            w_rel = self.spin_rel.value()
            w_res = self.spin_res.value()
            
            self.log(f"\n{'='*60}")
            self.log(f"🧬 GENETİK ALGORİTMA BAŞLIYOR...")
            self.log(f"{'='*60}")
            self.log(f"Kaynak: {s}, Hedef: {d}")
            self.log(f"Ağırlıklar - Gecikme: {w_delay}, Güvenilirlik: {w_rel}, Kaynak: {w_res}")
            self.log(f"Min Bandwidth: {min_bw} Mbps")
            
            # Önce basit yol kontrolü - networkx ile kontrol et
            try:
                simple_path = nx.shortest_path(self.G, s, d)
                self.log(f"✓ Graf bağlantılı - NetworkX yol buldu: {len(simple_path)} düğüm")
            except:
                self.log(f"❌ HATA: Kaynak {s} ile hedef {d} arasında hiç yol yok!")
                return None
            
            # Genetik algoritmasını çalıştır
            self.log(f"⏳ Genetik algoritma çalışıyor (popülasyon: 60, nesil: 120)...")
            best_path, best_cost = genetic_algorithm(
                self.G, s, d, min_bw, 
                w_delay, w_rel, w_res,
                pop_size=60, generations=120, mutation_rate=0.2,
                seed=42
            )
            
            if best_path and len(best_path) > 1:
                self.log(f"✅ Genetik Algoritma tamamlandı! Yol bulundu: {len(best_path)} düğüm")
                self.log(f"Yol: {' → '.join(map(str, best_path[:5]))}{'...' if len(best_path) > 5 else ''}")
                self.log(f"Maliyet: {best_cost:.4f}")
                self.log(f"{'='*60}\n")
                return best_path
            else:
                self.log(f"⚠️ Genetik Algoritma yol bulamadı")
                self.log(f"Not: Graf bağlantılı ama bandwidth kısıtını sağlayan yol yok olabilir")
                self.log(f"Çözüm: Bandwidth değerini düşürmeyi deneyin (şu an: {min_bw} Mbps)")
                self.log(f"{'='*60}\n")
                return None
                
        except Exception as e:
            self.log(f"❌ Genetik Algoritma hatası: {e}")
            import traceback
            traceback.print_exc()
            self.log(f"{'='*60}\n")
            return None
    def run_sarsa(self, s, d):
        """SARSA algoritması ile en iyi yolu bul"""
        try:
            # Parametre dialogunu göster - Ana ekrandaki BW'yi varsayılan olarak gönder
            default_bw = self.spin_main_bw.value()
            dialog = SARSAParamsDialog(self, default_bw=default_bw)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.log("⚠️ Kullanıcı SARSA parametrelerini iptal etti")
                return None
            
            # Kullanıcının seçtiği parametreleri al
            params = dialog.get_params()
            alpha = params['alpha']
            gamma = params['gamma']
            epsilon = params['epsilon']
            episodes = params['episodes']
            min_bandwidth = params['min_bandwidth']
            
            self.log(f"\n{'='*60}")
            self.log(f"🎯 SARSA BAŞLIYOR...")
            self.log(f"{'='*60}")
            self.log(f"Kaynak: {s}, Hedef: {d}")
            self.log(f"\nHiperparametreler:")
            self.log(f"  Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")
            self.log(f"  Episodes: {episodes}, Min Bandwidth: {min_bandwidth} Mbps")
            
            # SARSA algoritmasını çalıştır
            # SARSA modülü kendi graf yapısını kullanıyor, bu yüzden geçici olarak
            # mevcut grafı SARSA formatına uygun hale getiriyoruz
            best_path, best_cost = sarsa_route(self.G, s, d, min_bandwidth, episodes, seed=42)
            
            if best_path:
                self.log(f"✅ SARSA tamamlandı! Yol bulundu: {len(best_path)} düğüm")
                self.log(f"Maliyet: {best_cost:.4f}")
                self.log(f"{'='*60}\n")
                return best_path
            else:
                self.log(f"⚠️ SARSA yol bulamadı")
                self.log(f"{'='*60}\n")
                return None
                
        except Exception as e:
            self.log(f"❌ SARSA hatası: {e}")
            import traceback
            traceback.print_exc()
            self.log(f"{'='*60}\n")
            return None
    def run_aco_placeholder(self, s, d):
        """ACO Algoritma placeholder - Henüz başka bir run_aco var"""
        self.log("⚠️ ACO algoritma henüz implement edilmedi")
        return None
    def run_qlearning(self, s, d):
        """Q-Learning algoritması ile en iyi yolu bul"""
        try:
            # Parametre dialogunu göster
            dialog = QLearningParamsDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                # Kullanıcı iptal etti
                print("⚠️ Kullanıcı Q-Learning parametrelerini iptal etti")
                return None
            
            # Kullanıcının seçtiği parametreleri al
            params = dialog.get_params()
            alpha = params['alpha']
            gamma = params['gamma']
            epsilon = params['epsilon']
            episodes = params['episodes']
            max_steps = params['max_steps']
            
            self.log(f"\n{'='*60}")
            self.log(f"🎓 Q-LEARNING BAŞLIYOR...")
            self.log(f"{'='*60}")
            
            # Kullanıcının arayüzden girdiği ağırlıkları al
            w_delay = self.spin_delay.value()
            w_rel = self.spin_rel.value()
            w_res = self.spin_res.value()
            
            self.log(f"Kaynak: {s}, Hedef: {d}")
            self.log(f"Ağırlıklar - Gecikme: {w_delay}, Güvenilirlik: {w_rel}, Kaynak: {w_res}")
            self.log(f"\nHiperparametreler:")
            self.log(f"  Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon}")
            self.log(f"  Episodes: {episodes}, Max Steps: {max_steps}")
            
            # Q-Learning eğitimini başlat
            best_path, best_cost = train_q_learning(
                self.G, s, d,
                alpha, gamma, epsilon,
                episodes, max_steps,
                w_delay, w_rel, w_res,
                seed=42
            )
            
            if best_path:
                self.log(f"✅ Q-Learning tamamlandı! Yol bulundu: {len(best_path)} düğüm")
                self.log(f"{'='*60}\n")
                return best_path
            else:
                self.log(f"⚠️ Q-Learning yol bulamadı")
                self.log(f"{'='*60}\n")
                return None
                
        except Exception as e:
            self.log(f"❌ Q-Learning hatası: {e}")
            self.log(f"{'='*60}\n")
            return None
    def run_vns(self, s, d):
        """VNS algoritması ile en iyi yolu bul"""
        try:
            # Parametre dialogunu göster
            dialog = VNSParamsDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                # Kullanıcı iptal etti
                self.log("⚠️ Kullanıcı VNS parametrelerini iptal etti")
                return None
            
            # Kullanıcının seçtiği parametreleri al
            params = dialog.get_params()
            max_iterations = params['max_iterations']
            k_max = params['k_max']
            test_runs = params['test_runs']
            
            self.log(f"\n{'='*60}")
            self.log(f"🔍 VNS BAŞLIYOR...")
            self.log(f"{'='*60}")
            self.log(f"Kaynak: {s}, Hedef: {d}")
            self.log(f"\nHiperparametreler:")
            self.log(f"  Max Iterations: {max_iterations}")
            self.log(f"  K Max: {k_max}")
            self.log(f"  Test Runs: {test_runs}")
            
            # VNS için NetworkGraph oluştur
            vns_graph = NetworkGraph()
            # Mevcut grafı VNS formatına dönüştür
            for node in self.G.nodes():
                vns_graph.nodes[node] = {
                    "s_ms": self.G.nodes[node].get('proc_delay', random.uniform(1, 5)),
                    "r_node": self.G.nodes[node].get('node_rel', random.uniform(0.95, 0.999))
                }
                vns_graph.edges.setdefault(node, {})
            
            for u, v in self.G.edges():
                props = {
                    "bw": self.G.edges[u, v].get('bandwidth', random.uniform(100, 1000)),
                    "delay": self.G.edges[u, v].get('link_delay', random.uniform(3, 15)),
                    "r_link": self.G.edges[u, v].get('link_rel', random.uniform(0.95, 0.999))
                }
                vns_graph.edges.setdefault(u, {})[v] = props
                vns_graph.edges.setdefault(v, {})[u] = props
            
            # VNS algoritmasını çalıştır
            vns = VNS(vns_graph)
            
            # Global değişkenleri geçici olarak güncelle
            import VNS_Algorithm_Yigit_Emre as vns_mod
            old_max_iter = vns_mod.MAX_VNS_ITER
            old_k_max = vns_mod.K_MAX
            vns_mod.MAX_VNS_ITER = max_iterations
            vns_mod.K_MAX = k_max
            
            best_path = None
            best_cost = float('inf')
            
            for run in range(test_runs):
                self.log(f"  Run {run + 1}/{test_runs}...")
                path, result = vns.run(s, d, seed=42 + run)
                if path and result:
                    cost = result[1]["Cost"]
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            
            # Global değişkenleri geri al
            vns_mod.MAX_VNS_ITER = old_max_iter
            vns_mod.K_MAX = old_k_max
            
            if best_path:
                self.log(f"✅ VNS tamamlandı! Yol bulundu: {len(best_path)} düğüm")
                self.log(f"Maliyet: {best_cost:.4f}")
                self.log(f"{'='*60}\n")
                return best_path
            else:
                self.log(f"⚠️ VNS yol bulamadı")
                self.log(f"{'='*60}\n")
                return None
                
        except Exception as e:
            self.log(f"❌ VNS hatası: {e}")
            import traceback
            traceback.print_exc()
            self.log(f"{'='*60}\n")
            return None
    def run_pso(self, s, d):
        """PSO algoritması ile en iyi yolu bul"""
        try:
            # Parametre dialogunu göster
            default_bw = self.spin_main_bw.value()
            dialog = PSOParamsDialog(self, default_bw=default_bw)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.log("⚠️ Kullanıcı PSO parametrelerini iptal etti")
                return None
            
            # Parametreleri al
            params = dialog.get_params()
            num_particles = params['num_particles']
            iterations = params['iterations']
            min_bw = params['min_bandwidth']
            
            self.log(f"\n{'='*60}")
            self.log(f"🤖 PSO BAŞLIYOR...")
            self.log(f"{'='*60}")
            self.log(f"Kaynak: {s}, Hedef: {d}")
            self.log(f"Parametreler: Particles={num_particles}, Iterations={iterations}, Min BW={min_bw}")
            
            # PSO için uyumlu graf oluştur (Attribute Mapping)
            # PSO modülü: delay, reliability (edge), processing_delay, reliability (node)
            pso_G = nx.Graph()
            
            # Nodes
            for n in self.G.nodes():
                pso_G.add_node(
                    n,
                    processing_delay=self.G.nodes[n].get('proc_delay', 0),
                    reliability=self.G.nodes[n].get('node_rel', 1.0)
                )
            
            # Edges
            for u, v in self.G.edges():
                e = self.G[u][v]
                pso_G.add_edge(
                    u, v,
                    bandwidth=e.get('bandwidth', 1000),
                    delay=e.get('link_delay', 10),
                    reliability=e.get('link_rel', 1.0)
                )
            
            # PSO algoritmasını çalıştır
            pso = PSO(pso_G, s, d, min_bw, num_particles=num_particles, iterations=iterations, seed=42)
            path, cost = pso.run()
            
            if path:
                self.last_run_cost = cost  # Maliyeti kaydet
                self.log(f"✅ PSO tamamlandı! Yol bulundu: {len(path)} düğüm")
                self.log(f"Maliyet: {cost:.4f}")
                self.log(f"{'='*60}\n")
                return path
            else:
                self.log(f"⚠️ PSO yol bulamadı")
                self.log(f"{'='*60}\n")
                return None
                
        except Exception as e:
            self.log(f"❌ PSO hatası: {e}")
            import traceback
            traceback.print_exc()
            self.log(f"{'='*60}\n")
            return None

    def run_aco(self, s, d):
        """ACO algoritması ile en iyi yolu bul"""
        try:
            # Parametre dialogunu göster
            default_bw = self.spin_main_bw.value()
            dialog = ACOParamsDialog(self, default_bw=default_bw)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                self.log("⚠️ Kullanıcı ACO parametrelerini iptal etti")
                return None
            
            # Parametreleri al
            params = dialog.get_params()
            num_ants = params['num_ants']
            iterations = params['num_iterations']
            min_bw = params['min_bandwidth']
            
            # Ağırlıkları al
            w_delay = self.spin_delay.value()
            w_rel = self.spin_rel.value()
            w_res = self.spin_res.value()
            weights = (w_delay, w_rel, w_res)
            
            self.log(f"\n{'='*60}")
            self.log(f"🐜 ACO BAŞLIYOR...")
            self.log(f"{'='*60}")
            self.log(f"Kaynak: {s}, Hedef: {d}")
            self.log(f"Parametreler: Ants={num_ants}, Iterations={iterations}, Min BW={min_bw}")
            
            # ACO için uyumlu graf oluştur (Attribute Mapping)
            # ACO modülü: delay, reliability (edge), processing_delay, reliability (node)
            aco_G = nx.Graph()
            
            # Nodes
            for n in self.G.nodes():
                aco_G.add_node(
                    n,
                    processing_delay=self.G.nodes[n].get('proc_delay', 0),
                    reliability=self.G.nodes[n].get('node_rel', 1.0)
                )
            
            # Edges
            for u, v in self.G.edges():
                e = self.G[u][v]
                aco_G.add_edge(
                    u, v,
                    bandwidth=e.get('bandwidth', 1000),
                    delay=e.get('link_delay', 10),
                    reliability=e.get('link_rel', 1.0)
                )
            
            # ACO algoritmasını çalıştır
            path, cost, duration = ACOSolver.solve(
                aco_G, s, d, weights, min_bw,
                num_ants=num_ants, num_iterations=iterations,
                seed=42
            )
            
            if path:
                self.last_run_cost = cost  # Maliyeti kaydet
                self.log(f"✅ ACO tamamlandı! Yol bulundu: {len(path)} düğüm")
                self.log(f"Maliyet: {cost:.4f}, Süre: {duration:.2f} ms")
                self.log(f"{'='*60}\n")
                return path
            else:
                self.log(f"⚠️ ACO yol bulamadı")
                self.log(f"{'='*60}\n")
                return None
                
        except Exception as e:
            self.log(f"❌ ACO hatası: {e}")
            import traceback
            traceback.print_exc()
            self.log(f"{'='*60}\n")
            return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = CyberPunkApp()
    window.show()
    sys.exit(app.exec())