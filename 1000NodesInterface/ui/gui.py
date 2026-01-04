import sys
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QLineEdit, QTabWidget, 
                             QTextEdit, QFormLayout, QGroupBox, QGridLayout, QFrame, QMessageBox,
                             QToolBar, QFileDialog, QComboBox)
from PyQt6.QtGui import QPalette, QColor, QFont, QAction
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QStyleFactory

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Import Core Modules
try:
    # Try relative imports first (when run as module)
    from ..core.network_model import NetworkGraph
    from ..core.algorithms import (
        ACO_Solver,
        ILP_Solver,
        Pareto_Analyzer
    )
    from ..core.genetic_algorithm import GA_Solver
    from ..core.qlearning_algorithm import QLearning_Solver
    from ..core.sarsa_algorithm import SARSA_Solver
    from ..core.pso_algorithm import PSO_Solver
    from ..core.vns_algorithm import VNS_Solver
except ImportError:
    # Fall back to direct imports (when run directly)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.network_model import NetworkGraph
    from core.algorithms import (
        ACO_Solver,
        ILP_Solver,
        Pareto_Analyzer
    )
    from core.genetic_algorithm import GA_Solver
    from core.qlearning_algorithm import QLearning_Solver
    from core.sarsa_algorithm import SARSA_Solver
    from core.pso_algorithm import PSO_Solver
    from core.vns_algorithm import VNS_Solver
import json
import time

# ---------------------------------------------------------
# MODERN THEME CONFIGURATION
# ---------------------------------------------------------
COLOR_BG = "#020205" # Deep Space Black
COLOR_FG = "#f0f0f0"
COLOR_ACCENT = "#38bdf8"
COLOR_BTN = "#1e293b"
COLOR_BTN_HOVER = "#334155"
COLOR_BORDER = "#475569"

def apply_modern_theme(app):
    """Applies a modern Dark + Neon theme to the PyQt Application."""
    app.setStyle("Fusion")
    
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLOR_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLOR_FG))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLOR_BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLOR_BTN))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLOR_FG))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLOR_FG))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLOR_FG))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLOR_BTN))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLOR_FG))
    palette.setColor(QPalette.ColorRole.Link, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLOR_ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("black"))
    
    app.setPalette(palette)
    
    # Global Stylesheet
    app.setStyleSheet(f"""
        QMainWindow {{ background-color: {COLOR_BG}; }}
        QWidget {{ color: {COLOR_FG}; font-family: 'Segoe UI', sans-serif; font-size: 10pt; }}
        
        /* Groups */
        QGroupBox {{ 
            border: 2px solid {COLOR_BORDER}; 
            border-radius: 8px; 
            margin-top: 10px; 
            font-weight: bold;
            color: {COLOR_ACCENT};
            background-color: transparent;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
        
        /* Inputs */
        QLineEdit {{ 
            background-color: #1e293b; 
            border: 1px solid {COLOR_BORDER}; 
            border-radius: 6px; 
            padding: 6px; 
            color: {COLOR_FG};
        }}
        QLineEdit:focus {{ border: 1px solid {COLOR_ACCENT}; background-color: #334155; }}
        
        /* Buttons */
        QPushButton {{
            background-color: {COLOR_BTN};
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            padding: 8px 15px;
            font-weight: bold;
        }}
        QPushButton:hover {{ 
            background-color: {COLOR_BTN_HOVER}; 
            border: 1px solid {COLOR_ACCENT};
            color: {COLOR_ACCENT};
        }}
        QPushButton:pressed {{ 
            background-color: {COLOR_ACCENT}; 
            color: #000000; 
        }}
        QPushButton:disabled {{ background-color: #0f172a; color: #475569; border: 1px solid #1e293b; }}
        
        /* Tabs */
        QTabWidget::pane {{ border: 1px solid {COLOR_BORDER}; background: {COLOR_BG}; }}
        QTabBar::tab {{ 
            background: {COLOR_BTN}; 
            color: #94a3b8; 
            padding: 10px 20px; 
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected {{ 
            background: {COLOR_BTN_HOVER}; 
            color: {COLOR_ACCENT}; 
            border-bottom: 2px solid {COLOR_ACCENT}; 
        }}
        
        /* Text Edit */
        QTextEdit {{ background-color: #020617; color: #e2e8f0; border: 1px solid {COLOR_BORDER}; font-family: Consolas; }}
    """)

# ---------------------------------------------------------
# WORKER
# ---------------------------------------------------------
class Worker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self.func = func

    def run(self):
        try:
            result = self.func()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ---------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Large-Scale Network Simulation (Modern UI)")
        self.setGeometry(100, 100, 1300, 850)
        
        # Data
        self.network = None
        self.network_generated = False
        self.net_pos = {}
        self.ax = None # Handle for plotting
        
        # UI
        self.init_ui()
        
    def init_ui(self):
        # --- TOOLBAR ---
        toolbar = QToolBar("View Controls")
        self.addToolBar(toolbar)
        
        btn_fit = QAction("Fit to Screen", self)
        btn_fit.triggered.connect(self.reset_view)
        toolbar.addAction(btn_fit)
        
        toolbar.addSeparator()
        
        btn_zoom_in = QAction("Zoom In (+)", self)
        btn_zoom_in.triggered.connect(lambda: self.zoom_camera(0.8)) # Scalar < 1 zooms in
        toolbar.addAction(btn_zoom_in)
        
        btn_zoom_out = QAction("Zoom Out (-)", self)
        btn_zoom_out.triggered.connect(lambda: self.zoom_camera(1.2)) # Scalar > 1 zooms out
        toolbar.addAction(btn_zoom_out)
        
        toolbar.addSeparator()
        
        # Pareto Analysis Action (Prominent in toolbar)
        btn_pareto_action = QAction("🔍 RUN PARETO ANALYSIS", self)
        btn_pareto_action.triggered.connect(self.run_pareto)
        toolbar.addAction(btn_pareto_action)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20)
        
        # --- LEFT PANEL (Controls) ---
        left_panel = QFrame()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        
        # 1. Generation
        grp_gen = QGroupBox("1. Network Topology")
        layout_gen = QFormLayout()
        self.input_nodes = QLineEdit("1000")
        layout_gen.addRow("Nodes:", self.input_nodes)
        self.input_prob = QLineEdit("0.4")
        layout_gen.addRow("Probability:", self.input_prob)
        
        self.btn_generate = QPushButton("GENERATE NETWORK")
        self.btn_generate.setFixedHeight(40)
        self.btn_generate.clicked.connect(self.generate_network)
        layout_gen.addRow(self.btn_generate)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #888;")
        layout_gen.addRow(self.lbl_status)
        
        grp_gen.setLayout(layout_gen)
        left_layout.addWidget(grp_gen)
        
        # 2. Config
        grp_conf = QGroupBox("2. Simulation Config")
        layout_conf = QFormLayout()
        
        self.input_source = QLineEdit("0")
        layout_conf.addRow("Source ID:", self.input_source)
        self.input_dest = QLineEdit("999")
        layout_conf.addRow("Dest ID:", self.input_dest)
        
        layout_conf.addRow(QLabel("Weights (Delay / Rel / BW):"))
        w_layout = QHBoxLayout()
        self.input_w_delay = QLineEdit("0.33")
        self.input_w_rel = QLineEdit("0.33")
        self.input_w_res = QLineEdit("0.33")
        w_layout.addWidget(self.input_w_delay)
        w_layout.addWidget(self.input_w_rel)
        w_layout.addWidget(self.input_w_res)
        layout_conf.addRow(w_layout)
        
        grp_conf.setLayout(layout_conf)
        left_layout.addWidget(grp_conf)
        
        # 3. Algorithm Selection
        grp_algo = QGroupBox("3. Algorithm Selection")
        layout_algo = QVBoxLayout()
        layout_algo.setSpacing(10)
        
        # Algorithm Dropdown
        algo_label = QLabel("Select Algorithm:")
        layout_algo.addWidget(algo_label)
        
        self.algo_combo = QComboBox()
        self.algo_combo.addItems([
            "ACO (Ant Colony Optimization)",
            "ILP (Integer Linear Programming)",
            "Genetic Algorithm",
            "PSO (Particle Swarm Optimization)",
            "Q-Learning (Reinforcement Learning)",
            "SARSA (Reinforcement Learning)",
            "VNS (Variable Neighborhood Search)"
        ])
        self.algo_combo.setFixedHeight(40)
        layout_algo.addWidget(self.algo_combo)
        
        # Run Button
        self.btn_run_algo = QPushButton("RUN SELECTED ALGORITHM")
        self.btn_run_algo.setFixedHeight(50)
        self.btn_run_algo.clicked.connect(self.run_selected_algorithm)
        layout_algo.addWidget(self.btn_run_algo)
        
        grp_algo.setLayout(layout_algo)
        left_layout.addWidget(grp_algo)
        
        # 4. File Operations
        grp_file = QGroupBox("4. File Operations")
        layout_file = QGridLayout()
        layout_file.setSpacing(10)
        
        self.btn_save_net = QPushButton("💾 Save Network")
        self.btn_save_net.clicked.connect(self.save_network)
        layout_file.addWidget(self.btn_save_net, 0, 0)
        
        self.btn_load_net = QPushButton("📂 Load Network")
        self.btn_load_net.clicked.connect(self.load_network)
        layout_file.addWidget(self.btn_load_net, 0, 1)
        
        self.btn_export = QPushButton("📊 Export Results")
        self.btn_export.clicked.connect(self.export_results)
        layout_file.addWidget(self.btn_export, 1, 0, 1, 2)
        
        grp_file.setLayout(layout_file)
        left_layout.addWidget(grp_file)
        
        # 5. Logs
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        left_layout.addWidget(QLabel("Process Log:"))
        left_layout.addWidget(self.txt_output)
        
        main_layout.addWidget(left_panel)
        
        # --- RIGHT PANEL (Tabs) ---
        self.tabs = QTabWidget()
        
        # Visualizer Tab
        self.figure_net = plt.figure(facecolor=COLOR_BG)
        self.canvas_net = FigureCanvas(self.figure_net)
        self.tabs.addTab(self.canvas_net, "Network Visualizer")
        
        # Pareto Analysis Tab (Split: 3D Plot + Details)
        pareto_tab = QWidget()
        pareto_layout = QHBoxLayout(pareto_tab)
        
        # Left: 3D Visualization
        self.figure_3d = plt.figure(facecolor=COLOR_BG)
        self.canvas_3d = FigureCanvas(self.figure_3d)
        pareto_layout.addWidget(self.canvas_3d, stretch=2)
        
        # Right: Solution Details
        details_panel = QWidget()
        details_layout = QVBoxLayout(details_panel)
        details_layout.addWidget(QLabel("📊 Pareto Solutions Details"))
        
        self.pareto_details = QTextEdit()
        self.pareto_details.setReadOnly(True)
        self.pareto_details.setStyleSheet("""
            QTextEdit {
                background-color: #020617;
                color: #e2e8f0;
                border: 1px solid #475569;
                font-family: 'Consolas', monospace;
                font-size: 9pt;
            }
        """)
        details_layout.addWidget(self.pareto_details)
        pareto_layout.addWidget(details_panel, stretch=1)
        
        self.tabs.addTab(pareto_tab, "Pareto Analysis")
        
        main_layout.addWidget(self.tabs)
        
    # ----------------------------------------------------------------
    # LOGIC
    # ----------------------------------------------------------------
    
    def generate_network(self):
        try:
            n = int(self.input_nodes.text())
            p = float(self.input_prob.text())
            self.lbl_status.setText("Generating Topology...")
            QApplication.processEvents()
            
            self.network = NetworkGraph(num_nodes=n, probability=p)
            self.network.generate_topology(seed=42)
            self.network_generated = True
            
            self.txt_output.append(f"Network Generated: {n} Nodes, {self.network.graph.number_of_edges()} Edges")
            self.lbl_status.setText("Topology Ready")
            
            # Layout
            self.txt_output.append("Computing layout (expanded)...")
            import math
            # k=2.5/sqrt(n) for separation
            # scale=2.0 to expand the bounding box of the graph
            k_val = 2.5 / math.sqrt(n) 
            self.net_pos = nx.spring_layout(self.network.graph, k=k_val, iterations=50, seed=42, scale=5.0)
            
            self.draw_graph()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.lbl_status.setText("Error")

    def draw_graph(self, path=None, color_path="#00e676"):
        """Draws the graph with degree-based sizing and Modern colors."""
        self.figure_net.clear()
        self.ax = self.figure_net.add_subplot(111)
        self.ax.set_facecolor(COLOR_BG)
        
        if not self.net_pos: return
        
        G = self.network.graph
        
        # --- Draw Starfield Background ---
        if self.net_pos:
            all_x = [p[0] for p in self.net_pos.values()]
            all_y = [p[1] for p in self.net_pos.values()]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            w, h = max_x - min_x, max_y - min_y
            padding = 0.5
            
            # Generate 150 random stars
            sx = [random.uniform(min_x - padding, max_x + padding) for _ in range(150)]
            sy = [random.uniform(min_y - padding, max_y + padding) for _ in range(150)]
            self.ax.scatter(sx, sy, s=1, c='white', alpha=0.3, zorder=0)

        # --- Calculate Node Metrics for Styling ---
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        
        # Lists for drawing
        node_colors = []
        node_sizes = []
        
        for n in G.nodes():
            deg = degrees[n]
            
            # Size:
            size = 10 + (deg / max_deg) * 80
            node_sizes.append(size)
            
            # Color Palette:
            # Low: #94a3b8 (Slate 400)
            # Med: #38bdf8 (Sky 400)
            # High: #e879f9 (Fuchsia 400)
            if deg > max_deg * 0.6: 
                node_colors.append('#e879f9')  # High degree
            elif deg > max_deg * 0.3:
                node_colors.append('#38bdf8')  # Med degree
            else:
                node_colors.append('#94a3b8')  # Low degree

        # --- Draw Edges ---
        # Draw edges for nodes with higher than average degree to keep it clean but connected
        if len(G.edges()) < 5000:
             nx.draw_networkx_edges(G, self.net_pos, edge_color='#475569', width=0.5, alpha=0.3, ax=self.ax, arrows=True)
        else:
             mean_deg = sum(degrees.values()) / len(degrees)
             significant_nodes = {n for n, d in degrees.items() if d > mean_deg}
             edges_to_draw = [(u, v) for u, v in G.edges() if u in significant_nodes or v in significant_nodes]
             
             if len(edges_to_draw) > 15000:
                 edges_to_draw = edges_to_draw[:15000]
                 
             nx.draw_networkx_edges(G, self.net_pos, edgelist=edges_to_draw,
                                    edge_color='#475569', width=0.4, alpha=0.4, ax=self.ax, arrows=False)

        # --- Draw Nodes ---
        # Increased linewidths (borders) as requested
        nx.draw_networkx_nodes(G, self.net_pos, node_size=node_sizes, node_color=node_colors, linewidths=2.0, edgecolors='white', ax=self.ax)
        
        # --- Draw Path (Overlay) ---
        if path:
            path_edges = list(zip(path, path[1:]))
            nx.draw_networkx_nodes(G, self.net_pos, nodelist=path,
                                   node_size=60, node_color=COLOR_ACCENT, ax=self.ax)
            nx.draw_networkx_edges(G, self.net_pos, edgelist=path_edges,
                                   edge_color=COLOR_ACCENT, width=3.0, alpha=1.0, ax=self.ax)
            self.ax.set_title(f"Path Highlighted ({len(path)} hops)", color=COLOR_FG)
        else:
            self.ax.set_title("Network Topology", color=COLOR_FG)

        # --- FORCE DRAW NODE IDS (DEBUG / FINAL) ---
        # Remove conditional logic: force-draw all IDs for verification and debugging.
        print("DRAWING NODE LABELS")
        for node, (x, y) in self.net_pos.items():
            self.ax.text(
                x,
                y,
                str(node),  # Fixed: Use 0-based indexing consistently
                color="yellow",
                fontsize=7,
                ha="center",
                va="center",
                zorder=100
            )

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Add visible borders (Spines) to the graph frame as requested
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLOR_BORDER)
            spine.set_linewidth(2)
            
        self.canvas_net.draw()

    def reset_view(self):
        """Fits the graph to the screen."""
        if self.ax:
            self.ax.autoscale()
            self.canvas_net.draw()
            
    def zoom_camera(self, factor):
        """Zooms in/out by modifying axis limits."""
        if not self.ax: return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Calculate new dimensions
        w = (xlim[1] - xlim[0]) * factor
        h = (ylim[1] - ylim[0]) * factor
        
        # Center point
        cx = (xlim[1] + xlim[0]) / 2
        cy = (ylim[1] + ylim[0]) / 2
        
        self.ax.set_xlim([cx - w/2, cx + w/2])
        self.ax.set_ylim([cy - h/2, cy + h/2])
        self.canvas_net.draw()

    def is_zoomed_in(self):
        """Heuristic: return True when the current view is significantly smaller than the full layout."""
        if not self.ax or not self.net_pos:
            return False
        try:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            full_x = [p[0] for p in self.net_pos.values()]
            full_y = [p[1] for p in self.net_pos.values()]
            full_w = max(full_x) - min(full_x)
            full_h = max(full_y) - min(full_y)
            view_w = xlim[1] - xlim[0]
            view_h = ylim[1] - ylim[0]
            # Consider zoomed in when view is less than 60% of full extent
            return (view_w < full_w * 0.6) or (view_h < full_h * 0.6)
        except Exception:
            return False

    def get_weights(self):
        """Parse weight values from input fields with proper error handling."""
        try:
            w_delay = float(self.input_w_delay.text())
            w_rel = float(self.input_w_rel.text())
            w_res = float(self.input_w_res.text())
            
            # Validate weights are positive
            if w_delay < 0 or w_rel < 0 or w_res < 0:
                raise ValueError("Weights must be positive")
            
            return w_delay, w_rel, w_res
        except ValueError as e:
            self.txt_output.append(f"⚠️ Invalid weight values: {e}. Using defaults (0.33, 0.33, 0.33)")
            return 0.33, 0.33, 0.33

    def run_selected_algorithm(self):
        """Run the algorithm selected from dropdown."""
        selected = self.algo_combo.currentIndex()
        
        algorithm_map = {
            0: (ACO_Solver, "ACO"),
            1: (ILP_Solver, "ILP"),
            2: (GA_Solver, "Genetic"),
            3: (PSO_Solver, "PSO"),
            4: (QLearning_Solver, "Q-Learning"),
            5: (SARSA_Solver, "SARSA"),
            6: (VNS_Solver, "VNS")
        }
        
        solver_cls, name = algorithm_map[selected]
        self.run_solver(solver_cls, name)
    
    def run_aco(self):
        self.run_solver(ACO_Solver, "ACO")
        
    def run_ilp(self):
        self.run_solver(ILP_Solver, "ILP")
    
    def run_genetic(self):
        self.run_solver(GA_Solver, "Genetic")
    
    def run_qlearning(self):
        self.run_solver(QLearning_Solver, "Q-Learning")
    
    def run_sarsa(self):
        self.run_solver(SARSA_Solver, "SARSA")
    
    def run_pso(self):
        self.run_solver(PSO_Solver, "PSO")
    
    def run_vns(self):
        self.run_solver(VNS_Solver, "VNS")
        
    def run_solver(self, solver_cls, name):
        """Run a solver algorithm with proper validation."""
        if not self.network_generated:
            self.txt_output.append("⚠️ Please generate network first!")
            return
        
        try:
            s = int(self.input_source.text())
            d = int(self.input_dest.text())
            
            # Validate node IDs
            if s < 0 or s >= self.network.num_nodes:
                raise ValueError(f"Source node must be between 0 and {self.network.num_nodes-1}")
            if d < 0 or d >= self.network.num_nodes:
                raise ValueError(f"Destination node must be between 0 and {self.network.num_nodes-1}")
            if s == d:
                raise ValueError("Source and destination must be different")
            
            w = self.get_weights()
        except ValueError as e:
            self.txt_output.append(f"❌ Input Error: {e}")
            return
        
        self.txt_output.append(f"Running {name}...")
        solver = solver_cls(self.network, s, d, *w)
        
        self.worker = Worker(solver.solve)
        self.worker.finished.connect(lambda res: self.on_solver_done(res, name))
        self.worker.error.connect(lambda e: self.txt_output.append(f"Error: {e}"))
        self.worker.start()
        
    def on_solver_done(self, result, name):
        path, cost = result
        if path:
            self.txt_output.append(f"{name} Path Found! Cost: {cost:.4f}")
            self.draw_graph(path, color_path=COLOR_ACCENT)
        else:
            self.txt_output.append(f"{name} failed.")

    def run_pareto(self):
        if not self.network_generated: return
        try:
            s, d = int(self.input_source.text()), int(self.input_dest.text())
        except: return
        
        self.txt_output.append("Running Pareto Analysis (10 simulations, optimized)...")
        analyzer = Pareto_Analyzer(self.network, s, d)
        
        self.worker = Worker(lambda: analyzer.run_analysis(10))
        self.worker.finished.connect(self.on_pareto_done)
        self.worker.start()
        
    def on_pareto_done(self, solutions):
        self.txt_output.append(f"Pareto Analizi: {len(solutions)} çözüm bulundu.")
        
        # Update 3D Plot
        self.figure_3d.clear()
        ax = self.figure_3d.add_subplot(111, projection='3d')
        ax.set_facecolor(COLOR_BG)
        
        if solutions:
            xs = [s['metrics'][0] for s in solutions]
            ys = [s['metrics'][1] for s in solutions]
            zs = [s['metrics'][2] for s in solutions]
            
            ax.scatter(xs, ys, zs, c=COLOR_ACCENT, marker='o', s=40)
            ax.set_xlabel('Gecikme', color='white')
            ax.set_ylabel('Güvenilirlik Maliyeti', color='white')
            ax.set_zlabel('Bant Genişliği Maliyeti', color='white')
            ax.tick_params(colors='white')
            
            # Populate Details Panel
            details_text = "╔═══════════════════════════════════════════════════════════╗\n"
            details_text += "║         PARETO OPTİMAL ÇÖZÜMLER ANALİZİ                  ║\n"
            details_text += "╚═══════════════════════════════════════════════════════════╝\n\n"
            details_text += f"Bulunan Toplam Çözüm: {len(solutions)}\n"
            details_text += f"Kaynak → Hedef: {self.input_source.text()} → {self.input_dest.text()}\n\n"
            details_text += "─" * 60 + "\n\n"
            
            for idx, sol in enumerate(solutions, 1):
                path = sol['path']
                metrics = sol['metrics']
                delay, rel_cost, bw_cost = metrics
                
                # Calculate actual reliability (inverse of cost)
                actual_reliability = 1.0 / (1.0 + rel_cost)
                
                details_text += f"Çözüm #{idx}\n"
                details_text += "─" * 40 + "\n"
                details_text += f"  📍 Yol Uzunluğu:           {len(path)} atlama\n"
                details_text += f"  ⏱️  Toplam Gecikme:         {delay:.2f} ms\n"
                details_text += f"  🔒 Güvenilirlik Maliyeti:  {rel_cost:.4f}\n"
                details_text += f"  📶 Bant Genişliği Maliyeti: {bw_cost:.6f}\n"
                details_text += f"  🛤️  Yol: {' → '.join(map(str, path))}\n\n"
            
            details_text += "─" * 60 + "\n"
            details_text += "💡 İPUCU: Tüm metrikler için düşük değerler daha iyidir.\n"
            details_text += "   Her çözüm, gecikme, güvenilirlik ve bant genişliği\n"
            details_text += "   arasında farklı bir denge (trade-off) temsil eder.\n"
            
            self.pareto_details.setText(details_text)
        else:
            self.pareto_details.setText("Pareto çözümü bulunamadı.\nAğ parametrelerini veya kaynak/hedef düğümlerini ayarlamayı deneyin.")
            
        self.canvas_3d.draw()
    
    def save_network(self):
        """Save current network to file."""
        if not self.network_generated:
            self.txt_output.append("⚠️ No network to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Network", "", "Network Files (*.net);;All Files (*)"
        )
        
        if filename:
            try:
                self.network.save_network(filename)
                self.txt_output.append(f"✅ Network saved to {filename}")
            except Exception as e:
                self.txt_output.append(f"❌ Error saving network: {e}")
    
    def load_network(self):
        """Load network from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Network", "", "Network Files (*.net);;All Files (*)"
        )
        
        if filename:
            try:
                self.network = NetworkGraph()
                self.network.load_network(filename)
                self.network_generated = True
                
                # Update UI
                self.input_nodes.setText(str(self.network.num_nodes))
                self.input_prob.setText(str(self.network.probability))
                self.input_dest.setText(str(self.network.num_nodes - 1))
                
                # Compute layout
                self.txt_output.append("Computing layout...")
                import math
                k_val = 2.5 / math.sqrt(self.network.num_nodes)
                self.net_pos = nx.spring_layout(self.network.graph, k=k_val, iterations=50, seed=42, scale=5.0)
                
                self.draw_graph()
                self.txt_output.append(f"✅ Network loaded from {filename}")
                self.lbl_status.setText("Network Loaded")
            except Exception as e:
                self.txt_output.append(f"❌ Error loading network: {e}")
    
    def export_results(self):
        """Export current results to JSON file."""
        if not self.network_generated:
            self.txt_output.append("⚠️ No network to export!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results.json", "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                results = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "network": {
                        "num_nodes": self.network.num_nodes,
                        "num_edges": self.network.graph.number_of_edges(),
                        "probability": self.network.probability
                    },
                    "configuration": {
                        "source": int(self.input_source.text()),
                        "destination": int(self.input_dest.text()),
                        "weights": {
                            "delay": float(self.input_w_delay.text()),
                            "reliability": float(self.input_w_rel.text()),
                            "bandwidth": float(self.input_w_res.text())
                        }
                    }
                }
                
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2)
                
                self.txt_output.append(f"✅ Results exported to {filename}")
            except Exception as e:
                self.txt_output.append(f"❌ Error exporting results: {e}")


def main():
    app = QApplication(sys.argv)
    apply_modern_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
