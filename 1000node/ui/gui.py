from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem,
    QLabel, QSpinBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt

from ui.graph_canvas import GraphCanvas
from core.aco_solver import ACOSolver


class MainWindow(QWidget):
    def __init__(self, network):
        super().__init__()
        self.net = network

        # ===== GRAPH =====
        self.canvas = GraphCanvas()
        self.canvas.draw_graph(self.net.graph)

        # ===== SOURCE / DEST FORM =====
        self.source_box = QSpinBox()
        self.dest_box = QSpinBox()

        self.source_box.setRange(0, network.num_nodes - 1)
        self.dest_box.setRange(0, network.num_nodes - 1)

        # 🔧 3+ haneli sayı problemi fix
        self.source_box.setMinimumWidth(120)
        self.dest_box.setMinimumWidth(120)
        self.source_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dest_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.source_box.setValue(0)
        self.dest_box.setValue(10)

        form_group = QGroupBox("Routing Parameters")
        form_layout = QFormLayout()
        form_layout.addRow("Source Node:", self.source_box)
        form_layout.addRow("Destination Node:", self.dest_box)
        form_group.setLayout(form_layout)

        # ===== PATH INFO LABEL =====
        self.path_label = QLabel("Path: -")
        self.path_label.setWordWrap(True)
        self.path_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.path_label.setStyleSheet("""
        QLabel {
            font-weight: bold;
            padding: 8px;
            background-color: #f3f3f3;
            color: #FF0000;
        }
        """)

        # ===== QOS TABLE =====
        self.table = QTableWidget(3, 2)
        self.table.setHorizontalHeaderLabels(["QoS Metric", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(120)

        qos_group = QGroupBox("QoS Metrics")
        qos_layout = QVBoxLayout()
        qos_layout.addWidget(self.path_label)   # 👈 PATH BURADA
        qos_layout.addWidget(self.table)
        qos_group.setLayout(qos_layout)

        # ===== BUTTON =====
        self.run_btn = QPushButton("Run ACO Routing")
        self.run_btn.clicked.connect(self.run)

        # ===== RIGHT PANEL =====
        right = QVBoxLayout()
        right.addWidget(form_group)
        right.addWidget(qos_group)
        right.addWidget(self.run_btn)
        right.addStretch()

        # ===== MAIN LAYOUT =====
        layout = QHBoxLayout()
        layout.addWidget(self.canvas, 4)
        layout.addLayout(right, 1)
        self.setLayout(layout)

        self.setWindowTitle("QoS-aware Routing (ACO)")

    # ================================
    # RUN ROUTING
    # ================================
    def run(self):
        source = self.source_box.value()
        dest = self.dest_box.value()

        if source == dest:
            return

        solver = ACOSolver(self.net, source, dest)
        result = solver.solve()

        if not result:
            return

        # Update UI
        self.update_table(result.qos)
        self.update_path_label(source, dest, result.path)
        self.canvas.draw_graph(self.net.graph, result.path)

    # ================================
    # UPDATE QOS TABLE
    # ================================
    def update_table(self, qos):
        for i, (k, v) in enumerate(qos.as_dict().items()):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            self.table.setItem(i, 1, QTableWidgetItem(f"{v:.4f}"))

    # ================================
    # UPDATE PATH LABEL
    # ================================
    def update_path_label(self, source, dest, path):
        if not path or len(path) < 2:
            self.path_label.setText("Path: -")
            return

        path_str = " → ".join(map(str, path))

        text = (
            f"Source Node : {source}\n"
            f"Destination Node : {dest}\n"
            f"Path : {path_str}"
        )

        self.path_label.setText(text)
