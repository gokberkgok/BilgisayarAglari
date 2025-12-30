import sys
import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import QApplication
from core.network_graph import NetworkGraph
from ui.gui import MainWindow


def main():
    app = QApplication(sys.argv)

    # 🔥 GERÇEK GRAPH
    net = NetworkGraph(num_nodes=1000, probability=0.01)
    net.generate()

    window = MainWindow(net)
    window.resize(1400, 900)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
