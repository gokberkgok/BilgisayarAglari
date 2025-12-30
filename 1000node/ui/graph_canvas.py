from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import networkx as nx


class GraphCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        self.pos = None  # Layout cache

    def draw_graph(self, G, path=None):
        """
        Draws the network graph.
        If path is provided, highlights the path in red.
        """
        self.ax.clear()

        # --- Layout (cached) ---
        if self.pos is None or len(self.pos) != G.number_of_nodes():
            self.pos = nx.spring_layout(G, seed=42, k=0.15)

        # --- Draw nodes ---
        nx.draw_networkx_nodes(
            G,
            self.pos,
            node_size=15,
            node_color="lightgray",
            ax=self.ax
        )

        # --- Draw edges ---
        nx.draw_networkx_edges(
            G,
            self.pos,
            width=0.3,
            alpha=0.4,
            ax=self.ax
        )

        # --- Highlight path ---
        if path and len(path) > 1:
            path_edges = list(zip(path[:-1], path[1:]))

            nx.draw_networkx_nodes(
                G,
                self.pos,
                nodelist=path,
                node_color="red",
                node_size=40,
                ax=self.ax
            )

            nx.draw_networkx_edges(
                G,
                self.pos,
                edgelist=path_edges,
                edge_color="red",
                width=2,
                ax=self.ax
            )

        # --- Node labels (small font for 1000 nodes) ---
        nx.draw_networkx_labels(
            G,
            self.pos,
            font_size=5,
            font_color="black",
            ax=self.ax
        )

        self.ax.set_title("1000-Node QoS Routing Graph")
        self.ax.axis("off")

        # IMPORTANT: use draw_idle(), not draw()
        self.draw_idle()
