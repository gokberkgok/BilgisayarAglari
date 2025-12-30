import networkx as nx
import random
import numpy as np
import math


class NetworkGraph:
    def __init__(self, num_nodes=1000, probability=0.4):
        self.num_nodes = num_nodes
        self.probability = probability
        self.graph = nx.DiGraph()

    def generate(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        self.graph = nx.fast_gnp_random_graph(
            self.num_nodes, self.probability, seed=seed, directed=True
        )

        for u, v in self.graph.edges():
            self.graph[u][v]["delay"] = random.uniform(2, 20)
            self.graph[u][v]["reliability"] = random.uniform(0.95, 0.9999)
            self.graph[u][v]["bandwidth"] = random.uniform(100, 10000)

        for n in self.graph.nodes():
            self.graph.nodes[n]["proc_delay"] = random.uniform(1, 5)

    def neighbors(self, node):
        return self.graph.successors(node)

    def edge(self, u, v):
        return self.graph[u][v]

    def node_delay(self, node):
        return self.graph.nodes[node]["proc_delay"]
