import networkx as nx
import random
import math
import numpy as np
import pickle
import json
from typing import Optional, Dict, Any

class NetworkGraph:
    """
    Represents the large-scale network topology with 1000 nodes.
    Uses Erdős-Rényi model G(n, p) with p=0.4.
    """

    def __init__(self, num_nodes: int = 1000, probability: float = 0.4):
        self.num_nodes = num_nodes
        self.probability = probability
        self.adj_list: dict = {}  # Adjacency List: {node_id: {neighbor_id: {metrics}}}
        self.node_delays: dict = {} # {node_id: delay_val}
        self.graph = None # NetworkX graph for visualization/reference if needed, but mainly we use adj_list

    def generate_topology(self, seed: int = None) -> None:
        """Generates the network topology and assigns random QoS metrics."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"Generating G({self.num_nodes}, {self.probability}) topology... This might take a moment.")
        
        # We use NetworkX's fast generator for the structure
        # fast_gnp_random_graph is O(n+m), efficient for our scale
        self.graph = nx.fast_gnp_random_graph(self.num_nodes, self.probability, seed=seed, directed=True)
        
        # Initialize Adjacency List and Assign Metrics
        self.adj_list = {i: {} for i in range(self.num_nodes)}
        self.node_delays = {i: random.uniform(1, 5) for i in range(self.num_nodes)} # Node processing delay 1-5ms

        # Iterate over edges and assign weights
        # Metrics:
        # 1. Link Delay: 2-20 ms
        # 2. Reliability: 0.95 - 0.9999 (log cost will be applied later during pathfinding)
        # 3. Bandwidth: 100 - 10000 Mbps
        
        for u, v in self.graph.edges():
            link_delay = random.uniform(2, 20)
            reliability = random.uniform(0.95, 0.9999)
            bandwidth = random.uniform(100, 10000)
            
            # Storing in our efficient dictionary structure
            self.adj_list[u][v] = {
                'link_delay': link_delay,
                'reliability': reliability,
                'bandwidth': bandwidth
            }
            
            # Also update NetworkX graph attributes for consistency/drawing logic later
            self.graph[u][v]['link_delay'] = link_delay
            self.graph[u][v]['reliability'] = reliability
            self.graph[u][v]['bandwidth'] = bandwidth
            
        print(f"Topology generation complete. Nodes: {self.num_nodes}, Edges: {self.graph.number_of_edges()}")

    def get_neighbors(self, node: int) -> dict:
        """Returns neighbors of a node."""
        return self.adj_list.get(node, {})

    def get_edge_data(self, u: int, v: int) -> dict:
        """Returns metrics for edge u->v."""
        return self.adj_list.get(u, {}).get(v, None)

    def get_node_delay(self, node: int) -> float:
        """Returns processing delay for a node."""
        return self.node_delays.get(node, 0)
    
    def save_network(self, filename: str) -> None:
        """Save network topology to file using pickle."""
        data = {
            'num_nodes': self.num_nodes,
            'probability': self.probability,
            'adj_list': self.adj_list,
            'node_delays': self.node_delays,
            'edges': list(self.graph.edges()) if self.graph else []
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Network saved to {filename}")
    
    def load_network(self, filename: str) -> None:
        """Load network topology from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        self.num_nodes = data['num_nodes']
        self.probability = data['probability']
        self.adj_list = data['adj_list']
        self.node_delays = data['node_delays']
        
        # Reconstruct NetworkX graph
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.num_nodes))
        for u in self.adj_list:
            for v, metrics in self.adj_list[u].items():
                self.graph.add_edge(u, v, **metrics)
        
        print(f"Network loaded from {filename}")
    
    def export_metrics(self, filename: str = "network_metrics.json") -> None:
        """Export network metrics to JSON file."""
        metrics = {
            'num_nodes': self.num_nodes,
            'num_edges': self.graph.number_of_edges() if self.graph else 0,
            'probability': self.probability,
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.num_nodes if self.graph else 0,
            'density': nx.density(self.graph) if self.graph else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Network metrics exported to {filename}")
