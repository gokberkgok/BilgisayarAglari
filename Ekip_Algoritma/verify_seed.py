
import random
import networkx as nx
import sys
import os

# Add the project directory to path
sys.path.append(r"c:\Users\oguzz\Desktop\SarsaAlgorithmTest\Ekip_Algoritma")

# Import Sarsa
try:
    from Sarsa_Algoritmasi_Oguzhan_Demirbas import sarsa_route
    print("Sarsa module imported successfully.")
except ImportError as e:
    print(f"Failed to import Sarsa: {e}")
    sys.exit(1)

def create_dummy_graph():
    G = nx.Graph()
    # Create a small random graph
    random.seed(123) # Seed for graph creation only
    for i in range(20):
        G.add_node(i)
    for i in range(20):
        for j in range(i+1, 20):
            if random.random() < 0.3:
                G.add_edge(i, j, bandwidth=random.randint(10, 100), delay=random.randint(1, 10), reliability=random.random())
    
    # Calculate weights (simplified)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0 # Dummy weight
    return G

G = create_dummy_graph()
S, D = 0, 19
min_bw = 5

print("\n--- Running Sarsa Run 1 (Seed=42) ---")
path1, cost1 = sarsa_route(G, S, D, min_bw, episodes=100, seed=42)
print(f"Path1: {path1}")

print("\n--- Running Sarsa Run 2 (Seed=42) ---")
path2, cost2 = sarsa_route(G, S, D, min_bw, episodes=100, seed=42)
print(f"Path2: {path2}")

if path1 == path2:
    print("\n✅ Verification SUCCESS: Paths are identical.")
else:
    print("\n❌ Verification FAILED: Paths different.")

print("\n--- Running Sarsa Run 3 (Seed=99) ---")
path3, cost3 = sarsa_route(G, S, D, min_bw, episodes=100, seed=99)
print(f"Path3: {path3}")

if path1 != path3:
    print("✅ Seed variation confirmed: different seed produced different path (likely).")
else:
    print("⚠️ Warning: Different seed produced same path (could happen if graph is simple).")
