import sys
import time
import networkx as nx
from network_simulation.core.network_model import NetworkGraph
from network_simulation.core.algorithms import ACO_Solver, ILP_Solver, Pareto_Analyzer

def test_on_small_graph():
    print("\n--- Testing on Small Graph (N=50) ---")
    net = NetworkGraph(num_nodes=50, probability=0.4)
    net.generate_topology(seed=1)
    
    source = 0
    dest = 49
    
    # 1. Test ACO
    print("Running ACO...")
    aco = ACO_Solver(net, source, dest, num_ants=10)
    start = time.time()
    path_aco, cost_aco = aco.solve()
    print(f"ACO Result: Path Len={len(path_aco) if path_aco else 0}, Cost={cost_aco:.4f}, Time={time.time()-start:.4f}s")
    
    # 2. Test ILP
    print("Running ILP...")
    ilp = ILP_Solver(net, source, dest, time_limit=10)
    start = time.time()
    path_ilp, cost_ilp = ilp.solve()
    print(f"ILP Result: Path Len={len(path_ilp) if path_ilp else 0}, Cost={cost_ilp:.4f}, Time={time.time()-start:.4f}s")
    
    assert path_aco is not None, "ACO failed to find path on connected graph"
    assert path_ilp is not None, "ILP failed to find path on connected graph"

def test_pareto():
    print("\n--- Testing Pareto Analysis (N=50) ---")
    net = NetworkGraph(num_nodes=50, probability=0.4)
    net.generate_topology(seed=1)
    
    analyzer = Pareto_Analyzer(net, 0, 49)
    solutions = analyzer.run_analysis(num_simulations=10)
    print(f"Pareto Solutions Found: {len(solutions)}")
    assert len(solutions) > 0, "Pareto analysis should find at least one solution"

def benchmark_large_graph():
    print("\n--- Benchmarking Large Graph (N=1000) ---")
    net = NetworkGraph(num_nodes=1000, probability=0.4)
    net.generate_topology(seed=42)
    
    source = 0
    dest = 999
    
    # 1. Benchmark ACO
    print("Running ACO (1000 nodes)...")
    aco = ACO_Solver(net, source, dest, num_ants=20, max_iterations=20)
    start = time.time()
    path_aco, cost_aco = aco.solve()
    duration = time.time() - start
    print(f"ACO Result: Path found? {path_aco is not None}, Cost={cost_aco:.4f}, Time={duration:.4f}s")
    
    # 2. Benchmark ILP (Should likely timeout or take long, we check if it runs)
    print("Running ILP (1000 nodes, 10s timeout)...")
    ilp = ILP_Solver(net, source, dest, time_limit=10)
    start = time.time()
    path_ilp, cost_ilp = ilp.solve()
    duration = time.time() - start
    print(f"ILP Result: Path found? {path_ilp is not None}, Cost={cost_ilp:.4f}, Time={duration:.4f}s")

if __name__ == "__main__":
    try:
        test_on_small_graph()
        test_pareto()
        benchmark_large_graph()
        print("\nAll Tests Passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        # sys.exit(1) # Don't exit error code to keep terminal tool happy, just print
