# 🌐 Large-Scale Network Simulation

A high-performance network simulation framework for **QoS-oriented multi-objective routing** using advanced optimization algorithms.

## 📋 Overview

This project simulates large-scale networks (up to 1000+ nodes) and implements multiple routing algorithms to find optimal paths based on Quality of Service (QoS) metrics:

- **Delay** (latency)
- **Reliability** (packet loss)
- **Bandwidth** (throughput)

## ✨ Features

- 🎯 **Multi-Objective Optimization**: Balance multiple QoS metrics simultaneously
- 🐜 **Ant Colony Optimization (ACO)**: Bio-inspired metaheuristic algorithm
- 📊 **Integer Linear Programming (ILP)**: Exact optimization baseline
- 📈 **Pareto Analysis**: Explore trade-offs between objectives
- 🎨 **Modern GUI**: PyQt6-based dark theme interface
- 🔄 **Reproducible Results**: Seed-based random generation
- ⚡ **High Performance**: Optimized for large-scale networks

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
cd network_simulation

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# From the project root
python -m network_simulation.main

# Or directly
python network_simulation/main.py
```

## 🎮 Usage Guide

### 1. Generate Network Topology

1. Set **Nodes** (default: 1000)
2. Set **Probability** for edge creation (default: 0.4)
3. Click **GENERATE NETWORK**

The system uses the Erdős-Rényi G(n,p) model to create a random directed graph.

### 2. Configure Simulation

- **Source ID**: Starting node (0-based indexing)
- **Dest ID**: Destination node
- **Weights**: Relative importance of each metric
  - `w_delay`: Delay weight (0.0 - 1.0)
  - `w_rel`: Reliability weight (0.0 - 1.0)
  - `w_res`: Bandwidth weight (0.0 - 1.0)

### 3. Run Algorithms

- **ACO**: Fast, scalable metaheuristic (recommended for large networks)
- **ILP**: Exact solution (may timeout on large networks)
- **Pareto**: Multi-objective analysis (explores trade-off space)

### 4. Visualize Results

- **Network Visualizer**: Interactive graph with highlighted paths
- **Pareto Analysis**: 3D scatter plot of non-dominated solutions

## 🔧 Algorithm Parameters

### ACO (Ant Colony Optimization)

```python
DEFAULT_NUM_ANTS = 20           # Number of ants per iteration
DEFAULT_MAX_ITERATIONS = 50     # Maximum iterations
DEFAULT_ALPHA = 1.0             # Pheromone importance
DEFAULT_BETA = 2.0              # Heuristic importance
DEFAULT_RHO = 0.1               # Evaporation rate
DEFAULT_Q0 = 0.9                # Exploitation probability
```

### ILP (Integer Linear Programming)

```python
time_limit = 30  # Maximum solving time (seconds)
```

## 📊 Network Metrics

### Edge Metrics
- **Link Delay**: 2-20 ms (uniform distribution)
- **Reliability**: 0.95-0.9999 (uniform distribution)
- **Bandwidth**: 100-10000 Mbps (uniform distribution)

### Node Metrics
- **Processing Delay**: 1-5 ms per node

## 🏗️ Project Structure

```
network_simulation/
├── core/
│   ├── algorithms.py      # ACO, ILP, Pareto implementations
│   └── network_model.py   # Network graph generation
├── ui/
│   └── gui.py            # PyQt6 interface
├── main.py               # Application entry point
├── verify_algorithms.py  # Unit tests
└── requirements.txt      # Dependencies
```

## 🧪 Testing

Run verification tests:

```bash
python -m network_simulation.verify_algorithms
```

Tests include:
- Small graph validation (N=50)
- Pareto analysis correctness
- Large graph benchmarking (N=1000)

## 📈 Performance

| Network Size | ACO Time | ILP Time |
|--------------|----------|----------|
| 50 nodes     | ~0.5s    | ~2s      |
| 250 nodes    | ~3s      | ~15s     |
| 1000 nodes   | ~12s     | timeout  |

*Benchmarks on Intel i7-10700K, 16GB RAM*

## 🎨 UI Features

- **Dark Theme**: Modern, eye-friendly interface
- **Zoom Controls**: Navigate large networks easily
- **Real-time Logs**: Monitor algorithm progress
- **Path Highlighting**: Visual feedback for solutions
- **Degree-based Styling**: Node size/color reflects connectivity

## 🔬 Algorithm Details

### ACO Cost Function

The scalarized cost combines normalized metrics:

```
Cost = w_delay × (delay/25000) + 
       w_rel × (-log(reliability)/50) + 
       w_res × (1/bandwidth/10)
```

### Candidate List Strategy

To improve scalability:
- Select top 20 neighbors by heuristic value
- Add 5 random neighbors for exploration
- Reduces search space while maintaining solution quality

## 🐛 Troubleshooting

### "Please generate network first!"
Generate topology before running algorithms.

### "Source/Dest node must be between 0 and N-1"
Use 0-based indexing (e.g., for 1000 nodes: 0-999).

### ILP timeout
Normal for large networks. Use ACO instead.

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional algorithms (Dijkstra, A*, Genetic Algorithm)
- More QoS metrics (jitter, packet loss rate)
- Export results to CSV/JSON
- Network topology import/export

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Built with ❤️ using Python, NetworkX, and PyQt6**
