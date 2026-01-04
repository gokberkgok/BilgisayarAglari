"""
Network Simulation Package

A high-performance framework for QoS-oriented multi-objective routing
in large-scale networks using ACO, ILP, and Pareto analysis.

Modules:
    - core.network_model: Network topology generation
    - core.algorithms: Routing optimization algorithms
    - ui.gui: PyQt6 graphical interface
"""

__version__ = "1.0.0"
__author__ = "Network Simulation Team"

from .core.network_model import NetworkGraph
from .core.algorithms import ACO_Solver, ILP_Solver, Pareto_Analyzer

__all__ = [
    "NetworkGraph",
    "ACO_Solver", 
    "ILP_Solver",
    "Pareto_Analyzer"
]
