"""
MetaTrajectory: Optimal Transport-based Metabolic Trajectory Analysis

A mathematically rigorous framework for analyzing metabolic state transitions
in single-cell RNA-seq data using optimal transport theory.

Core Modules:
- knowledge_base: Metabolic pathway and network knowledge (77 KEGG pathways)
- state_space: Metabolic state space construction
- optimal_transport: OT-based trajectory inference
- integration: RNA velocity integration with uncertainty propagation
- drivers: Driver gene identification with rigorous statistics
- scoring: Advanced pathway scoring (AUCell, VISION, ssGSEA, weighted)
- statistics: Effect sizes, multiple testing, uncertainty quantification
- network_distance: Network-topology aware distance metrics
- parameter_selection: Adaptive parameter selection
- benchmark: Systematic benchmarking and validation

Key Features:
1. Mathematical rigor: Absorbing Markov chain for fate probabilities
2. Network-constrained: Biologically meaningful distance metrics
3. Multiple scoring methods: AUCell, VISION, ssGSEA, weighted
4. Rigorous statistics: Cohen's d, Hedges' g, global FDR
5. Adaptive parameters: Automatic regularization selection
6. Validation framework: Synthetic ground truth benchmarks

Citation:
    If you use MetaTrajectory, please cite:
    [Publication information]
    
    Metabolic pathway data from scMetabolism:
    Wu Y, Gao Q, et al. Cancer Discovery, 2022.
"""

__version__ = "1.0.0"
__author__ = "MetaTrajectory Team"

# Core classes
from .knowledge_base import MetabolicKnowledgeBase, parse_gmt_file, filter_pathways
from .state_space import MetabolicStateSpace
from .optimal_transport import MetabolicOT
from .integration import VelocityIntegration
from .drivers import DriverAnalysis

# Advanced scoring
from .scoring import PathwayScoringMethods, compute_pathway_activity

# Statistical methods
from .statistics import (
    EffectSizeCalculator,
    MultipleTestingCorrection,
    UncertaintyQuantification,
    rigorous_differential_test,
    global_fdr_correction,
)

# Network-aware distances
from .network_distance import MetabolicNetworkDistance, build_pathway_network_from_kb

# Parameter selection
from .parameter_selection import AdaptiveParameterSelector, select_optimal_k_neighbors

# Benchmarking
from .benchmark import TrajectoryBenchmark, SyntheticDataGenerator

# Plotting
from .plotting import (
    plot_metabolic_landscape,
    plot_transport_flow,
    plot_fate_probabilities,
    plot_driver_genes,
)

__all__ = [
    # Core Classes
    "MetabolicKnowledgeBase",
    "MetabolicStateSpace", 
    "MetabolicOT",
    "VelocityIntegration",
    "DriverAnalysis",
    
    # Knowledge base utilities
    "parse_gmt_file",
    "filter_pathways",
    
    # Advanced Scoring
    "PathwayScoringMethods",
    "compute_pathway_activity",
    
    # Statistical Methods
    "EffectSizeCalculator",
    "MultipleTestingCorrection", 
    "UncertaintyQuantification",
    "rigorous_differential_test",
    "global_fdr_correction",
    
    # Network Distance
    "MetabolicNetworkDistance",
    "build_pathway_network_from_kb",
    
    # Parameter Selection
    "AdaptiveParameterSelector",
    "select_optimal_k_neighbors",
    
    # Benchmarking
    "TrajectoryBenchmark",
    "SyntheticDataGenerator",
    
    # Plotting
    "plot_metabolic_landscape",
    "plot_transport_flow",
    "plot_fate_probabilities",
    "plot_driver_genes",
]
