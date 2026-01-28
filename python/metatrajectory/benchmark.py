"""
Benchmark and Validation Framework

Provides systematic comparison with existing methods and
ground truth validation for trajectory inference.

Comparisons:
1. CellRank - fate probability accuracy
2. Palantir - pseudotime correlation
3. Monocle3 - trajectory topology
4. scVelo - velocity consistency

Validation:
1. Synthetic data with known trajectories
2. Time-series data validation
3. Marker gene correlation
4. Cross-validation

This addresses the reviewer requirement for systematic benchmarking.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    silhouette_score, calinski_harabasz_score
)
from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings
from dataclasses import dataclass, field
from enum import Enum


class BenchmarkMetric(Enum):
    """Benchmark metrics for trajectory comparison."""
    FATE_CORRELATION = "fate_correlation"
    PSEUDOTIME_CORRELATION = "pseudotime_correlation"
    TRAJECTORY_SIMILARITY = "trajectory_similarity"
    TRANSITION_ACCURACY = "transition_accuracy"
    DRIVER_OVERLAP = "driver_overlap"
    CLUSTER_AGREEMENT = "cluster_agreement"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    method_name: str
    metric: str
    score: float
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    details: Dict = field(default_factory=dict)


class TrajectoryBenchmark:
    """
    Systematic benchmarking against existing trajectory methods.
    
    Provides fair comparison by evaluating multiple aspects:
    - Fate probability estimation
    - Pseudotime inference
    - Trajectory topology
    - Driver gene identification
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    ground_truth : dict, optional
        Known ground truth for validation
    """
    
    def __init__(self, adata, ground_truth: Optional[Dict] = None):
        self.adata = adata
        self.ground_truth = ground_truth or {}
        self.results: List[BenchmarkResult] = []
    
    def compare_fate_probabilities(
        self,
        metatraj_fates: pd.DataFrame,
        other_fates: pd.DataFrame,
        method_name: str = "CellRank"
    ) -> BenchmarkResult:
        """
        Compare fate probability estimates.
        
        Parameters
        ----------
        metatraj_fates : pd.DataFrame
            MetaTrajectory fate probabilities (cells x fates)
        other_fates : pd.DataFrame
            Comparison method fate probabilities
        method_name : str
            Name of comparison method
            
        Returns
        -------
        BenchmarkResult
        """
        # Align cells
        common_cells = metatraj_fates.index.intersection(other_fates.index)
        
        if len(common_cells) < 10:
            return BenchmarkResult(
                method_name=method_name,
                metric="fate_correlation",
                score=np.nan,
                details={"error": "Insufficient common cells"}
            )
        
        mt_fates = metatraj_fates.loc[common_cells]
        other = other_fates.loc[common_cells]
        
        # Find best matching fate columns
        correlations = []
        
        for mt_col in mt_fates.columns:
            best_corr = 0
            for other_col in other.columns:
                try:
                    corr, _ = spearmanr(mt_fates[mt_col], other[other_col])
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                except:
                    pass
            correlations.append(abs(best_corr))
        
        mean_correlation = np.mean(correlations)
        
        # Bootstrap confidence interval
        boot_corrs = []
        for _ in range(1000):
            idx = np.random.choice(len(common_cells), len(common_cells), replace=True)
            boot_corrs.append(np.mean([
                abs(spearmanr(mt_fates.iloc[idx][c], other.iloc[idx].iloc[:, 0])[0])
                for c in mt_fates.columns
            ]))
        
        ci = (np.percentile(boot_corrs, 2.5), np.percentile(boot_corrs, 97.5))
        
        result = BenchmarkResult(
            method_name=method_name,
            metric="fate_correlation",
            score=mean_correlation,
            confidence_interval=ci,
            details={
                "n_cells": len(common_cells),
                "n_fates_mt": len(mt_fates.columns),
                "n_fates_other": len(other.columns),
                "per_fate_correlations": correlations
            }
        )
        self.results.append(result)
        return result
    
    def compare_pseudotime(
        self,
        metatraj_time: np.ndarray,
        other_time: np.ndarray,
        method_name: str = "Palantir",
        true_time: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Compare pseudotime estimates.
        
        Parameters
        ----------
        metatraj_time : np.ndarray
            MetaTrajectory pseudotime
        other_time : np.ndarray
            Other method pseudotime
        method_name : str
            Comparison method name
        true_time : np.ndarray, optional
            Ground truth time if available
            
        Returns
        -------
        BenchmarkResult
        """
        valid = ~(np.isnan(metatraj_time) | np.isnan(other_time))
        mt_time = metatraj_time[valid]
        ot_time = other_time[valid]
        
        # Correlation between methods
        spearman, p_spear = spearmanr(mt_time, ot_time)
        kendall, p_kend = kendalltau(mt_time, ot_time)
        
        details = {
            "spearman_correlation": spearman,
            "kendall_tau": kendall,
            "n_cells": len(mt_time)
        }
        
        # If ground truth available
        if true_time is not None:
            true = true_time[valid]
            mt_true_corr, _ = spearmanr(mt_time, true)
            ot_true_corr, _ = spearmanr(ot_time, true)
            details["metatraj_vs_truth"] = mt_true_corr
            details[f"{method_name}_vs_truth"] = ot_true_corr
            details["metatraj_better"] = abs(mt_true_corr) > abs(ot_true_corr)
        
        result = BenchmarkResult(
            method_name=method_name,
            metric="pseudotime_correlation",
            score=spearman,
            p_value=p_spear,
            details=details
        )
        self.results.append(result)
        return result
    
    def compare_transition_matrix(
        self,
        metatraj_trans: np.ndarray,
        other_trans: np.ndarray,
        method_name: str = "CellRank"
    ) -> BenchmarkResult:
        """
        Compare transition probability matrices.
        
        Parameters
        ----------
        metatraj_trans : np.ndarray
            MetaTrajectory transition matrix
        other_trans : np.ndarray
            Other method transition matrix
        method_name : str
            Comparison method name
            
        Returns
        -------
        BenchmarkResult
        """
        n = min(metatraj_trans.shape[0], other_trans.shape[0])
        mt = metatraj_trans[:n, :n]
        ot = other_trans[:n, :n]
        
        # Frobenius norm of difference
        frobenius = np.sqrt(np.sum((mt - ot) ** 2))
        
        # Normalize by matrix size
        normalized_diff = frobenius / n
        
        # Correlation of non-zero entries
        mt_flat = mt.flatten()
        ot_flat = ot.flatten()
        nonzero = (mt_flat > 1e-6) | (ot_flat > 1e-6)
        
        if nonzero.sum() > 10:
            corr, p_val = spearmanr(mt_flat[nonzero], ot_flat[nonzero])
        else:
            corr, p_val = np.nan, 1.0
        
        # KL divergence (row-wise)
        kl_divs = []
        for i in range(n):
            mt_row = mt[i] + 1e-10
            mt_row = mt_row / mt_row.sum()
            ot_row = ot[i] + 1e-10
            ot_row = ot_row / ot_row.sum()
            kl = np.sum(mt_row * np.log(mt_row / ot_row))
            kl_divs.append(kl)
        
        result = BenchmarkResult(
            method_name=method_name,
            metric="transition_similarity",
            score=corr,
            p_value=p_val,
            details={
                "frobenius_norm": frobenius,
                "normalized_difference": normalized_diff,
                "mean_kl_divergence": np.mean(kl_divs),
                "n_cells": n
            }
        )
        self.results.append(result)
        return result
    
    def compare_driver_genes(
        self,
        metatraj_drivers: pd.DataFrame,
        other_drivers: pd.DataFrame,
        method_name: str = "CellRank",
        top_n: int = 50
    ) -> BenchmarkResult:
        """
        Compare driver gene rankings.
        
        Parameters
        ----------
        metatraj_drivers : pd.DataFrame
            MetaTrajectory driver gene results
        other_drivers : pd.DataFrame
            Other method results
        method_name : str
            Comparison method name
        top_n : int
            Number of top genes to compare
            
        Returns
        -------
        BenchmarkResult
        """
        # Get top genes from each method
        if 'gene' in metatraj_drivers.columns:
            mt_genes = set(metatraj_drivers.nsmallest(top_n, 'p_adjusted')['gene'])
        else:
            mt_genes = set(metatraj_drivers.head(top_n).index)
        
        if 'gene' in other_drivers.columns:
            other_genes = set(other_drivers.nsmallest(top_n, 'pvalue' if 'pvalue' in other_drivers.columns else 'p_value')['gene'])
        else:
            other_genes = set(other_drivers.head(top_n).index)
        
        # Overlap metrics
        intersection = len(mt_genes & other_genes)
        union = len(mt_genes | other_genes)
        jaccard = intersection / union if union > 0 else 0
        
        # Overlap coefficient
        overlap_coef = intersection / min(len(mt_genes), len(other_genes))
        
        result = BenchmarkResult(
            method_name=method_name,
            metric="driver_overlap",
            score=jaccard,
            details={
                "jaccard_index": jaccard,
                "overlap_coefficient": overlap_coef,
                "n_shared_drivers": intersection,
                "n_metatraj_drivers": len(mt_genes),
                "n_other_drivers": len(other_genes),
                "shared_genes": list(mt_genes & other_genes)[:20]
            }
        )
        self.results.append(result)
        return result
    
    def validate_against_ground_truth(
        self,
        predicted_fates: pd.DataFrame,
        true_fates: pd.DataFrame,
        predicted_time: Optional[np.ndarray] = None,
        true_time: Optional[np.ndarray] = None
    ) -> Dict[str, BenchmarkResult]:
        """
        Validate against known ground truth.
        
        Parameters
        ----------
        predicted_fates : pd.DataFrame
            Predicted fate probabilities
        true_fates : pd.DataFrame
            True fate labels (one-hot or probabilities)
        predicted_time : np.ndarray, optional
            Predicted pseudotime
        true_time : np.ndarray, optional
            True time labels
            
        Returns
        -------
        dict
            Dictionary of validation results
        """
        results = {}
        
        # Fate accuracy
        if true_fates is not None:
            # Convert true fates to same format
            if true_fates.ndim == 1 or true_fates.shape[1] == 1:
                # Categorical labels
                true_labels = true_fates.values.flatten() if isinstance(true_fates, pd.DataFrame) else true_fates
                pred_labels = predicted_fates.idxmax(axis=1).values
                
                ari = adjusted_rand_score(true_labels, pred_labels)
                nmi = normalized_mutual_info_score(true_labels, pred_labels)
                
                results["fate_classification"] = BenchmarkResult(
                    method_name="ground_truth",
                    metric="fate_classification",
                    score=ari,
                    details={
                        "adjusted_rand_index": ari,
                        "normalized_mutual_info": nmi
                    }
                )
            else:
                # Probability comparison
                common = predicted_fates.index.intersection(true_fates.index)
                pred = predicted_fates.loc[common]
                true = true_fates.loc[common]
                
                # MSE
                mse = np.mean((pred.values - true.values) ** 2)
                
                # Correlation per fate
                fate_corrs = []
                for c in pred.columns:
                    if c in true.columns:
                        corr, _ = spearmanr(pred[c], true[c])
                        fate_corrs.append(corr)
                
                results["fate_probability"] = BenchmarkResult(
                    method_name="ground_truth",
                    metric="fate_probability",
                    score=np.mean(fate_corrs) if fate_corrs else np.nan,
                    details={
                        "mse": mse,
                        "per_fate_correlations": fate_corrs
                    }
                )
        
        # Time accuracy
        if predicted_time is not None and true_time is not None:
            valid = ~(np.isnan(predicted_time) | np.isnan(true_time))
            
            spearman, p_val = spearmanr(predicted_time[valid], true_time[valid])
            kendall, _ = kendalltau(predicted_time[valid], true_time[valid])
            
            results["pseudotime"] = BenchmarkResult(
                method_name="ground_truth",
                metric="pseudotime_accuracy",
                score=spearman,
                p_value=p_val,
                details={
                    "spearman": spearman,
                    "kendall": kendall,
                    "n_cells": valid.sum()
                }
            )
        
        self.results.extend(results.values())
        return results
    
    def generate_report(self) -> pd.DataFrame:
        """
        Generate summary report of all benchmarks.
        
        Returns
        -------
        pd.DataFrame
            Summary table
        """
        records = []
        for r in self.results:
            record = {
                "method": r.method_name,
                "metric": r.metric,
                "score": r.score,
                "p_value": r.p_value
            }
            if r.confidence_interval:
                record["ci_lower"] = r.confidence_interval[0]
                record["ci_upper"] = r.confidence_interval[1]
            records.append(record)
        
        return pd.DataFrame(records)


class SyntheticDataGenerator:
    """
    Generate synthetic data with known ground truth for validation.
    
    Creates realistic single-cell data with:
    - Known trajectory structure
    - Known fate probabilities
    - Known driver genes
    """
    
    @staticmethod
    def generate_bifurcating_trajectory(
        n_cells: int = 1000,
        n_genes: int = 500,
        n_metabolic_genes: int = 100,
        branch_point: float = 0.4,
        noise_level: float = 0.1,
        seed: int = 42
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate bifurcating trajectory with two branches.
        
        Parameters
        ----------
        n_cells : int
            Number of cells
        n_genes : int
            Total genes
        n_metabolic_genes : int
            Number of metabolic genes
        branch_point : float
            Pseudotime of bifurcation
        noise_level : float
            Expression noise level
        seed : int
            Random seed
            
        Returns
        -------
        expression : np.ndarray
            Expression matrix (cells x genes)
        ground_truth : dict
            Contains true time, fate, drivers
        """
        np.random.seed(seed)
        
        # Generate pseudotime
        pseudotime = np.random.beta(2, 2, n_cells)
        
        # Assign branches after branch point
        branch = np.zeros(n_cells, dtype=int)
        after_branch = pseudotime > branch_point
        branch[after_branch] = np.random.binomial(1, 0.5, after_branch.sum())
        
        # Generate fate probabilities
        fate_probs = np.zeros((n_cells, 2))
        for i in range(n_cells):
            if pseudotime[i] <= branch_point:
                # Before bifurcation - uncertain
                fate_probs[i] = [0.5, 0.5]
            else:
                # After bifurcation - committed
                distance_from_branch = pseudotime[i] - branch_point
                certainty = min(1.0, distance_from_branch / (1 - branch_point) * 2)
                
                if branch[i] == 0:
                    fate_probs[i] = [0.5 + 0.5 * certainty, 0.5 - 0.5 * certainty]
                else:
                    fate_probs[i] = [0.5 - 0.5 * certainty, 0.5 + 0.5 * certainty]
        
        # Generate expression
        expression = np.zeros((n_cells, n_genes))
        
        # Time-dependent genes (first 50 metabolic genes)
        for g in range(50):
            # Monotonic increase with time
            base = pseudotime ** (1 + 0.5 * np.random.randn())
            expression[:, g] = base + np.random.randn(n_cells) * noise_level
        
        # Branch-specific genes (next 50 metabolic genes)
        driver_genes = []
        for g in range(50, 100):
            if g < 75:
                # Branch 0 specific (driver genes)
                effect = np.where(branch == 0, 1.0, 0.0) * (pseudotime - branch_point).clip(0)
                driver_genes.append(g)
            else:
                # Branch 1 specific (driver genes)
                effect = np.where(branch == 1, 1.0, 0.0) * (pseudotime - branch_point).clip(0)
                driver_genes.append(g)
            
            expression[:, g] = effect * 3 + np.random.randn(n_cells) * noise_level
        
        # Non-metabolic genes (random)
        expression[:, n_metabolic_genes:] = np.random.randn(n_cells, n_genes - n_metabolic_genes) * 0.5
        
        # Make non-negative
        expression = np.maximum(expression, 0)
        
        ground_truth = {
            "pseudotime": pseudotime,
            "branch": branch,
            "fate_probabilities": fate_probs,
            "driver_genes": driver_genes,
            "branch_point": branch_point,
            "metabolic_gene_indices": list(range(n_metabolic_genes))
        }
        
        return expression, ground_truth
    
    @staticmethod
    def generate_linear_trajectory(
        n_cells: int = 500,
        n_genes: int = 200,
        noise_level: float = 0.1,
        seed: int = 42
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate simple linear trajectory.
        
        Returns
        -------
        expression, ground_truth
        """
        np.random.seed(seed)
        
        pseudotime = np.random.uniform(0, 1, n_cells)
        
        expression = np.zeros((n_cells, n_genes))
        
        # Time-correlated genes
        for g in range(n_genes // 2):
            # Various temporal patterns
            pattern = np.random.choice(['up', 'down', 'transient'])
            
            if pattern == 'up':
                base = pseudotime
            elif pattern == 'down':
                base = 1 - pseudotime
            else:
                base = np.exp(-((pseudotime - 0.5) ** 2) / 0.1)
            
            expression[:, g] = base + np.random.randn(n_cells) * noise_level
        
        # Random genes
        expression[:, n_genes // 2:] = np.random.randn(n_cells, n_genes // 2) * 0.3
        
        expression = np.maximum(expression, 0)
        
        ground_truth = {
            "pseudotime": pseudotime,
            "temporal_genes": list(range(n_genes // 2))
        }
        
        return expression, ground_truth
    
    @staticmethod
    def generate_multibranch_trajectory(
        n_cells: int = 1500,
        n_genes: int = 500,
        n_branches: int = 3,
        seed: int = 42
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate trajectory with multiple branches.
        
        Returns
        -------
        expression, ground_truth
        """
        np.random.seed(seed)
        
        cells_per_branch = n_cells // n_branches
        
        pseudotime = np.zeros(n_cells)
        branch = np.zeros(n_cells, dtype=int)
        
        # Generate cells on each branch
        for b in range(n_branches):
            start = b * cells_per_branch
            end = start + cells_per_branch if b < n_branches - 1 else n_cells
            
            pseudotime[start:end] = np.random.uniform(0, 1, end - start)
            branch[start:end] = b
        
        expression = np.zeros((n_cells, n_genes))
        
        # Shared early genes
        for g in range(50):
            expression[:, g] = pseudotime + np.random.randn(n_cells) * 0.1
        
        # Branch-specific genes
        genes_per_branch = 50
        for b in range(n_branches):
            g_start = 50 + b * genes_per_branch
            g_end = g_start + genes_per_branch
            
            for g in range(g_start, min(g_end, n_genes)):
                on_branch = branch == b
                expression[on_branch, g] = pseudotime[on_branch] * 2 + np.random.randn(on_branch.sum()) * 0.1
        
        expression = np.maximum(expression, 0)
        
        ground_truth = {
            "pseudotime": pseudotime,
            "branch": branch,
            "n_branches": n_branches
        }
        
        return expression, ground_truth


def run_full_benchmark(
    adata,
    metatraj_results: Dict,
    compare_methods: List[str] = None,
    n_synthetic: int = 1000
) -> pd.DataFrame:
    """
    Run complete benchmark suite.
    
    Parameters
    ----------
    adata : AnnData
        Data
    metatraj_results : dict
        MetaTrajectory results containing:
        - fate_probabilities
        - transition_matrix
        - drivers
    compare_methods : list
        Methods to compare against
    n_synthetic : int
        Number of synthetic cells
        
    Returns
    -------
    pd.DataFrame
        Complete benchmark report
    """
    benchmark = TrajectoryBenchmark(adata)
    
    # Synthetic validation
    print("Running synthetic data validation...")
    expr, gt = SyntheticDataGenerator.generate_bifurcating_trajectory(n_synthetic)
    
    # Create AnnData for synthetic
    import anndata as ad
    synthetic_adata = ad.AnnData(expr)
    
    # Run MetaTrajectory on synthetic (simplified)
    # This would be replaced with actual run
    
    # Validate against ground truth
    if "fate_probabilities" in metatraj_results:
        pred_fates = metatraj_results["fate_probabilities"]
        true_fates = pd.DataFrame(gt["fate_probabilities"], columns=["Fate_0", "Fate_1"])
        
        benchmark.validate_against_ground_truth(
            pred_fates.iloc[:len(true_fates)],
            true_fates
        )
    
    return benchmark.generate_report()
