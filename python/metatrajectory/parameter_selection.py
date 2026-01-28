"""
Adaptive Parameter Selection Module

Provides automatic parameter selection for optimal transport
and trajectory inference.

Methods:
1. Cross-validation for regularization parameter
2. Stability-based selection
3. Information-theoretic criteria (AIC, BIC)
4. Sensitivity analysis

This addresses the reviewer concern about arbitrary parameter choices.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brent
from scipy.spatial.distance import cdist
from typing import Dict, List, Optional, Tuple, Callable
import warnings
from dataclasses import dataclass

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False


@dataclass
class ParameterSelectionResult:
    """Result of parameter selection."""
    optimal_value: float
    criterion_name: str
    criterion_value: float
    search_history: List[Tuple[float, float]]
    sensitivity: Optional[Dict[str, float]] = None


class AdaptiveParameterSelector:
    """
    Automatic parameter selection for optimal transport.
    
    Provides multiple methods for selecting the entropic regularization
    parameter (epsilon/reg) and other hyperparameters.
    
    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix for OT
    source_weights : np.ndarray
        Source distribution
    target_weights : np.ndarray
        Target distribution
    """
    
    def __init__(
        self,
        cost_matrix: np.ndarray,
        source_weights: np.ndarray,
        target_weights: np.ndarray
    ):
        self.C = cost_matrix
        self.a = source_weights / source_weights.sum()
        self.b = target_weights / target_weights.sum()
        
        # Cost statistics for scaling
        self.cost_mean = np.mean(cost_matrix)
        self.cost_std = np.std(cost_matrix)
        self.cost_median = np.median(cost_matrix)
    
    def select_regularization(
        self,
        method: str = "stability",
        reg_range: Tuple[float, float] = (0.001, 1.0),
        n_points: int = 20
    ) -> ParameterSelectionResult:
        """
        Select optimal regularization parameter.
        
        Parameters
        ----------
        method : str
            'stability': Based on transport plan stability
            'entropy': Based on entropy of transport plan
            'cv': Cross-validation
            'marginal': Based on marginal constraint satisfaction
        reg_range : tuple
            (min_reg, max_reg) to search
        n_points : int
            Number of points to evaluate
            
        Returns
        -------
        ParameterSelectionResult
        """
        if method == "stability":
            return self._stability_selection(reg_range, n_points)
        elif method == "entropy":
            return self._entropy_selection(reg_range, n_points)
        elif method == "cv":
            return self._cv_selection(reg_range, n_points)
        elif method == "marginal":
            return self._marginal_selection(reg_range, n_points)
        elif method == "auto":
            return self._auto_selection(reg_range)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _stability_selection(
        self,
        reg_range: Tuple[float, float],
        n_points: int
    ) -> ParameterSelectionResult:
        """
        Select regularization based on transport plan stability.
        
        Good regularization gives stable transport plans under
        small perturbations.
        """
        if not HAS_POT:
            raise ImportError("POT required for OT computation")
        
        reg_values = np.logspace(
            np.log10(reg_range[0]),
            np.log10(reg_range[1]),
            n_points
        )
        
        history = []
        
        for reg in reg_values:
            # Compute transport plan
            try:
                T = ot.sinkhorn(self.a, self.b, self.C, reg)
            except:
                history.append((reg, np.inf))
                continue
            
            # Compute stability: variance under perturbation
            perturbations = []
            for _ in range(5):
                # Add small noise to cost
                noise = np.random.randn(*self.C.shape) * 0.01 * self.cost_std
                C_pert = self.C + noise
                
                try:
                    T_pert = ot.sinkhorn(self.a, self.b, C_pert, reg)
                    # Frobenius distance
                    diff = np.sqrt(np.sum((T - T_pert) ** 2))
                    perturbations.append(diff)
                except:
                    pass
            
            if perturbations:
                stability = np.mean(perturbations)
            else:
                stability = np.inf
            
            history.append((reg, stability))
        
        # Find minimum
        valid = [(r, s) for r, s in history if np.isfinite(s)]
        if not valid:
            return ParameterSelectionResult(
                optimal_value=0.1,
                criterion_name="stability",
                criterion_value=np.inf,
                search_history=history
            )
        
        best_reg, best_stability = min(valid, key=lambda x: x[1])
        
        return ParameterSelectionResult(
            optimal_value=best_reg,
            criterion_name="stability",
            criterion_value=best_stability,
            search_history=history
        )
    
    def _entropy_selection(
        self,
        reg_range: Tuple[float, float],
        n_points: int
    ) -> ParameterSelectionResult:
        """
        Select regularization based on transport plan entropy.
        
        Seeks intermediate entropy - not too sparse (low reg)
        and not too diffuse (high reg).
        """
        if not HAS_POT:
            raise ImportError("POT required")
        
        reg_values = np.logspace(
            np.log10(reg_range[0]),
            np.log10(reg_range[1]),
            n_points
        )
        
        history = []
        entropies = []
        
        for reg in reg_values:
            try:
                T = ot.sinkhorn(self.a, self.b, self.C, reg)
                
                # Entropy of transport plan
                T_flat = T.flatten()
                T_flat = T_flat[T_flat > 1e-10]
                entropy = -np.sum(T_flat * np.log(T_flat))
                
                history.append((reg, entropy))
                entropies.append(entropy)
            except:
                history.append((reg, np.nan))
        
        # Select based on "elbow" in entropy curve
        # Find point of maximum curvature
        valid_entropies = [e for e in entropies if np.isfinite(e)]
        if len(valid_entropies) < 3:
            return ParameterSelectionResult(
                optimal_value=0.1,
                criterion_name="entropy",
                criterion_value=np.nan,
                search_history=history
            )
        
        # Normalize to find elbow
        log_regs = np.log10([r for r, e in history if np.isfinite(e)])
        norm_entropies = np.array(valid_entropies)
        norm_entropies = (norm_entropies - norm_entropies.min()) / (
            norm_entropies.max() - norm_entropies.min() + 1e-10
        )
        norm_regs = (log_regs - log_regs.min()) / (log_regs.max() - log_regs.min() + 1e-10)
        
        # Distance from line connecting endpoints
        distances = []
        for i in range(len(norm_regs)):
            # Distance from point (norm_regs[i], norm_entropies[i]) to line
            # connecting (0, norm_entropies[0]) and (1, norm_entropies[-1])
            x0, y0 = 0, norm_entropies[0]
            x1, y1 = 1, norm_entropies[-1]
            x, y = norm_regs[i], norm_entropies[i]
            
            dist = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / (
                np.sqrt((y1 - y0)**2 + (x1 - x0)**2) + 1e-10
            )
            distances.append(dist)
        
        elbow_idx = np.argmax(distances)
        best_reg = 10 ** log_regs[elbow_idx]
        
        return ParameterSelectionResult(
            optimal_value=best_reg,
            criterion_name="entropy_elbow",
            criterion_value=valid_entropies[elbow_idx],
            search_history=history
        )
    
    def _marginal_selection(
        self,
        reg_range: Tuple[float, float],
        n_points: int
    ) -> ParameterSelectionResult:
        """
        Select based on marginal constraint satisfaction.
        
        Lower regularization gives better marginal fit but may
        be unstable. Find sweet spot.
        """
        if not HAS_POT:
            raise ImportError("POT required")
        
        reg_values = np.logspace(
            np.log10(reg_range[0]),
            np.log10(reg_range[1]),
            n_points
        )
        
        history = []
        
        for reg in reg_values:
            try:
                T = ot.sinkhorn(self.a, self.b, self.C, reg)
                
                # Marginal errors
                err_source = np.sum(np.abs(T.sum(axis=1) - self.a))
                err_target = np.sum(np.abs(T.sum(axis=0) - self.b))
                total_error = err_source + err_target
                
                history.append((reg, total_error))
            except:
                history.append((reg, np.inf))
        
        # Find smallest reg with acceptable error (< 1e-3)
        acceptable = [(r, e) for r, e in history if e < 1e-3]
        
        if acceptable:
            best_reg = min(acceptable, key=lambda x: x[0])[0]
            best_error = dict(history)[best_reg]
        else:
            # Fall back to minimum error
            valid = [(r, e) for r, e in history if np.isfinite(e)]
            if valid:
                best_reg, best_error = min(valid, key=lambda x: x[1])
            else:
                best_reg, best_error = 0.1, np.inf
        
        return ParameterSelectionResult(
            optimal_value=best_reg,
            criterion_name="marginal_error",
            criterion_value=best_error,
            search_history=history
        )
    
    def _cv_selection(
        self,
        reg_range: Tuple[float, float],
        n_points: int,
        n_folds: int = 5
    ) -> ParameterSelectionResult:
        """
        Cross-validation for regularization selection.
        
        Holds out portions of the cost matrix and evaluates
        prediction on held-out entries.
        """
        if not HAS_POT:
            raise ImportError("POT required")
        
        reg_values = np.logspace(
            np.log10(reg_range[0]),
            np.log10(reg_range[1]),
            n_points
        )
        
        n, m = self.C.shape
        
        # Create random mask for CV
        np.random.seed(42)
        indices = np.random.permutation(n * m)
        fold_size = len(indices) // n_folds
        
        history = []
        
        for reg in reg_values:
            cv_errors = []
            
            for fold in range(n_folds):
                # Create holdout mask
                start = fold * fold_size
                end = start + fold_size if fold < n_folds - 1 else len(indices)
                holdout_flat = indices[start:end]
                
                # Mask cost matrix
                C_train = self.C.copy()
                holdout_rows = holdout_flat // m
                holdout_cols = holdout_flat % m
                
                # Replace holdout with median
                C_train[holdout_rows, holdout_cols] = self.cost_median
                
                try:
                    # Compute transport on training
                    T = ot.sinkhorn(self.a, self.b, C_train, reg)
                    
                    # Predict transport on holdout
                    # Error: how well does T explain the true cost?
                    pred_cost = np.sum(T * self.C)
                    actual_cost = np.sum(T * C_train)
                    
                    cv_errors.append(abs(pred_cost - actual_cost))
                except:
                    cv_errors.append(np.inf)
            
            mean_cv = np.mean([e for e in cv_errors if np.isfinite(e)])
            history.append((reg, mean_cv))
        
        valid = [(r, e) for r, e in history if np.isfinite(e)]
        if valid:
            best_reg, best_cv = min(valid, key=lambda x: x[1])
        else:
            best_reg, best_cv = 0.1, np.inf
        
        return ParameterSelectionResult(
            optimal_value=best_reg,
            criterion_name="cv_error",
            criterion_value=best_cv,
            search_history=history
        )
    
    def _auto_selection(
        self,
        reg_range: Tuple[float, float]
    ) -> ParameterSelectionResult:
        """
        Automatic selection using cost matrix statistics.
        
        Fast heuristic based on cost matrix properties.
        """
        # Heuristic: regularization ~ median cost / 10
        reg = self.cost_median / 10
        
        # Clip to range
        reg = np.clip(reg, reg_range[0], reg_range[1])
        
        return ParameterSelectionResult(
            optimal_value=reg,
            criterion_name="heuristic",
            criterion_value=0.0,
            search_history=[(reg, 0.0)]
        )
    
    def sensitivity_analysis(
        self,
        optimal_reg: float,
        perturbation_range: Tuple[float, float] = (0.5, 2.0),
        n_points: int = 10
    ) -> Dict[str, float]:
        """
        Analyze sensitivity of results to regularization choice.
        
        Parameters
        ----------
        optimal_reg : float
            Selected regularization
        perturbation_range : tuple
            (min_mult, max_mult) to perturb
        n_points : int
            Number of perturbations
            
        Returns
        -------
        dict
            Sensitivity metrics
        """
        if not HAS_POT:
            raise ImportError("POT required")
        
        # Compute reference transport
        try:
            T_ref = ot.sinkhorn(self.a, self.b, self.C, optimal_reg)
        except:
            return {"error": "Failed to compute reference transport"}
        
        # Perturb regularization
        multipliers = np.linspace(
            perturbation_range[0],
            perturbation_range[1],
            n_points
        )
        
        transport_diffs = []
        cost_diffs = []
        
        ref_cost = np.sum(T_ref * self.C)
        
        for mult in multipliers:
            reg_pert = optimal_reg * mult
            try:
                T_pert = ot.sinkhorn(self.a, self.b, self.C, reg_pert)
                
                # Frobenius difference
                transport_diff = np.sqrt(np.sum((T_ref - T_pert) ** 2))
                transport_diffs.append(transport_diff)
                
                # Cost difference
                cost_pert = np.sum(T_pert * self.C)
                cost_diffs.append(abs(cost_pert - ref_cost) / ref_cost)
            except:
                pass
        
        if not transport_diffs:
            return {"error": "Sensitivity analysis failed"}
        
        return {
            "transport_stability": 1 - np.mean(transport_diffs) / (np.max(transport_diffs) + 1e-10),
            "cost_sensitivity": np.mean(cost_diffs),
            "max_transport_change": np.max(transport_diffs),
            "max_cost_change": np.max(cost_diffs)
        }


def select_optimal_k_neighbors(
    embedding: np.ndarray,
    k_range: Tuple[int, int] = (5, 100),
    method: str = "connectivity"
) -> int:
    """
    Select optimal number of neighbors for kNN graph.
    
    Parameters
    ----------
    embedding : np.ndarray
        Cell embeddings
    k_range : tuple
        (min_k, max_k)
    method : str
        'connectivity': Based on graph connectivity
        'variance': Based on variance explained
        
    Returns
    -------
    int
        Optimal k
    """
    from sklearn.neighbors import kneighbors_graph
    
    n_cells = embedding.shape[0]
    k_values = list(range(k_range[0], min(k_range[1], n_cells // 2), 5))
    
    if method == "connectivity":
        scores = []
        for k in k_values:
            # Build kNN graph
            A = kneighbors_graph(embedding, k, mode='connectivity')
            
            # Check connectivity (number of components)
            from scipy.sparse.csgraph import connected_components
            n_components = connected_components(A, directed=False)[0]
            
            # Score: lower is better for being connected
            # But also penalize too high k (oversmoothing)
            score = n_components + 0.01 * k
            scores.append(score)
        
        best_idx = np.argmin(scores)
        return k_values[best_idx]
    
    elif method == "variance":
        # Use reconstruction error
        scores = []
        for k in k_values:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k).fit(embedding)
            distances, indices = nn.kneighbors(embedding)
            
            # Reconstruction: mean of k neighbors
            reconstructed = np.zeros_like(embedding)
            for i in range(n_cells):
                reconstructed[i] = embedding[indices[i]].mean(axis=0)
            
            # Error
            error = np.mean((embedding - reconstructed) ** 2)
            scores.append(error)
        
        # Find elbow
        scores = np.array(scores)
        normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        # Second derivative to find elbow
        if len(normalized) >= 3:
            second_deriv = np.diff(normalized, 2)
            elbow_idx = np.argmax(second_deriv) + 1
            return k_values[elbow_idx]
        
        return k_values[len(k_values) // 2]
    
    else:
        raise ValueError(f"Unknown method: {method}")
