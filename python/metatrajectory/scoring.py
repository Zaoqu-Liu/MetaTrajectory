"""
Advanced Pathway Scoring Methods

This module provides multiple scientifically rigorous methods for computing
pathway activity scores from single-cell gene expression data.

Methods implemented:
1. AUCell-like: Area Under the recovery Curve
2. VISION-like: Signature scoring with proper normalization  
3. ssGSEA-like: Single-sample Gene Set Enrichment Analysis
4. Weighted scoring: Rate-limiting enzyme weighted average
5. GSVA-like: Gene Set Variation Analysis

References:
- Aibar et al. (2017) SCENIC: AUCell method
- DeTomaso et al. (2019) VISION
- Barbie et al. (2009) ssGSEA
- Hänzelmann et al. (2013) GSVA
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, zscore
from scipy.sparse import issparse
from typing import Dict, List, Optional, Tuple, Union
import warnings
from concurrent.futures import ThreadPoolExecutor


class PathwayScoringMethods:
    """
    Collection of pathway scoring methods with proper statistical foundation.
    
    This class provides multiple methods for computing pathway activity scores,
    each with different assumptions and use cases.
    
    Parameters
    ----------
    pathways : dict
        Pathway name -> gene list
    rate_limiting_enzymes : set
        Set of rate-limiting enzyme genes for weighted scoring
        
    Attributes
    ----------
    method_descriptions : dict
        Description and recommended use case for each method
    """
    
    def __init__(
        self,
        pathways: Dict[str, List[str]],
        rate_limiting_enzymes: Optional[set] = None
    ):
        self.pathways = pathways
        self.rate_limiting = rate_limiting_enzymes or set()
        
        self.method_descriptions = {
            "aucell": "Rank-based, robust to outliers. Best for identifying cells with active pathways.",
            "vision": "Z-score normalized, interpretable. Best for comparing pathway activity across conditions.",
            "ssgsea": "Enrichment-based, captures coordinated changes. Best for pathways with coherent regulation.",
            "weighted": "Rate-limiting enzyme weighted. Best for metabolic pathway activity estimation.",
            "gsva": "Non-parametric, unsupervised. Best for pathway-level differential analysis.",
            "combined": "Ensemble of multiple methods. Most robust but computationally expensive."
        }
    
    def score_pathways(
        self,
        expression_matrix: np.ndarray,
        gene_names: List[str],
        method: str = "vision",
        n_jobs: int = 1,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute pathway scores using specified method.
        
        Parameters
        ----------
        expression_matrix : np.ndarray
            Cells x Genes expression matrix
        gene_names : list
            Gene names corresponding to columns
        method : str
            Scoring method: 'aucell', 'vision', 'ssgsea', 'weighted', 'gsva', 'combined'
        n_jobs : int
            Number of parallel jobs
        **kwargs
            Method-specific parameters
            
        Returns
        -------
        pd.DataFrame
            Cells x Pathways score matrix with uncertainty estimates
        """
        if method == "aucell":
            return self._aucell_score(expression_matrix, gene_names, **kwargs)
        elif method == "vision":
            return self._vision_score(expression_matrix, gene_names, **kwargs)
        elif method == "ssgsea":
            return self._ssgsea_score(expression_matrix, gene_names, **kwargs)
        elif method == "weighted":
            return self._weighted_score(expression_matrix, gene_names, **kwargs)
        elif method == "gsva":
            return self._gsva_score(expression_matrix, gene_names, **kwargs)
        elif method == "combined":
            return self._combined_score(expression_matrix, gene_names, n_jobs=n_jobs, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.method_descriptions.keys())}")
    
    def _aucell_score(
        self,
        X: np.ndarray,
        genes: List[str],
        n_top: int = 500,
        auc_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        AUCell-like scoring based on gene rankings.
        
        For each cell, genes are ranked by expression. The pathway score is
        the area under the curve of pathway gene recovery.
        
        Parameters
        ----------
        X : np.ndarray
            Expression matrix (cells x genes)
        genes : list
            Gene names
        n_top : int
            Number of top-ranked genes to consider (default: 5% of genes)
        auc_threshold : float
            Fraction of genes to use for AUC calculation
            
        Returns
        -------
        pd.DataFrame
            Pathway scores and AUC values
        """
        n_cells, n_genes = X.shape
        
        # Determine threshold
        if n_top is None:
            n_top = int(n_genes * auc_threshold)
        n_top = min(n_top, n_genes)
        
        # Rank genes for each cell (higher expression = lower rank number)
        # Use negative for descending order
        rankings = np.zeros_like(X)
        for i in range(n_cells):
            rankings[i] = rankdata(-X[i], method='ordinal')
        
        scores = {}
        uncertainties = {}
        
        for pathway, pathway_genes in self.pathways.items():
            # Find available genes
            gene_indices = [genes.index(g) for g in pathway_genes if g in genes]
            
            if len(gene_indices) < 3:
                continue
            
            n_pathway = len(gene_indices)
            
            # For each cell, compute AUC
            cell_scores = np.zeros(n_cells)
            
            for cell in range(n_cells):
                # Get ranks of pathway genes in this cell
                pathway_ranks = rankings[cell, gene_indices]
                
                # Count how many pathway genes are in top n_top
                n_in_top = np.sum(pathway_ranks <= n_top)
                
                # AUC calculation: fraction of possible area
                # Maximum area = n_pathway * n_top (if all pathway genes at top)
                # Actual area approximation using recovery curve
                sorted_ranks = np.sort(pathway_ranks)
                
                # Recovery curve: cumulative count of pathway genes at each rank threshold
                auc = 0
                for j, rank in enumerate(sorted_ranks):
                    if rank <= n_top:
                        # Contribution to AUC
                        auc += (n_top - rank + 1) / n_top
                
                # Normalize by maximum possible AUC
                max_auc = n_pathway
                cell_scores[cell] = auc / max_auc if max_auc > 0 else 0
            
            scores[pathway] = cell_scores
            
            # Bootstrap uncertainty estimation
            uncertainties[pathway] = self._bootstrap_uncertainty(
                X[:, gene_indices], n_bootstrap=100
            )
        
        # Create result DataFrame
        result = pd.DataFrame(scores)
        
        # Add uncertainty as separate columns
        for pathway in uncertainties:
            result[f"{pathway}_uncertainty"] = uncertainties[pathway]
        
        return result
    
    def _vision_score(
        self,
        X: np.ndarray,
        genes: List[str],
        scale: bool = True
    ) -> pd.DataFrame:
        """
        VISION-like signature scoring.
        
        Score = mean(z-scored expression of signature genes)
        With proper handling of missing genes and outliers.
        
        Parameters
        ----------
        X : np.ndarray
            Expression matrix
        genes : list
            Gene names
        scale : bool
            Whether to z-score normalize final scores
            
        Returns
        -------
        pd.DataFrame
            Pathway scores
        """
        n_cells = X.shape[0]
        
        # First, normalize library size
        lib_sizes = X.sum(axis=1, keepdims=True)
        lib_sizes = np.maximum(lib_sizes, 1)  # Avoid division by zero
        X_norm = X / lib_sizes * np.median(lib_sizes)
        
        # Log transform
        X_log = np.log1p(X_norm)
        
        # Z-score genes across cells
        X_z = zscore(X_log, axis=0, nan_policy='omit')
        X_z = np.nan_to_num(X_z, nan=0.0)
        
        scores = {}
        uncertainties = {}
        
        for pathway, pathway_genes in self.pathways.items():
            gene_indices = [genes.index(g) for g in pathway_genes if g in genes]
            
            if len(gene_indices) < 3:
                continue
            
            # Compute signature score as mean of z-scores
            pathway_expr = X_z[:, gene_indices]
            
            # Robust mean (trimmed to remove outliers)
            trimmed_scores = np.zeros(n_cells)
            for i in range(n_cells):
                vals = pathway_expr[i]
                # Trim top and bottom 10%
                n_trim = max(1, int(len(vals) * 0.1))
                sorted_vals = np.sort(vals)
                trimmed_scores[i] = np.mean(sorted_vals[n_trim:-n_trim]) if len(sorted_vals) > 2*n_trim else np.mean(vals)
            
            scores[pathway] = trimmed_scores
            
            # Standard error as uncertainty
            uncertainties[pathway] = np.std(pathway_expr, axis=1) / np.sqrt(len(gene_indices))
        
        result = pd.DataFrame(scores)
        
        if scale:
            result = (result - result.mean()) / (result.std() + 1e-10)
        
        # Add uncertainties
        for pathway in uncertainties:
            result[f"{pathway}_se"] = uncertainties[pathway]
        
        return result
    
    def _ssgsea_score(
        self,
        X: np.ndarray,
        genes: List[str],
        alpha: float = 0.25
    ) -> pd.DataFrame:
        """
        Single-sample GSEA scoring.
        
        Uses rank-based enrichment score for each cell.
        
        Parameters
        ----------
        X : np.ndarray
            Expression matrix
        genes : list
            Gene names
        alpha : float
            Weighting exponent for ranked list
            
        Returns
        -------
        pd.DataFrame
            Enrichment scores
        """
        n_cells, n_genes = X.shape
        
        scores = {}
        
        for pathway, pathway_genes in self.pathways.items():
            gene_indices = set(genes.index(g) for g in pathway_genes if g in genes)
            
            if len(gene_indices) < 3:
                continue
            
            n_pathway = len(gene_indices)
            n_other = n_genes - n_pathway
            
            cell_scores = np.zeros(n_cells)
            
            for cell in range(n_cells):
                # Get ranked gene indices (descending by expression)
                expr = X[cell]
                ranked_indices = np.argsort(-expr)
                
                # Calculate running sum
                running_sum = 0
                max_dev = 0
                min_dev = 0
                
                # Normalization factors
                sum_in_set = np.sum(np.abs(expr[list(gene_indices)]) ** alpha)
                
                if sum_in_set == 0:
                    cell_scores[cell] = 0
                    continue
                
                for i, gene_idx in enumerate(ranked_indices):
                    if gene_idx in gene_indices:
                        # Gene is in pathway
                        running_sum += (np.abs(expr[gene_idx]) ** alpha) / sum_in_set
                    else:
                        # Gene is not in pathway
                        running_sum -= 1.0 / n_other
                    
                    max_dev = max(max_dev, running_sum)
                    min_dev = min(min_dev, running_sum)
                
                # Enrichment score
                cell_scores[cell] = max_dev + min_dev
            
            scores[pathway] = cell_scores
        
        return pd.DataFrame(scores)
    
    def _weighted_score(
        self,
        X: np.ndarray,
        genes: List[str],
        rate_limiting_weight: float = 3.0
    ) -> pd.DataFrame:
        """
        Rate-limiting enzyme weighted scoring.
        
        Gives higher weight to rate-limiting enzymes which control flux.
        
        Parameters
        ----------
        X : np.ndarray
            Expression matrix
        genes : list
            Gene names
        rate_limiting_weight : float
            Weight multiplier for rate-limiting enzymes
            
        Returns
        -------
        pd.DataFrame
            Weighted scores
        """
        n_cells = X.shape[0]
        
        # Normalize
        lib_sizes = X.sum(axis=1, keepdims=True) + 1
        X_norm = np.log1p(X / lib_sizes * 10000)
        
        scores = {}
        
        for pathway, pathway_genes in self.pathways.items():
            gene_indices = []
            weights = []
            
            for g in pathway_genes:
                if g in genes:
                    gene_indices.append(genes.index(g))
                    # Higher weight for rate-limiting enzymes
                    if g in self.rate_limiting:
                        weights.append(rate_limiting_weight)
                    else:
                        weights.append(1.0)
            
            if len(gene_indices) < 3:
                continue
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            # Weighted average
            pathway_expr = X_norm[:, gene_indices]
            scores[pathway] = np.average(pathway_expr, axis=1, weights=weights)
        
        # Z-score normalize
        result = pd.DataFrame(scores)
        result = (result - result.mean()) / (result.std() + 1e-10)
        
        return result
    
    def _gsva_score(
        self,
        X: np.ndarray,
        genes: List[str],
        tau: float = 1.0
    ) -> pd.DataFrame:
        """
        GSVA-like scoring using kernel density estimation.
        
        Parameters
        ----------
        X : np.ndarray
            Expression matrix
        genes : list
            Gene names
        tau : float
            Parameter for expression threshold
            
        Returns
        -------
        pd.DataFrame
            GSVA scores
        """
        n_cells, n_genes = X.shape
        
        # Normalize and log transform
        lib_sizes = X.sum(axis=1, keepdims=True) + 1
        X_norm = np.log1p(X / lib_sizes * 10000)
        
        # Rank genes within each cell
        rankings = np.zeros_like(X_norm)
        for i in range(n_cells):
            rankings[i] = rankdata(X_norm[i], method='average')
        
        # Convert to density estimates
        rankings = (rankings - 0.5) / n_genes
        
        # Apply Gaussian CDF transformation
        rankings = norm.ppf(np.clip(rankings, 0.001, 0.999))
        
        scores = {}
        
        for pathway, pathway_genes in self.pathways.items():
            gene_indices = [genes.index(g) for g in pathway_genes if g in genes]
            
            if len(gene_indices) < 3:
                continue
            
            other_indices = [i for i in range(n_genes) if i not in gene_indices]
            
            # Compute enrichment-like score
            in_set = rankings[:, gene_indices]
            out_set = rankings[:, other_indices] if other_indices else np.zeros((n_cells, 1))
            
            # Difference of means
            scores[pathway] = np.mean(in_set, axis=1) - np.mean(out_set, axis=1)
        
        return pd.DataFrame(scores)
    
    def _combined_score(
        self,
        X: np.ndarray,
        genes: List[str],
        n_jobs: int = 1,
        methods: List[str] = None
    ) -> pd.DataFrame:
        """
        Combined ensemble scoring using multiple methods.
        
        Uses rank aggregation to combine scores from multiple methods,
        providing more robust estimates.
        
        Parameters
        ----------
        X : np.ndarray
            Expression matrix
        genes : list
            Gene names
        n_jobs : int
            Parallel jobs
        methods : list
            Methods to combine (default: all except combined)
            
        Returns
        -------
        pd.DataFrame
            Combined scores with confidence intervals
        """
        if methods is None:
            methods = ["vision", "weighted", "ssgsea"]
        
        all_scores = {}
        
        for method in methods:
            scores = self.score_pathways(X, genes, method=method)
            # Keep only score columns (not uncertainty columns)
            score_cols = [c for c in scores.columns if '_' not in c or c.count('_') == 0]
            all_scores[method] = scores[score_cols] if score_cols else scores
        
        # Find common pathways
        common_pathways = set(all_scores[methods[0]].columns)
        for method in methods[1:]:
            common_pathways &= set(all_scores[method].columns)
        
        common_pathways = sorted(common_pathways)
        
        # Rank aggregation
        combined = pd.DataFrame(index=range(X.shape[0]))
        
        for pathway in common_pathways:
            # Collect scores from each method
            pathway_scores = []
            for method in methods:
                if pathway in all_scores[method].columns:
                    # Z-score normalize within method
                    s = all_scores[method][pathway].values
                    s = (s - np.mean(s)) / (np.std(s) + 1e-10)
                    pathway_scores.append(s)
            
            if len(pathway_scores) >= 2:
                # Average of z-scores
                stacked = np.column_stack(pathway_scores)
                combined[pathway] = np.mean(stacked, axis=1)
                combined[f"{pathway}_std"] = np.std(stacked, axis=1)
            elif len(pathway_scores) == 1:
                combined[pathway] = pathway_scores[0]
                combined[f"{pathway}_std"] = 0.0
        
        return combined
    
    def _bootstrap_uncertainty(
        self,
        pathway_expr: np.ndarray,
        n_bootstrap: int = 100
    ) -> np.ndarray:
        """
        Estimate uncertainty via bootstrap.
        
        Parameters
        ----------
        pathway_expr : np.ndarray
            Expression matrix for pathway genes (cells x genes)
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns
        -------
        np.ndarray
            Standard deviation of bootstrap scores
        """
        n_cells, n_genes = pathway_expr.shape
        
        if n_genes < 3:
            return np.zeros(n_cells)
        
        bootstrap_scores = np.zeros((n_cells, n_bootstrap))
        
        for b in range(n_bootstrap):
            # Sample genes with replacement
            gene_idx = np.random.choice(n_genes, size=n_genes, replace=True)
            sampled = pathway_expr[:, gene_idx]
            bootstrap_scores[:, b] = np.mean(sampled, axis=1)
        
        return np.std(bootstrap_scores, axis=1)
    
    def get_method_recommendations(
        self,
        data_characteristics: Dict[str, any]
    ) -> str:
        """
        Recommend scoring method based on data characteristics.
        
        Parameters
        ----------
        data_characteristics : dict
            - n_cells: number of cells
            - sparsity: fraction of zeros
            - has_batches: whether data has batch effects
            - goal: 'classification', 'trajectory', 'differential'
            
        Returns
        -------
        str
            Recommended method with explanation
        """
        n_cells = data_characteristics.get('n_cells', 1000)
        sparsity = data_characteristics.get('sparsity', 0.9)
        has_batches = data_characteristics.get('has_batches', False)
        goal = data_characteristics.get('goal', 'trajectory')
        
        recommendations = []
        
        if sparsity > 0.95:
            recommendations.append(("aucell", "High sparsity data - rank-based method is robust"))
        
        if has_batches:
            recommendations.append(("vision", "Batch effects present - z-score normalization helps"))
        
        if goal == 'differential':
            recommendations.append(("gsva", "Differential analysis - GSVA designed for this"))
        elif goal == 'trajectory':
            recommendations.append(("weighted", "Trajectory analysis - rate-limiting enzymes important"))
        
        if n_cells > 10000:
            # Avoid computationally expensive methods
            recommendations = [(m, r) for m, r in recommendations if m != "combined"]
        
        if not recommendations:
            recommendations.append(("combined", "Default - ensemble method most robust"))
        
        return recommendations[0]


def compute_pathway_activity(
    adata,
    pathways: Dict[str, List[str]],
    rate_limiting: Optional[set] = None,
    method: str = "vision",
    layer: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to compute pathway activity scores.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    pathways : dict
        Pathway definitions
    rate_limiting : set, optional
        Rate-limiting enzymes
    method : str
        Scoring method
    layer : str, optional
        Expression layer
    **kwargs
        Additional arguments for scoring method
        
    Returns
    -------
    pd.DataFrame
        Pathway scores for each cell
    """
    from scipy.sparse import issparse
    
    # Get expression matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    
    if issparse(X):
        X = X.toarray()
    
    genes = adata.var_names.tolist()
    
    scorer = PathwayScoringMethods(pathways, rate_limiting)
    scores = scorer.score_pathways(X, genes, method=method, **kwargs)
    scores.index = adata.obs_names
    
    return scores
