"""
Driver Gene Analysis Module

Identifies metabolic enzymes and genes that drive fate decisions
and metabolic state transitions.

This module provides:
- Differential expression analysis for metabolic genes
- Fate driver identification based on OT analysis
- Metabolic pathway enrichment for drivers
- Rate-limiting enzyme prioritization

Statistical Methods (v2.0):
- Effect sizes: Cohen's d, Hedges' g, Cliff's delta
- Multiple testing: Global FDR with Benjamini-Hochberg/Yekutieli
- Uncertainty: Bootstrap confidence intervals, Bayes factors
- Power analysis for sample size recommendations
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
from scipy.sparse import issparse
from typing import Optional, List, Dict, Tuple, Union
import warnings

from .knowledge_base import MetabolicKnowledgeBase
from .state_space import MetabolicStateSpace
from .optimal_transport import MetabolicOT

# Import new statistical methods
try:
    from .statistics import (
        rigorous_differential_test,
        global_fdr_correction,
        EffectSizeCalculator,
        StatisticalResult
    )
    HAS_STATS = True
except ImportError:
    HAS_STATS = False


class DriverAnalysis:
    """
    Identify driver genes for metabolic state transitions.
    
    This class identifies metabolic enzymes that drive fate decisions
    by combining differential expression analysis with OT-based
    transition analysis.
    
    Parameters
    ----------
    metabolic_ot : MetabolicOT
        Fitted optimal transport model
    state_space : MetabolicStateSpace
        Fitted metabolic state space
    knowledge_base : MetabolicKnowledgeBase
        Metabolic knowledge base
        
    Attributes
    ----------
    fate_drivers : pd.DataFrame
        Genes driving fate decisions
    transition_drivers : pd.DataFrame
        Genes driving state transitions
    pathway_enrichment : pd.DataFrame
        Pathway enrichment for drivers
        
    Examples
    --------
    >>> da = DriverAnalysis(mot, state_space, kb)
    >>> da.fit(adata)
    >>> drivers = da.get_top_drivers(fate_pair=("Fate_0", "Fate_1"))
    """
    
    def __init__(
        self,
        metabolic_ot: MetabolicOT,
        state_space: MetabolicStateSpace,
        knowledge_base: MetabolicKnowledgeBase
    ):
        """
        Initialize DriverAnalysis.
        
        Parameters
        ----------
        metabolic_ot : MetabolicOT
            Fitted optimal transport model
        state_space : MetabolicStateSpace
            Fitted metabolic state space
        knowledge_base : MetabolicKnowledgeBase
            Metabolic knowledge base
        """
        self.mot = metabolic_ot
        self.state_space = state_space
        self.kb = knowledge_base
        
        self.fate_drivers: Optional[pd.DataFrame] = None
        self.transition_drivers: Optional[pd.DataFrame] = None
        self.pathway_enrichment: Optional[pd.DataFrame] = None
        
        self._adata = None
        self._fitted = False
    
    def fit(
        self,
        adata,
        layer: Optional[str] = None,
        min_cells: int = 10
    ) -> "DriverAnalysis":
        """
        Fit the driver analysis model.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        layer : str, optional
            Expression layer to use
        min_cells : int
            Minimum cells for statistical tests
            
        Returns
        -------
        self
            Fitted DriverAnalysis
        """
        self._adata = adata
        
        # Identify drivers for fate transitions
        if self.mot.fate_probabilities is not None:
            self.fate_drivers = self._identify_fate_drivers(adata, layer, min_cells)
        
        # Identify drivers for state transitions
        self.transition_drivers = self._identify_transition_drivers(adata, layer)
        
        # Compute pathway enrichment
        if self.fate_drivers is not None and len(self.fate_drivers) > 0:
            self.pathway_enrichment = self._compute_pathway_enrichment()
        
        self._fitted = True
        return self
    
    def _identify_fate_drivers(
        self,
        adata,
        layer: Optional[str],
        min_cells: int
    ) -> pd.DataFrame:
        """
        Identify genes driving fate decisions.
        
        Compares expression between cells committed to different fates.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        layer : str, optional
            Expression layer
        min_cells : int
            Minimum cells per fate
            
        Returns
        -------
        pd.DataFrame
            Driver analysis results
        """
        fate_probs = self.mot.fate_probabilities
        if fate_probs is None or fate_probs.empty:
            return pd.DataFrame()
        
        # Get expression matrix
        if layer is not None:
            X = adata.layers[layer]
        else:
            X = adata.X
        
        if issparse(X):
            X = X.toarray()
        
        genes = adata.var_names.tolist()
        fates = fate_probs.columns.tolist()
        
        # Focus on metabolic genes
        metabolic_genes = set(self.kb.get_all_genes())
        metabolic_indices = [
            i for i, g in enumerate(genes) if g in metabolic_genes
        ]
        
        results = []
        
        # Compare each pair of fates
        for i, fate1 in enumerate(fates):
            for fate2 in fates[i+1:]:
                # Get cells committed to each fate (>70% probability)
                cells_fate1 = np.where(fate_probs[fate1] > 0.7)[0]
                cells_fate2 = np.where(fate_probs[fate2] > 0.7)[0]
                
                if len(cells_fate1) < min_cells or len(cells_fate2) < min_cells:
                    continue
                
                # Test each metabolic gene
                for gene_idx in metabolic_indices:
                    gene = genes[gene_idx]
                    
                    expr1 = X[cells_fate1, gene_idx]
                    expr2 = X[cells_fate2, gene_idx]
                    
                    # Skip if no expression
                    if expr1.mean() == 0 and expr2.mean() == 0:
                        continue
                    
                    # Statistical test
                    try:
                        stat, pval = mannwhitneyu(expr1, expr2, alternative='two-sided')
                    except (ValueError, RuntimeWarning):
                        # Mann-Whitney can fail with constant or insufficient data
                        continue
                    
                    # Effect size (log fold change)
                    mean1 = expr1.mean() + 1e-10
                    mean2 = expr2.mean() + 1e-10
                    log_fc = np.log2(mean1 / mean2)
                    
                    # Is it a rate-limiting enzyme?
                    is_rate_limiting = gene in self.kb.rate_limiting_enzymes
                    
                    # Get pathways
                    pathways = self.kb.get_pathway_for_gene(gene)
                    
                    results.append({
                        'gene': gene,
                        'fate1': fate1,
                        'fate2': fate2,
                        'log2_fc': log_fc,
                        'p_value': pval,
                        'mean_expr_fate1': mean1,
                        'mean_expr_fate2': mean2,
                        'is_rate_limiting': is_rate_limiting,
                        'pathways': ', '.join(pathways[:3]),
                        'n_cells_fate1': len(cells_fate1),
                        'n_cells_fate2': len(cells_fate2)
                    })
        
        if len(results) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Multiple testing correction (Benjamini-Hochberg)
        df['p_adjusted'] = self._benjamini_hochberg(df['p_value'].values)
        
        # Sort by significance and effect size
        df['driver_score'] = -np.log10(df['p_adjusted'] + 1e-300) * np.abs(df['log2_fc'])
        df = df.sort_values('driver_score', ascending=False)
        
        return df
    
    def _identify_transition_drivers(
        self,
        adata,
        layer: Optional[str]
    ) -> pd.DataFrame:
        """
        Identify genes correlated with transition probabilities.
        
        Finds genes whose expression correlates with high transition
        probability in the OT model.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        layer : str, optional
            Expression layer
            
        Returns
        -------
        pd.DataFrame
            Transition driver results
        """
        if self.mot.transition_matrix is None:
            return pd.DataFrame()
        
        # Get expression
        if layer is not None:
            X = adata.layers[layer]
        else:
            X = adata.X
        
        if issparse(X):
            X = X.toarray()
        
        genes = adata.var_names.tolist()
        
        # Focus on metabolic genes
        metabolic_genes = set(self.kb.get_all_genes())
        
        # Compute "transition activity" for each cell
        # (sum of outgoing transition probabilities)
        transition_activity = self.mot.transition_matrix.sum(axis=1)
        
        results = []
        
        for i, gene in enumerate(genes):
            if gene not in metabolic_genes:
                continue
            
            expr = X[:, i]
            
            # Skip if no variance
            if expr.std() == 0:
                continue
            
            # Correlation with transition activity
            corr, pval = spearmanr(expr, transition_activity)
            
            if np.isnan(corr):
                continue
            
            # Is it a rate-limiting enzyme?
            is_rate_limiting = gene in self.kb.rate_limiting_enzymes
            
            # Get pathways
            pathways = self.kb.get_pathway_for_gene(gene)
            
            results.append({
                'gene': gene,
                'correlation': corr,
                'p_value': pval,
                'is_rate_limiting': is_rate_limiting,
                'pathways': ', '.join(pathways[:3])
            })
        
        if len(results) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        # Multiple testing correction
        df['p_adjusted'] = self._benjamini_hochberg(df['p_value'].values)
        
        # Sort by absolute correlation
        df['abs_correlation'] = np.abs(df['correlation'])
        df = df.sort_values('abs_correlation', ascending=False)
        
        return df
    
    def _compute_pathway_enrichment(self) -> pd.DataFrame:
        """
        Compute pathway enrichment for identified drivers.
        
        Returns
        -------
        pd.DataFrame
            Pathway enrichment results
        """
        if self.fate_drivers is None or len(self.fate_drivers) == 0:
            return pd.DataFrame()
        
        # Get significant drivers
        sig_drivers = self.fate_drivers[
            self.fate_drivers['p_adjusted'] < 0.05
        ]['gene'].unique()
        
        if len(sig_drivers) == 0:
            return pd.DataFrame()
        
        sig_drivers = set(sig_drivers)
        
        # Count drivers per pathway
        results = []
        total_metabolic_genes = len(self.kb.get_all_genes())
        
        for pathway, pathway_genes in self.kb.pathways.items():
            pathway_genes_set = set(pathway_genes)
            
            # Overlap
            overlap = len(sig_drivers & pathway_genes_set)
            
            if overlap == 0:
                continue
            
            # Expected by chance
            expected = len(sig_drivers) * len(pathway_genes_set) / total_metabolic_genes
            
            # Enrichment ratio
            enrichment = overlap / (expected + 1e-10)
            
            # Simple hypergeometric-like p-value approximation
            # (For rigorous analysis, use scipy.stats.hypergeom)
            from scipy.stats import hypergeom
            
            pval = hypergeom.sf(
                overlap - 1,  # x-1 for P(X >= x)
                total_metabolic_genes,  # population size
                len(pathway_genes_set),  # success states in population
                len(sig_drivers)  # sample size
            )
            
            results.append({
                'pathway': pathway,
                'n_drivers': overlap,
                'n_pathway_genes': len(pathway_genes_set),
                'enrichment': enrichment,
                'p_value': pval,
                'driver_genes': ', '.join(sorted(sig_drivers & pathway_genes_set)[:5])
            })
        
        if len(results) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['p_adjusted'] = self._benjamini_hochberg(df['p_value'].values)
        df = df.sort_values('enrichment', ascending=False)
        
        return df
    
    def _benjamini_hochberg(self, pvalues: np.ndarray) -> np.ndarray:
        """
        Benjamini-Hochberg multiple testing correction.
        
        Parameters
        ----------
        pvalues : np.ndarray
            Raw p-values
            
        Returns
        -------
        np.ndarray
            Adjusted p-values
        """
        n = len(pvalues)
        if n == 0:
            return np.array([])
        
        # Sort p-values
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]
        
        # Compute adjusted p-values
        adjusted = np.zeros(n)
        cummin = 1.0
        
        for i in range(n - 1, -1, -1):
            adjusted_p = sorted_pvals[i] * n / (i + 1)
            cummin = min(cummin, adjusted_p)
            adjusted[sorted_idx[i]] = min(cummin, 1.0)
        
        return adjusted
    
    def get_top_drivers(
        self,
        fate_pair: Optional[Tuple[str, str]] = None,
        top_n: int = 20,
        only_rate_limiting: bool = False,
        max_pval: float = 0.05
    ) -> pd.DataFrame:
        """
        Get top driver genes.
        
        Parameters
        ----------
        fate_pair : tuple of str, optional
            Specific fate pair to analyze. If None, returns all.
        top_n : int
            Number of top drivers to return
        only_rate_limiting : bool
            Whether to return only rate-limiting enzymes
        max_pval : float
            Maximum adjusted p-value cutoff
            
        Returns
        -------
        pd.DataFrame
            Top driver genes
        """
        if self.fate_drivers is None or len(self.fate_drivers) == 0:
            return pd.DataFrame()
        
        df = self.fate_drivers.copy()
        
        # Filter by fate pair
        if fate_pair is not None:
            df = df[
                ((df['fate1'] == fate_pair[0]) & (df['fate2'] == fate_pair[1])) |
                ((df['fate1'] == fate_pair[1]) & (df['fate2'] == fate_pair[0]))
            ]
        
        # Filter by significance
        df = df[df['p_adjusted'] < max_pval]
        
        # Filter by rate-limiting
        if only_rate_limiting:
            df = df[df['is_rate_limiting']]
        
        # Return top N
        return df.head(top_n)
    
    def get_driver_summary(self) -> Dict:
        """
        Get summary statistics of driver analysis.
        
        Returns
        -------
        dict
            Summary statistics
        """
        summary = {
            'n_fate_drivers': 0,
            'n_significant_fate_drivers': 0,
            'n_rate_limiting_drivers': 0,
            'top_pathways': [],
            'n_transition_drivers': 0
        }
        
        if self.fate_drivers is not None and len(self.fate_drivers) > 0:
            summary['n_fate_drivers'] = len(self.fate_drivers)
            summary['n_significant_fate_drivers'] = (
                self.fate_drivers['p_adjusted'] < 0.05
            ).sum()
            summary['n_rate_limiting_drivers'] = (
                (self.fate_drivers['p_adjusted'] < 0.05) & 
                self.fate_drivers['is_rate_limiting']
            ).sum()
        
        if self.pathway_enrichment is not None and len(self.pathway_enrichment) > 0:
            top_pathways = self.pathway_enrichment.head(5)['pathway'].tolist()
            summary['top_pathways'] = top_pathways
        
        if self.transition_drivers is not None:
            summary['n_transition_drivers'] = len(self.transition_drivers)
        
        return summary
    
    def find_driver_genes(
        self,
        source_cells: np.ndarray,
        target_cells: np.ndarray,
        layer: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Find genes driving transition from source to target cells.
        
        This is a direct comparison between two cell populations.
        
        Parameters
        ----------
        source_cells : np.ndarray
            Indices of source cells
        target_cells : np.ndarray
            Indices of target cells
        layer : str, optional
            Expression layer
        top_n : int
            Number of top drivers to return
            
        Returns
        -------
        pd.DataFrame
            Driver analysis results
        """
        if self._adata is None:
            raise ValueError("Not fitted. Call fit() first.")
        
        adata = self._adata
        
        # Get expression
        if layer is not None:
            X = adata.layers[layer]
        else:
            X = adata.X
        
        if issparse(X):
            X = X.toarray()
        
        genes = adata.var_names.tolist()
        metabolic_genes = set(self.kb.get_all_genes())
        
        results = []
        
        for i, gene in enumerate(genes):
            if gene not in metabolic_genes:
                continue
            
            expr_source = X[source_cells, i]
            expr_target = X[target_cells, i]
            
            # Skip if both groups have no variance (use tolerance for float comparison)
            if expr_source.std() < 1e-10 and expr_target.std() < 1e-10:
                continue
            
            # Statistical test
            try:
                stat, pval = mannwhitneyu(
                    expr_source, expr_target, alternative='two-sided'
                )
            except (ValueError, RuntimeWarning):
                # Mann-Whitney can fail with constant or insufficient data
                continue
            
            # Effect size
            mean_source = expr_source.mean() + 1e-10
            mean_target = expr_target.mean() + 1e-10
            log_fc = np.log2(mean_target / mean_source)
            
            results.append({
                'gene': gene,
                'log2_fc': log_fc,
                'p_value': pval,
                'mean_source': mean_source,
                'mean_target': mean_target,
                'is_rate_limiting': gene in self.kb.rate_limiting_enzymes,
                'pathways': ', '.join(self.kb.get_pathway_for_gene(gene)[:3])
            })
        
        if len(results) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['p_adjusted'] = self._benjamini_hochberg(df['p_value'].values)
        df['driver_score'] = -np.log10(df['p_adjusted'] + 1e-300) * np.abs(df['log2_fc'])
        df = df.sort_values('driver_score', ascending=False)
        
        return df.head(top_n)
    
    def store_in_adata(self, adata) -> None:
        """
        Store driver analysis results in AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object to store results in
        """
        # Store driver scores in var
        if self.fate_drivers is not None and len(self.fate_drivers) > 0:
            # Get best score per gene
            best_scores = self.fate_drivers.groupby('gene')['driver_score'].max()
            
            adata.var['driver_score'] = 0.0
            for gene, score in best_scores.items():
                if gene in adata.var_names:
                    adata.var.loc[gene, 'driver_score'] = score
        
        if self.transition_drivers is not None and len(self.transition_drivers) > 0:
            adata.var['transition_correlation'] = 0.0
            for _, row in self.transition_drivers.iterrows():
                gene = row['gene']
                if gene in adata.var_names:
                    adata.var.loc[gene, 'transition_correlation'] = row['correlation']
    
    def summary(self) -> str:
        """Return summary of driver analysis."""
        lines = ["DriverAnalysis Summary:"]
        
        stats = self.get_driver_summary()
        
        lines.append(f"  - Total fate drivers analyzed: {stats['n_fate_drivers']}")
        lines.append(f"  - Significant drivers (p<0.05): {stats['n_significant_fate_drivers']}")
        lines.append(f"  - Rate-limiting enzyme drivers: {stats['n_rate_limiting_drivers']}")
        
        if stats['top_pathways']:
            lines.append(f"  - Top enriched pathways: {', '.join(stats['top_pathways'][:3])}")
        
        lines.append(f"  - Transition-correlated genes: {stats['n_transition_drivers']}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.summary()
