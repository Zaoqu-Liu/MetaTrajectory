"""
Metabolic State Space Module

Constructs the metabolic state space from single-cell expression data
using pathway scoring and network-constrained dimensionality reduction.

This module provides:
- Pathway activity scoring
- Metabolic axis scoring
- Network-constrained embedding
- State space visualization preparation
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Dict, Union, Tuple
import warnings

from .knowledge_base import MetabolicKnowledgeBase
from .utils import get_expression_matrix, check_adata


class MetabolicStateSpace:
    """
    Construct and analyze metabolic state space from single-cell data.
    
    This class transforms gene expression into a metabolic state space
    defined by pathway activities, with optional network constraints.
    
    Parameters
    ----------
    knowledge_base : MetabolicKnowledgeBase
        Metabolic knowledge base with pathway definitions
        
    Attributes
    ----------
    pathway_scores : pd.DataFrame
        Cell x Pathway score matrix
    axis_scores : pd.DataFrame
        Cell x Axis score matrix
    embedding : np.ndarray
        Low-dimensional embedding of metabolic states
    rate_limiting_scores : pd.DataFrame
        Scores for rate-limiting enzymes only
        
    Examples
    --------
    >>> from metatrajectory import MetabolicKnowledgeBase, MetabolicStateSpace
    >>> kb = MetabolicKnowledgeBase()
    >>> kb.load_kegg_pathways()
    >>> mss = MetabolicStateSpace(kb)
    >>> mss.fit(adata)
    >>> print(mss.pathway_scores.shape)
    """
    
    def __init__(self, knowledge_base: MetabolicKnowledgeBase):
        """
        Initialize MetabolicStateSpace.
        
        Parameters
        ----------
        knowledge_base : MetabolicKnowledgeBase
            Initialized metabolic knowledge base
        """
        self.kb = knowledge_base
        self.pathway_scores: Optional[pd.DataFrame] = None
        self.axis_scores: Optional[pd.DataFrame] = None
        self.embedding: Optional[np.ndarray] = None
        self.rate_limiting_scores: Optional[pd.DataFrame] = None
        self._adata = None
        self._fitted = False
        
        # PCA components for later projection
        self._pca = None
        self._scaler = None
    
    def fit(
        self,
        adata,
        layer: Optional[str] = None,
        scoring_method: str = "mean",
        min_genes: int = 3,
        scale: bool = True,
        n_components: int = 20
    ) -> "MetabolicStateSpace":
        """
        Fit the metabolic state space model.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix with gene expression
        layer : str, optional
            Layer to use for expression. If None, uses .X
        scoring_method : str
            Method for pathway scoring: 'mean', 'median', or 'zscore'
        min_genes : int
            Minimum genes required per pathway
        scale : bool
            Whether to z-score normalize pathway scores
        n_components : int
            Number of PCA components for embedding
            
        Returns
        -------
        self
            Fitted MetabolicStateSpace
        """
        check_adata(adata)
        self._adata = adata
        
        # Step 1: Compute pathway scores
        self.pathway_scores = self._compute_pathway_scores(
            adata, layer, scoring_method, min_genes
        )
        
        # Step 2: Compute metabolic axis scores
        self.axis_scores = self._compute_axis_scores(adata, layer)
        
        # Step 3: Compute rate-limiting enzyme scores
        self.rate_limiting_scores = self._compute_rate_limiting_scores(adata, layer)
        
        # Step 4: Scale if requested
        if scale:
            self._scaler = StandardScaler()
            scaled_scores = self._scaler.fit_transform(self.pathway_scores.values)
            self.pathway_scores = pd.DataFrame(
                scaled_scores,
                index=self.pathway_scores.index,
                columns=self.pathway_scores.columns
            )
        
        # Step 5: Compute embedding
        self.embedding = self._compute_embedding(n_components)
        
        self._fitted = True
        return self
    
    def _compute_pathway_scores(
        self,
        adata,
        layer: Optional[str],
        method: str,
        min_genes: int
    ) -> pd.DataFrame:
        """
        Compute pathway activity scores for each cell.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        layer : str, optional
            Expression layer to use
        method : str
            Scoring method
        min_genes : int
            Minimum genes per pathway
            
        Returns
        -------
        pd.DataFrame
            Cell x Pathway score matrix
        """
        # Get expression matrix
        X = get_expression_matrix(adata, layer)
        genes = adata.var_names.tolist()
        cells = adata.obs_names.tolist()
        
        scores = {}
        
        for pathway, pathway_genes in self.kb.pathways.items():
            # Find available genes
            available_genes = [g for g in pathway_genes if g in genes]
            
            if len(available_genes) < min_genes:
                warnings.warn(
                    f"Pathway '{pathway}' has only {len(available_genes)} genes, "
                    f"minimum is {min_genes}. Skipping."
                )
                continue
            
            # Get expression for pathway genes
            gene_indices = [genes.index(g) for g in available_genes]
            pathway_expr = X[:, gene_indices]
            
            # Compute score based on method
            if method == "mean":
                score = np.mean(pathway_expr, axis=1)
            elif method == "median":
                score = np.median(pathway_expr, axis=1)
            elif method == "zscore":
                # Z-score each gene, then average
                pathway_expr_z = zscore(pathway_expr, axis=0, nan_policy='omit')
                pathway_expr_z = np.nan_to_num(pathway_expr_z, nan=0.0)
                score = np.mean(pathway_expr_z, axis=1)
            else:
                raise ValueError(f"Unknown scoring method: {method}")
            
            scores[pathway] = score
        
        return pd.DataFrame(scores, index=cells)
    
    def _compute_axis_scores(
        self,
        adata,
        layer: Optional[str]
    ) -> pd.DataFrame:
        """
        Compute metabolic axis scores.
        
        Each axis represents a biological trade-off (e.g., glycolysis vs OXPHOS).
        Score = mean(positive genes) - mean(negative genes)
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        layer : str, optional
            Expression layer to use
            
        Returns
        -------
        pd.DataFrame
            Cell x Axis score matrix
        """
        X = get_expression_matrix(adata, layer)
        genes = adata.var_names.tolist()
        cells = adata.obs_names.tolist()
        
        axis_scores = {}
        
        for axis_name, axis_def in self.kb.metabolic_axes.items():
            pos_genes = [g for g in axis_def["positive"] if g in genes]
            neg_genes = [g for g in axis_def["negative"] if g in genes]
            
            if len(pos_genes) < 2 or len(neg_genes) < 2:
                warnings.warn(
                    f"Axis '{axis_name}' has insufficient genes. Skipping."
                )
                continue
            
            # Get expression
            pos_indices = [genes.index(g) for g in pos_genes]
            neg_indices = [genes.index(g) for g in neg_genes]
            
            pos_expr = np.mean(X[:, pos_indices], axis=1)
            neg_expr = np.mean(X[:, neg_indices], axis=1)
            
            # Axis score = positive - negative
            axis_scores[axis_name] = pos_expr - neg_expr
        
        df = pd.DataFrame(axis_scores, index=cells)
        
        # Z-score normalize (add epsilon to prevent division by zero)
        df = (df - df.mean()) / (df.std() + 1e-10)
        
        return df
    
    def _compute_rate_limiting_scores(
        self,
        adata,
        layer: Optional[str]
    ) -> pd.DataFrame:
        """
        Compute expression scores for rate-limiting enzymes.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        layer : str, optional
            Expression layer to use
            
        Returns
        -------
        pd.DataFrame
            Cell x Enzyme score matrix
        """
        X = get_expression_matrix(adata, layer)
        genes = adata.var_names.tolist()
        cells = adata.obs_names.tolist()
        
        # Find available rate-limiting enzymes
        available_enzymes = [
            g for g in self.kb.rate_limiting_enzymes if g in genes
        ]
        
        if len(available_enzymes) == 0:
            warnings.warn("No rate-limiting enzymes found in data.")
            return pd.DataFrame(index=cells)
        
        # Get expression
        enzyme_indices = [genes.index(g) for g in available_enzymes]
        enzyme_expr = X[:, enzyme_indices]
        
        df = pd.DataFrame(enzyme_expr, index=cells, columns=available_enzymes)
        
        # Z-score normalize
        df = (df - df.mean()) / (df.std() + 1e-10)
        
        return df
    
    def _compute_embedding(self, n_components: int) -> np.ndarray:
        """
        Compute low-dimensional embedding using PCA.
        
        Parameters
        ----------
        n_components : int
            Number of PCA components
            
        Returns
        -------
        np.ndarray
            Cell x Component embedding matrix
        """
        if self.pathway_scores is None:
            raise ValueError("Pathway scores not computed. Call fit() first.")
        
        # Limit components to available dimensions
        n_components = min(
            n_components,
            self.pathway_scores.shape[1],
            self.pathway_scores.shape[0] - 1
        )
        
        # PCA
        self._pca = PCA(n_components=n_components, random_state=42)
        embedding = self._pca.fit_transform(self.pathway_scores.values)
        
        return embedding
    
    def transform(self, adata, layer: Optional[str] = None) -> np.ndarray:
        """
        Transform new data into the fitted metabolic state space.
        
        Parameters
        ----------
        adata : AnnData
            New annotated data matrix
        layer : str, optional
            Expression layer to use
            
        Returns
        -------
        np.ndarray
            Embedding of new cells
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Compute pathway scores for new data
        new_scores = self._compute_pathway_scores(
            adata, layer, "mean", min_genes=3
        )
        
        # Ensure same pathways
        new_scores = new_scores.reindex(
            columns=self.pathway_scores.columns, fill_value=0
        )
        
        # Scale using fitted scaler
        if self._scaler is not None:
            new_scores_scaled = self._scaler.transform(new_scores.values)
        else:
            new_scores_scaled = new_scores.values
        
        # Transform using fitted PCA
        return self._pca.transform(new_scores_scaled)
    
    def get_metabolic_distance(
        self,
        method: str = "euclidean",
        use_network: bool = False
    ) -> np.ndarray:
        """
        Compute pairwise metabolic distances between cells.
        
        Parameters
        ----------
        method : str
            Distance metric: 'euclidean', 'cosine', or 'manhattan'
        use_network : bool
            Whether to weight distances by metabolic network structure
            
        Returns
        -------
        np.ndarray
            Pairwise distance matrix (n_cells x n_cells)
        """
        if self.embedding is None:
            raise ValueError("Embedding not computed. Call fit() first.")
        
        from scipy.spatial.distance import pdist, squareform
        
        # Compute raw distances
        if method == "euclidean":
            distances = squareform(pdist(self.embedding, metric="euclidean"))
        elif method == "cosine":
            distances = squareform(pdist(self.embedding, metric="cosine"))
        elif method == "manhattan":
            distances = squareform(pdist(self.embedding, metric="cityblock"))
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        # Optionally weight by metabolic network
        if use_network and self.kb.network is not None:
            distances = self._apply_network_weighting(distances)
        
        return distances
    
    def _apply_network_weighting(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply metabolic network-based weighting to distances.
        
        This method adjusts distances based on the metabolic network structure,
        penalizing transitions that are unlikely based on metabolic topology.
        
        Parameters
        ----------
        distances : np.ndarray
            Raw distance matrix
            
        Returns
        -------
        np.ndarray
            Network-weighted distance matrix
        """
        # Get pathway distance matrix
        pathway_distances = self.kb.get_pathway_distance_matrix(method="shortest_path")
        
        # Compute which pathways each cell is "closest to" based on high scores
        pathway_affinities = self._compute_pathway_affinities()
        
        n_cells = distances.shape[0]
        weighted_distances = distances.copy()
        
        # For each cell pair, compute pathway-based penalty
        for i in range(n_cells):
            for j in range(i + 1, n_cells):
                # Get dominant pathways for each cell
                top_pathways_i = pathway_affinities[i]
                top_pathways_j = pathway_affinities[j]
                
                # Compute average pathway distance
                pw_dist = 0
                count = 0
                for pw_i in top_pathways_i:
                    for pw_j in top_pathways_j:
                        if pw_i in pathway_distances.index and pw_j in pathway_distances.columns:
                            d = pathway_distances.loc[pw_i, pw_j]
                            if not np.isinf(d):
                                pw_dist += d
                                count += 1
                
                if count > 0:
                    pw_dist /= count
                    # Apply penalty: multiply by (1 + pathway_distance)
                    penalty = 1 + 0.1 * pw_dist
                    weighted_distances[i, j] *= penalty
                    weighted_distances[j, i] *= penalty
        
        return weighted_distances
    
    def _compute_pathway_affinities(self, top_k: int = 3) -> List[List[str]]:
        """
        Compute top pathways for each cell based on scores.
        
        Parameters
        ----------
        top_k : int
            Number of top pathways to return per cell
            
        Returns
        -------
        list of list of str
            Top pathway names for each cell
        """
        if self.pathway_scores is None:
            return [[] for _ in range(self._adata.n_obs)]
        
        affinities = []
        pathways = self.pathway_scores.columns.tolist()
        
        for i in range(self.pathway_scores.shape[0]):
            scores = self.pathway_scores.iloc[i].values
            top_indices = np.argsort(scores)[-top_k:][::-1]
            top_pathways = [pathways[idx] for idx in top_indices]
            affinities.append(top_pathways)
        
        return affinities
    
    def identify_metabolic_clusters(
        self,
        n_clusters: int = None,
        resolution: float = 1.0,
        method: str = "leiden"
    ) -> np.ndarray:
        """
        Identify metabolic clusters in the state space.
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters (for kmeans)
        resolution : float
            Resolution parameter (for leiden/louvain)
        method : str
            Clustering method: 'leiden', 'louvain', or 'kmeans'
            
        Returns
        -------
        np.ndarray
            Cluster labels for each cell
        """
        if self.embedding is None:
            raise ValueError("Embedding not computed. Call fit() first.")
        
        if method == "kmeans":
            from sklearn.cluster import KMeans
            
            if n_clusters is None:
                n_clusters = 5
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(self.embedding)
            
        elif method in ["leiden", "louvain"]:
            import scanpy as sc
            
            # Create temporary AnnData for clustering
            import anndata as ad
            temp_adata = ad.AnnData(X=self.embedding)
            
            # Compute neighbors
            sc.pp.neighbors(temp_adata, n_neighbors=30, use_rep="X")
            
            # Cluster
            if method == "leiden":
                sc.tl.leiden(temp_adata, resolution=resolution)
                clusters = temp_adata.obs["leiden"].astype(int).values
            else:
                sc.tl.louvain(temp_adata, resolution=resolution)
                clusters = temp_adata.obs["louvain"].astype(int).values
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        return clusters
    
    def get_cluster_profiles(
        self,
        clusters: np.ndarray,
        stat: str = "median"
    ) -> pd.DataFrame:
        """
        Compute metabolic profiles for each cluster.
        
        Parameters
        ----------
        clusters : np.ndarray
            Cluster labels
        stat : str
            Summary statistic: 'mean' or 'median'
            
        Returns
        -------
        pd.DataFrame
            Cluster x Pathway profile matrix
        """
        if self.pathway_scores is None:
            raise ValueError("Pathway scores not computed. Call fit() first.")
        
        scores = self.pathway_scores.copy()
        scores["cluster"] = clusters
        
        if stat == "mean":
            profiles = scores.groupby("cluster").mean()
        elif stat == "median":
            profiles = scores.groupby("cluster").median()
        else:
            raise ValueError(f"Unknown statistic: {stat}")
        
        return profiles
    
    def identify_terminal_states(
        self,
        n_states: int = 3,
        method: str = "extremes"
    ) -> np.ndarray:
        """
        Identify potential terminal metabolic states.
        
        Terminal states are defined as extremes in the metabolic state space,
        representing stable metabolic configurations.
        
        Parameters
        ----------
        n_states : int
            Number of terminal states to identify
        method : str
            Method: 'extremes' (axis extremes) or 'density' (low density regions)
            
        Returns
        -------
        np.ndarray
            Indices of cells representing terminal states
        """
        if self.axis_scores is None or self.axis_scores.empty:
            raise ValueError("Axis scores not computed or empty.")
        
        terminal_indices = []
        
        if method == "extremes":
            # Find cells at extremes of each metabolic axis
            for axis in self.axis_scores.columns:
                scores = self.axis_scores[axis].values
                
                # Most positive
                max_idx = np.argmax(scores)
                terminal_indices.append(max_idx)
                
                # Most negative
                min_idx = np.argmin(scores)
                terminal_indices.append(min_idx)
            
            # Remove duplicates and limit to n_states
            terminal_indices = list(set(terminal_indices))[:n_states]
            
        elif method == "density":
            from sklearn.neighbors import NearestNeighbors
            
            # Find low-density regions (potential attractors)
            nn = NearestNeighbors(n_neighbors=30)
            nn.fit(self.embedding)
            distances, _ = nn.kneighbors(self.embedding)
            
            # High mean distance = low density
            mean_distances = distances.mean(axis=1)
            
            # Get top n_states low-density cells
            terminal_indices = np.argsort(mean_distances)[-n_states:].tolist()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return np.array(terminal_indices)
    
    def store_in_adata(self, adata) -> None:
        """
        Store computed results in AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object to store results in
        """
        if not self._fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Store pathway scores
        for pathway in self.pathway_scores.columns:
            adata.obs[f"metabolic_{pathway}"] = self.pathway_scores[pathway].values
        
        # Store axis scores
        for axis in self.axis_scores.columns:
            adata.obs[f"axis_{axis}"] = self.axis_scores[axis].values
        
        # Store embedding
        adata.obsm["X_metabolic"] = self.embedding
        
        # Store rate-limiting enzyme scores
        if not self.rate_limiting_scores.empty:
            adata.obsm["X_rate_limiting"] = self.rate_limiting_scores.values
    
    def summary(self) -> str:
        """Return summary of the metabolic state space."""
        if not self._fitted:
            return "MetabolicStateSpace not fitted. Call fit() first."
        
        n_cells = self.pathway_scores.shape[0]
        n_pathways = self.pathway_scores.shape[1]
        n_axes = self.axis_scores.shape[1] if self.axis_scores is not None else 0
        n_enzymes = self.rate_limiting_scores.shape[1] if self.rate_limiting_scores is not None else 0
        n_components = self.embedding.shape[1] if self.embedding is not None else 0
        
        explained_var = sum(self._pca.explained_variance_ratio_) if self._pca else 0
        
        return (
            f"MetabolicStateSpace Summary:\n"
            f"  - Cells: {n_cells}\n"
            f"  - Pathways scored: {n_pathways}\n"
            f"  - Metabolic axes: {n_axes}\n"
            f"  - Rate-limiting enzymes: {n_enzymes}\n"
            f"  - Embedding dimensions: {n_components}\n"
            f"  - Variance explained: {explained_var:.2%}"
        )
    
    def __repr__(self) -> str:
        return self.summary()
