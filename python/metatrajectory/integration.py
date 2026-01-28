"""
RNA Velocity Integration Module

Integrates metabolic trajectory analysis with RNA velocity results
to provide biological interpretation of developmental directions.

Key concept: We don't replace RNA velocity - we use it to provide
metabolic context for the directions it identifies.

This module provides:
- Decomposition of RNA velocity into metabolic pathway contributions
- Correlation analysis between metabolic and transcriptional dynamics
- Projection of velocity onto metabolic state space
- Combined visualization of velocity and metabolic states
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from typing import Optional, List, Dict, Tuple, Union
import warnings

from .knowledge_base import MetabolicKnowledgeBase
from .state_space import MetabolicStateSpace
from .utils import check_adata, cosine_similarity


class VelocityIntegration:
    """
    Integrate RNA velocity with metabolic trajectory analysis.
    
    This class provides methods to understand the metabolic implications
    of RNA velocity-inferred developmental directions.
    
    Key insight: RNA velocity tells us direction, metabolic analysis
    tells us the biological meaning of that direction.
    
    Parameters
    ----------
    state_space : MetabolicStateSpace
        Fitted metabolic state space
    knowledge_base : MetabolicKnowledgeBase
        Metabolic knowledge base
        
    Attributes
    ----------
    pathway_velocity : pd.DataFrame
        Velocity contribution by pathway
    velocity_metabolic_correlation : pd.DataFrame
        Correlation between velocity and metabolic changes
    metabolic_projection : np.ndarray
        Velocity projected onto metabolic space
        
    Examples
    --------
    >>> vi = VelocityIntegration(state_space, knowledge_base)
    >>> vi.fit(adata)  # adata should have velocity layer from scVelo
    >>> pathway_contrib = vi.get_pathway_velocity_contribution()
    """
    
    def __init__(
        self,
        state_space: MetabolicStateSpace,
        knowledge_base: MetabolicKnowledgeBase
    ):
        """
        Initialize VelocityIntegration.
        
        Parameters
        ----------
        state_space : MetabolicStateSpace
            Fitted metabolic state space
        knowledge_base : MetabolicKnowledgeBase
            Metabolic knowledge base
        """
        self.state_space = state_space
        self.kb = knowledge_base
        
        self.pathway_velocity: Optional[pd.DataFrame] = None
        self.velocity_metabolic_correlation: Optional[pd.DataFrame] = None
        self.metabolic_projection: Optional[np.ndarray] = None
        
        self._adata = None
        self._fitted = False
    
    def fit(self, adata, velocity_layer: str = "velocity") -> "VelocityIntegration":
        """
        Fit the velocity integration model.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data with RNA velocity computed (scVelo)
        velocity_layer : str
            Name of velocity layer in adata.layers
            
        Returns
        -------
        self
            Fitted VelocityIntegration
        """
        # Check for velocity
        if velocity_layer not in adata.layers:
            raise ValueError(
                f"Velocity layer '{velocity_layer}' not found. "
                "Run scVelo first: scv.tl.velocity(adata)"
            )
        
        self._adata = adata
        
        # Compute pathway-level velocity decomposition
        self.pathway_velocity = self._compute_pathway_velocity(adata, velocity_layer)
        
        # Compute correlation between velocity and metabolic states
        self.velocity_metabolic_correlation = self._compute_velocity_correlation(
            adata, velocity_layer
        )
        
        # Project velocity onto metabolic space
        self.metabolic_projection = self._project_velocity_to_metabolic(
            adata, velocity_layer
        )
        
        self._fitted = True
        return self
    
    def _compute_pathway_velocity(
        self,
        adata,
        velocity_layer: str
    ) -> pd.DataFrame:
        """
        Decompose RNA velocity into pathway contributions.
        
        For each pathway, compute the aggregate velocity of its genes,
        indicating whether the pathway is being upregulated or downregulated.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data with velocity
        velocity_layer : str
            Velocity layer name
            
        Returns
        -------
        pd.DataFrame
            Cell x Pathway velocity contribution matrix
        """
        velocity = adata.layers[velocity_layer]
        if issparse(velocity):
            velocity = velocity.toarray()
        
        genes = adata.var_names.tolist()
        cells = adata.obs_names.tolist()
        
        pathway_velocities = {}
        
        for pathway, pathway_genes in self.kb.pathways.items():
            # Find available genes
            available_genes = [g for g in pathway_genes if g in genes]
            
            if len(available_genes) < 3:
                continue
            
            # Get velocity for pathway genes
            gene_indices = [genes.index(g) for g in available_genes]
            pathway_vel = velocity[:, gene_indices]
            
            # Aggregate: mean velocity (positive = upregulating)
            # Weight by absolute expression to focus on expressed genes
            expr = adata.X
            if issparse(expr):
                expr = expr.toarray()
            expr_weights = np.abs(expr[:, gene_indices]).mean(axis=0)
            expr_weights = expr_weights / (expr_weights.sum() + 1e-10)
            
            # Weighted mean velocity
            pathway_velocities[pathway] = np.average(
                pathway_vel, axis=1, weights=expr_weights
            )
        
        return pd.DataFrame(pathway_velocities, index=cells)
    
    def _compute_velocity_correlation(
        self,
        adata,
        velocity_layer: str
    ) -> pd.DataFrame:
        """
        Compute correlation between velocity direction and metabolic changes.
        
        For each cell, correlate its velocity vector with the direction
        of change in metabolic state space.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data with velocity
        velocity_layer : str
            Velocity layer name
            
        Returns
        -------
        pd.DataFrame
            Correlation results
        """
        results = []
        
        if self.pathway_velocity is None:
            return pd.DataFrame()
        
        # For each metabolic axis, correlate with pathway velocity
        for axis_name, axis_def in self.kb.metabolic_axes.items():
            pos_genes = axis_def["positive"]
            neg_genes = axis_def["negative"]
            
            # Get velocity for axis genes (if in pathway velocity)
            pos_pathways = []
            neg_pathways = []
            
            for pathway in self.pathway_velocity.columns:
                pathway_genes = set(self.kb.pathways.get(pathway, []))
                
                pos_overlap = len(set(pos_genes) & pathway_genes)
                neg_overlap = len(set(neg_genes) & pathway_genes)
                
                if pos_overlap > neg_overlap:
                    pos_pathways.append(pathway)
                elif neg_overlap > pos_overlap:
                    neg_pathways.append(pathway)
            
            if len(pos_pathways) > 0 and len(neg_pathways) > 0:
                # Compute axis velocity: positive pathway velocity - negative
                pos_vel = self.pathway_velocity[pos_pathways].mean(axis=1)
                neg_vel = self.pathway_velocity[neg_pathways].mean(axis=1)
                axis_vel = pos_vel - neg_vel
                
                # Correlate with metabolic axis score
                axis_score = self.state_space.axis_scores[axis_name]
                
                corr, pval = pearsonr(axis_vel, axis_score)
                
                results.append({
                    'axis': axis_name,
                    'correlation': corr,
                    'p_value': pval,
                    'n_positive_pathways': len(pos_pathways),
                    'n_negative_pathways': len(neg_pathways)
                })
        
        return pd.DataFrame(results)
    
    def _project_velocity_to_metabolic(
        self,
        adata,
        velocity_layer: str
    ) -> np.ndarray:
        """
        Project RNA velocity onto metabolic state space.
        
        This allows visualization of velocity arrows in metabolic coordinates.
        
        Parameters
        ----------
        adata : AnnData
            Annotated data with velocity
        velocity_layer : str
            Velocity layer name
            
        Returns
        -------
        np.ndarray
            Velocity vectors in metabolic embedding space (n_cells x n_dims)
        """
        if self.pathway_velocity is None or self.pathway_velocity.empty:
            return np.zeros((adata.n_obs, 2))
        
        # Get pathway velocity matrix
        pathway_vel = self.pathway_velocity.values
        
        # The PCA used for embedding can project velocity
        if self.state_space._pca is not None:
            # Get pathway scores for alignment
            pathway_scores = self.state_space.pathway_scores
            
            # Align pathway velocity with pathway scores
            common_pathways = list(
                set(self.pathway_velocity.columns) & set(pathway_scores.columns)
            )
            
            if len(common_pathways) > 0:
                # Reorder to match
                vel_aligned = self.pathway_velocity[common_pathways].values
                
                # Scale velocity same as scores were scaled
                if self.state_space._scaler is not None:
                    # Don't center, just scale by same factors
                    scale = self.state_space._scaler.scale_
                    pathway_idx = [
                        list(pathway_scores.columns).index(p) 
                        for p in common_pathways
                    ]
                    vel_scaled = vel_aligned / scale[pathway_idx]
                else:
                    vel_scaled = vel_aligned
                
                # Project using PCA components
                pca = self.state_space._pca
                
                # Pad with zeros for missing pathways
                full_vel = np.zeros((adata.n_obs, len(pathway_scores.columns)))
                for i, p in enumerate(common_pathways):
                    col_idx = list(pathway_scores.columns).index(p)
                    full_vel[:, col_idx] = vel_scaled[:, i]
                
                # Transform (project onto PCs)
                metabolic_velocity = full_vel @ pca.components_.T
                
                return metabolic_velocity[:, :2]  # First 2 components
        
        # Fallback: use pathway velocity directly
        return pathway_vel[:, :2] if pathway_vel.shape[1] >= 2 else np.zeros((adata.n_obs, 2))
    
    def get_pathway_velocity_contribution(
        self,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Get pathway contributions to overall velocity.
        
        Parameters
        ----------
        normalize : bool
            Whether to normalize contributions to sum to 1
            
        Returns
        -------
        pd.DataFrame
            Pathway velocity contributions per cell
        """
        if self.pathway_velocity is None:
            raise ValueError("Velocity not computed. Call fit() first.")
        
        vel = self.pathway_velocity.copy()
        
        if normalize:
            abs_vel = np.abs(vel.values)
            row_sums = abs_vel.sum(axis=1, keepdims=True)
            vel = pd.DataFrame(
                abs_vel / (row_sums + 1e-10),
                index=vel.index,
                columns=vel.columns
            )
        
        return vel
    
    def get_top_velocity_pathways(
        self,
        direction: str = "both",
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get pathways with highest velocity magnitude.
        
        Parameters
        ----------
        direction : str
            'up' (positive velocity), 'down' (negative), or 'both'
        top_n : int
            Number of top pathways to return
            
        Returns
        -------
        pd.DataFrame
            Top pathways with mean velocity
        """
        if self.pathway_velocity is None:
            raise ValueError("Velocity not computed. Call fit() first.")
        
        # Mean velocity per pathway
        mean_vel = self.pathway_velocity.mean()
        
        if direction == "up":
            top_pathways = mean_vel.nlargest(top_n)
        elif direction == "down":
            top_pathways = mean_vel.nsmallest(top_n)
        else:
            # Sort by absolute value
            top_pathways = mean_vel.reindex(
                mean_vel.abs().nlargest(top_n).index
            )
        
        return pd.DataFrame({
            'pathway': top_pathways.index,
            'mean_velocity': top_pathways.values,
            'direction': ['up' if v > 0 else 'down' for v in top_pathways.values]
        })
    
    def get_velocity_metabolic_alignment(
        self,
        k_neighbors: int = 30
    ) -> np.ndarray:
        """
        Compute alignment between velocity direction and metabolic gradient.
        
        For each cell, compute the cosine similarity between its velocity
        direction and the gradient of metabolic state.
        
        Parameters
        ----------
        k_neighbors : int
            Number of neighbors for gradient estimation
            
        Returns
        -------
        np.ndarray
            Alignment scores per cell (-1 to 1)
        """
        if self.metabolic_projection is None:
            raise ValueError("Metabolic projection not computed. Call fit() first.")
        
        from sklearn.neighbors import NearestNeighbors
        
        embedding = self.state_space.embedding
        n_cells = embedding.shape[0]
        
        # Compute metabolic gradient for each cell
        nn = NearestNeighbors(n_neighbors=k_neighbors)
        nn.fit(embedding)
        _, indices = nn.kneighbors(embedding)
        
        alignment = np.zeros(n_cells)
        
        for i in range(n_cells):
            # Get neighbor positions
            neighbors = indices[i, 1:]  # Exclude self
            neighbor_positions = embedding[neighbors]
            
            # Estimate gradient: direction to neighbors with higher metabolic state
            # Using first axis score as metabolic state proxy
            if self.state_space.axis_scores is not None and len(self.state_space.axis_scores.columns) > 0:
                axis = self.state_space.axis_scores.columns[0]
                current_score = self.state_space.axis_scores[axis].iloc[i]
                neighbor_scores = self.state_space.axis_scores[axis].iloc[neighbors].values
                
                # Direction to higher-score neighbors
                higher_mask = neighbor_scores > current_score
                if higher_mask.sum() > 0:
                    gradient_dir = neighbor_positions[higher_mask].mean(axis=0) - embedding[i]
                    gradient_dir = gradient_dir / (np.linalg.norm(gradient_dir) + 1e-10)
                    
                    # Velocity direction
                    vel_dir = self.metabolic_projection[i]
                    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-10)
                    
                    # Cosine similarity
                    alignment[i] = np.dot(gradient_dir[:len(vel_dir)], vel_dir)
        
        return alignment
    
    def identify_metabolic_drivers_of_velocity(
        self,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Identify metabolic enzymes that drive velocity direction.
        
        Parameters
        ----------
        top_n : int
            Number of top drivers to return
            
        Returns
        -------
        pd.DataFrame
            Top metabolic enzyme drivers
        """
        if self._adata is None:
            raise ValueError("Not fitted. Call fit() first.")
        
        velocity = self._adata.layers.get("velocity")
        if velocity is None:
            raise ValueError("No velocity layer found.")
        
        if issparse(velocity):
            velocity = velocity.toarray()
        
        genes = self._adata.var_names.tolist()
        
        # Focus on rate-limiting enzymes
        enzyme_scores = []
        
        for enzyme in self.kb.rate_limiting_enzymes:
            if enzyme in genes:
                idx = genes.index(enzyme)
                
                # Mean absolute velocity
                mean_vel = velocity[:, idx].mean()
                abs_vel = np.abs(velocity[:, idx]).mean()
                
                # Get pathways for this enzyme
                pathways = self.kb.get_pathway_for_gene(enzyme)
                
                enzyme_scores.append({
                    'enzyme': enzyme,
                    'mean_velocity': mean_vel,
                    'abs_velocity': abs_vel,
                    'direction': 'up' if mean_vel > 0 else 'down',
                    'pathways': ', '.join(pathways[:3])
                })
        
        df = pd.DataFrame(enzyme_scores)
        
        if len(df) > 0:
            df = df.sort_values('abs_velocity', ascending=False).head(top_n)
        
        return df
    
    def store_in_adata(self, adata) -> None:
        """
        Store integration results in AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object to store results in
        """
        if not self._fitted:
            raise ValueError("Not fitted. Call fit() first.")
        
        # Store pathway velocities
        if self.pathway_velocity is not None:
            for pathway in self.pathway_velocity.columns:
                adata.obs[f"velocity_{pathway}"] = self.pathway_velocity[pathway].values
        
        # Store metabolic projection
        if self.metabolic_projection is not None:
            adata.obsm["velocity_metabolic"] = self.metabolic_projection
        
        # Store velocity-metabolic alignment
        alignment = self.get_velocity_metabolic_alignment()
        adata.obs["velocity_metabolic_alignment"] = alignment
    
    def summary(self) -> str:
        """Return summary of velocity integration."""
        lines = ["VelocityIntegration Summary:"]
        
        if self.pathway_velocity is not None:
            n_pathways = self.pathway_velocity.shape[1]
            lines.append(f"  - Pathways with velocity: {n_pathways}")
            
            # Top pathways
            top = self.get_top_velocity_pathways(top_n=3)
            for _, row in top.iterrows():
                lines.append(f"    - {row['pathway']}: {row['mean_velocity']:.3f} ({row['direction']})")
        
        if self.velocity_metabolic_correlation is not None and len(self.velocity_metabolic_correlation) > 0:
            lines.append("  - Velocity-Metabolic correlations:")
            for _, row in self.velocity_metabolic_correlation.iterrows():
                lines.append(f"    - {row['axis']}: r={row['correlation']:.3f} (p={row['p_value']:.2e})")
        
        if len(lines) == 1:
            lines.append("  - Not fitted yet")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.summary()
