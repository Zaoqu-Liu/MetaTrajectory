"""
Optimal Transport Module

Core module implementing optimal transport-based trajectory inference
for metabolic state transitions.

This module provides:
- Sinkhorn algorithm for entropic regularized OT
- Transport plan computation between cell populations
- Trajectory inference from time-series or pseudo-ordered data
- Fate probability estimation
- Wasserstein distance computation
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from typing import Optional, List, Dict, Union, Tuple
import warnings

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    warnings.warn("POT (Python Optimal Transport) not installed. Install with: pip install POT")

from .state_space import MetabolicStateSpace
from .utils import entropy, softmax


class MetabolicOT:
    """
    Optimal Transport-based metabolic trajectory analysis.
    
    This class implements trajectory inference using optimal transport theory,
    which provides a mathematically rigorous framework for analyzing how
    cell populations transition between metabolic states.
    
    Key advantages over pseudo-time methods:
    1. No assumption of continuous trajectories
    2. Probabilistic interpretation of fate transitions
    3. Handles branching and multi-lineage naturally
    4. Wasserstein distance as a meaningful metric
    
    Parameters
    ----------
    state_space : MetabolicStateSpace
        Fitted metabolic state space
        
    Attributes
    ----------
    transport_plans : dict
        Dictionary of computed transport plans
    transition_matrix : np.ndarray
        Cell-to-cell transition probability matrix
    fate_probabilities : pd.DataFrame
        Probability of each cell reaching terminal states
    wasserstein_distances : dict
        Pairwise Wasserstein distances between populations
        
    Examples
    --------
    >>> mot = MetabolicOT(state_space)
    >>> mot.compute_transport_matrix(time_points=[0, 1, 2])
    >>> fate_probs = mot.compute_fate_probabilities(terminal_states)
    """
    
    def __init__(self, state_space: MetabolicStateSpace):
        """
        Initialize MetabolicOT.
        
        Parameters
        ----------
        state_space : MetabolicStateSpace
            Fitted metabolic state space
        """
        if not HAS_POT:
            raise ImportError(
                "POT (Python Optimal Transport) is required. "
                "Install with: pip install POT"
            )
        
        self.state_space = state_space
        self.transport_plans: Dict[str, np.ndarray] = {}
        self.transition_matrix: Optional[np.ndarray] = None
        self.fate_probabilities: Optional[pd.DataFrame] = None
        self.wasserstein_distances: Dict[str, float] = {}
        
        # Internal state
        self._cost_matrix: Optional[np.ndarray] = None
        self._fitted = False
    
    def compute_cost_matrix(
        self,
        source_cells: Optional[np.ndarray] = None,
        target_cells: Optional[np.ndarray] = None,
        metric: str = "euclidean",
        use_network: bool = True
    ) -> np.ndarray:
        """
        Compute cost matrix for optimal transport.
        
        The cost matrix defines the "work" required to move mass from
        source to target locations in metabolic space.
        
        Parameters
        ----------
        source_cells : np.ndarray, optional
            Indices of source cells. If None, uses all cells.
        target_cells : np.ndarray, optional
            Indices of target cells. If None, uses all cells.
        metric : str
            Distance metric: 'euclidean', 'cosine', or 'metabolic'
        use_network : bool
            Whether to apply metabolic network weighting
            
        Returns
        -------
        np.ndarray
            Cost matrix (n_source x n_target)
        """
        embedding = self.state_space.embedding
        
        if source_cells is None:
            source_embedding = embedding
        else:
            source_embedding = embedding[source_cells]
        
        if target_cells is None:
            target_embedding = embedding
        else:
            target_embedding = embedding[target_cells]
        
        # Compute base distance
        if metric == "euclidean":
            cost = cdist(source_embedding, target_embedding, metric="euclidean")
        elif metric == "cosine":
            cost = cdist(source_embedding, target_embedding, metric="cosine")
        elif metric == "metabolic":
            # Use the network-weighted distance from state space
            cost = self.state_space.get_metabolic_distance(
                method="euclidean", use_network=use_network
            )
            if source_cells is not None:
                cost = cost[source_cells, :]
            if target_cells is not None:
                cost = cost[:, target_cells]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Normalize cost matrix
        cost = cost / (cost.max() + 1e-10)
        
        self._cost_matrix = cost
        return cost
    
    def compute_transport_plan(
        self,
        source_weights: Optional[np.ndarray] = None,
        target_weights: Optional[np.ndarray] = None,
        reg: float = 0.1,
        method: str = "sinkhorn",
        cost_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute optimal transport plan between source and target distributions.
        
        The transport plan T[i,j] represents the amount of "mass" transported
        from source cell i to target cell j.
        
        Parameters
        ----------
        source_weights : np.ndarray, optional
            Weights for source distribution. If None, uniform.
        target_weights : np.ndarray, optional
            Weights for target distribution. If None, uniform.
        reg : float
            Entropic regularization parameter. Higher = smoother transport.
        method : str
            OT method: 'sinkhorn' (fast, regularized) or 'emd' (exact)
        cost_matrix : np.ndarray, optional
            Pre-computed cost matrix. If None, uses stored matrix.
            
        Returns
        -------
        np.ndarray
            Transport plan matrix
        """
        if cost_matrix is None:
            if self._cost_matrix is None:
                raise ValueError("Cost matrix not computed. Call compute_cost_matrix() first.")
            cost_matrix = self._cost_matrix
        
        n_source, n_target = cost_matrix.shape
        
        # Set uniform weights if not provided
        if source_weights is None:
            source_weights = np.ones(n_source) / n_source
        if target_weights is None:
            target_weights = np.ones(n_target) / n_target
        
        # Normalize weights
        source_weights = source_weights / source_weights.sum()
        target_weights = target_weights / target_weights.sum()
        
        # Compute transport plan
        if method == "sinkhorn":
            transport_plan = ot.sinkhorn(
                source_weights, target_weights, cost_matrix,
                reg=reg, numItermax=1000, stopThr=1e-9
            )
        elif method == "emd":
            transport_plan = ot.emd(
                source_weights, target_weights, cost_matrix
            )
        elif method == "sinkhorn_stabilized":
            transport_plan = ot.sinkhorn_stabilized(
                source_weights, target_weights, cost_matrix,
                reg=reg, numItermax=1000
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return transport_plan
    
    def compute_trajectory_transport(
        self,
        time_labels: np.ndarray,
        reg: float = 0.1,
        growth_rate: Optional[np.ndarray] = None
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute transport plans between consecutive time points.
        
        This is the main function for trajectory inference from time-series data.
        
        Parameters
        ----------
        time_labels : np.ndarray
            Time point labels for each cell (e.g., [0, 0, 1, 1, 2, 2, ...])
        reg : float
            Entropic regularization parameter
        growth_rate : np.ndarray, optional
            Cell growth/death rates to adjust mass conservation
            
        Returns
        -------
        dict
            Dictionary mapping (t1, t2) to transport plan matrices
        """
        time_points = sorted(np.unique(time_labels))
        embedding = self.state_space.embedding
        
        self.transport_plans = {}
        
        for i in range(len(time_points) - 1):
            t1, t2 = time_points[i], time_points[i + 1]
            
            # Get cell indices for each time point
            source_idx = np.where(time_labels == t1)[0]
            target_idx = np.where(time_labels == t2)[0]
            
            # Compute cost matrix for this transition
            source_emb = embedding[source_idx]
            target_emb = embedding[target_idx]
            cost = cdist(source_emb, target_emb, metric="euclidean")
            cost = cost / (cost.max() + 1e-10)
            
            # Set weights (can incorporate growth rates)
            if growth_rate is not None:
                source_weights = growth_rate[source_idx]
                source_weights = source_weights / source_weights.sum()
            else:
                source_weights = np.ones(len(source_idx)) / len(source_idx)
            
            target_weights = np.ones(len(target_idx)) / len(target_idx)
            
            # Compute transport plan
            transport = self.compute_transport_plan(
                source_weights, target_weights, reg=reg, cost_matrix=cost
            )
            
            self.transport_plans[(t1, t2)] = {
                'plan': transport,
                'source_idx': source_idx,
                'target_idx': target_idx,
                'cost': cost
            }
        
        self._fitted = True
        return self.transport_plans
    
    def compute_transition_matrix(
        self,
        time_labels: Optional[np.ndarray] = None,
        k_neighbors: int = 30,
        sigma: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute cell-to-cell transition probability matrix.
        
        This matrix P[i,j] represents the probability that cell i
        transitions to cell j's state.
        
        Parameters
        ----------
        time_labels : np.ndarray, optional
            Time labels for time-series data
        k_neighbors : int
            Number of neighbors for kNN-based estimation
        sigma : float, optional
            Kernel bandwidth. If None, uses median distance.
            
        Returns
        -------
        np.ndarray
            Transition probability matrix (n_cells x n_cells)
        """
        embedding = self.state_space.embedding
        n_cells = embedding.shape[0]
        
        if time_labels is not None:
            # Use OT-based transition for time-series data
            if not self.transport_plans:
                self.compute_trajectory_transport(time_labels)
            
            # Build global transition matrix from individual transports
            self.transition_matrix = np.zeros((n_cells, n_cells))
            
            for (t1, t2), data in self.transport_plans.items():
                plan = data['plan']
                source_idx = data['source_idx']
                target_idx = data['target_idx']
                
                # Convert transport plan to transition probabilities
                # Normalize rows to get probabilities
                plan_normalized = plan / (plan.sum(axis=1, keepdims=True) + 1e-10)
                
                # Place in global matrix
                for i, si in enumerate(source_idx):
                    for j, tj in enumerate(target_idx):
                        self.transition_matrix[si, tj] = plan_normalized[i, j]
        
        else:
            # kNN-based transition for snapshot data
            from sklearn.neighbors import NearestNeighbors
            
            nn = NearestNeighbors(n_neighbors=k_neighbors)
            nn.fit(embedding)
            distances, indices = nn.kneighbors(embedding)
            
            # Determine sigma
            if sigma is None:
                sigma = np.median(distances[:, 1:])
            
            # Compute Gaussian kernel transitions
            self.transition_matrix = np.zeros((n_cells, n_cells))
            
            for i in range(n_cells):
                for k, j in enumerate(indices[i]):
                    if i != j:
                        d = distances[i, k]
                        self.transition_matrix[i, j] = np.exp(-d**2 / (2 * sigma**2))
            
            # Row normalize
            row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
            self.transition_matrix = self.transition_matrix / (row_sums + 1e-10)
        
        return self.transition_matrix
    
    def compute_fate_probabilities(
        self,
        terminal_states: np.ndarray,
        n_iterations: int = 100
    ) -> pd.DataFrame:
        """
        Compute probability of each cell reaching terminal states.
        
        Uses absorption probabilities in a Markov chain where terminal
        states are absorbing states.
        
        Parameters
        ----------
        terminal_states : np.ndarray
            Indices of terminal state cells
        n_iterations : int
            Number of iterations for power method
            
        Returns
        -------
        pd.DataFrame
            Cell x Terminal state probability matrix
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not computed. Call compute_transition_matrix() first.")
        
        n_cells = self.transition_matrix.shape[0]
        n_terminals = len(terminal_states)
        terminal_set = set(terminal_states)
        
        # Identify transient (non-terminal) states
        transient_states = [i for i in range(n_cells) if i not in terminal_set]
        n_transient = len(transient_states)
        
        if n_transient == 0:
            # All states are terminal
            fate_probs = np.zeros((n_cells, n_terminals))
            for i, t in enumerate(terminal_states):
                fate_probs[t, i] = 1.0
        else:
            # Use absorbing Markov chain theory:
            # B = N * R, where N = (I - Q)^-1 is the fundamental matrix
            # Q = transition submatrix among transient states
            # R = transition submatrix from transient to absorbing states
            
            P = self.transition_matrix
            
            # Extract Q (transient -> transient) and R (transient -> absorbing)
            Q = P[np.ix_(transient_states, transient_states)]
            R = P[np.ix_(transient_states, terminal_states)]
            
            # Compute fundamental matrix N = (I - Q)^-1
            # Use pseudo-inverse for numerical stability
            I = np.eye(n_transient)
            try:
                N = np.linalg.inv(I - Q)
            except np.linalg.LinAlgError:
                # Matrix is singular, use pseudo-inverse
                N = np.linalg.pinv(I - Q)
            
            # Absorption probabilities B = N * R
            B = N @ R
            
            # Clip to valid probability range [0, 1]
            B = np.clip(B, 0, 1)
            
            # Build full fate probability matrix
            fate_probs = np.zeros((n_cells, n_terminals))
            
            # Fill in transient state probabilities
            for idx, cell in enumerate(transient_states):
                fate_probs[cell, :] = B[idx, :]
            
            # Terminal states have probability 1 for their own fate
            for i, t in enumerate(terminal_states):
                fate_probs[t, :] = 0
                fate_probs[t, i] = 1.0
        
        # Normalize rows to sum to 1 (handle numerical errors)
        row_sums = fate_probs.sum(axis=1, keepdims=True)
        fate_probs = fate_probs / (row_sums + 1e-10)
        
        # Create DataFrame
        cell_names = self.state_space.pathway_scores.index.tolist()
        terminal_names = [f"Terminal_{i}" for i in range(n_terminals)]
        
        self.fate_probabilities = pd.DataFrame(
            fate_probs,
            index=cell_names,
            columns=terminal_names
        )
        
        return self.fate_probabilities
    
    def identify_bifurcation_cells(
        self,
        entropy_threshold: float = 0.8
    ) -> np.ndarray:
        """
        Identify cells at bifurcation points based on fate entropy.
        
        Cells with high fate entropy are undecided between multiple fates,
        suggesting they are at or near a bifurcation point.
        
        Parameters
        ----------
        entropy_threshold : float
            Threshold for identifying bifurcation (relative to max entropy)
            
        Returns
        -------
        np.ndarray
            Indices of bifurcation cells
        """
        if self.fate_probabilities is None:
            raise ValueError("Fate probabilities not computed. Call compute_fate_probabilities() first.")
        
        # Compute entropy for each cell
        probs = self.fate_probabilities.values
        cell_entropy = entropy(probs, axis=1)
        
        # Max possible entropy
        n_fates = probs.shape[1]
        max_entropy = np.log(n_fates)
        
        # Normalize entropy
        normalized_entropy = cell_entropy / max_entropy
        
        # Identify high-entropy (bifurcation) cells
        bifurcation_idx = np.where(normalized_entropy > entropy_threshold)[0]
        
        return bifurcation_idx
    
    def compute_wasserstein_distance(
        self,
        group1_idx: np.ndarray,
        group2_idx: np.ndarray,
        p: int = 2
    ) -> float:
        """
        Compute Wasserstein distance between two cell populations.
        
        The Wasserstein distance (Earth Mover's Distance) quantifies
        how different two populations are in metabolic state space.
        
        Parameters
        ----------
        group1_idx : np.ndarray
            Indices of first group
        group2_idx : np.ndarray
            Indices of second group
        p : int
            Wasserstein-p distance (default 2 for W2)
            
        Returns
        -------
        float
            Wasserstein distance
        """
        embedding = self.state_space.embedding
        
        # Validate inputs
        if len(group1_idx) == 0 or len(group2_idx) == 0:
            raise ValueError("Groups cannot be empty for Wasserstein distance computation")
        
        if p <= 0:
            raise ValueError("p must be positive")
        
        emb1 = embedding[group1_idx]
        emb2 = embedding[group2_idx]
        
        # Uniform weights
        a = np.ones(len(group1_idx)) / len(group1_idx)
        b = np.ones(len(group2_idx)) / len(group2_idx)
        
        # Cost matrix
        M = cdist(emb1, emb2, metric="euclidean") ** p
        
        # Compute Wasserstein distance
        W = ot.emd2(a, b, M)
        
        return W ** (1/p)
    
    def compute_pairwise_wasserstein(
        self,
        group_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute pairwise Wasserstein distances between groups.
        
        Parameters
        ----------
        group_labels : np.ndarray
            Group labels for each cell
            
        Returns
        -------
        pd.DataFrame
            Pairwise distance matrix
        """
        groups = np.unique(group_labels)
        n_groups = len(groups)
        
        distances = np.zeros((n_groups, n_groups))
        
        for i, g1 in enumerate(groups):
            for j, g2 in enumerate(groups):
                if i < j:
                    idx1 = np.where(group_labels == g1)[0]
                    idx2 = np.where(group_labels == g2)[0]
                    
                    d = self.compute_wasserstein_distance(idx1, idx2)
                    distances[i, j] = d
                    distances[j, i] = d
        
        self.wasserstein_distances = pd.DataFrame(
            distances,
            index=groups,
            columns=groups
        )
        
        return self.wasserstein_distances
    
    def get_transition_flow(
        self,
        source_idx: np.ndarray,
        target_idx: np.ndarray,
        top_k: int = 5
    ) -> pd.DataFrame:
        """
        Get top transition flows between source and target cells.
        
        Useful for understanding which cells transition to which.
        
        Parameters
        ----------
        source_idx : np.ndarray
            Source cell indices
        target_idx : np.ndarray
            Target cell indices
        top_k : int
            Number of top flows to return per source cell
            
        Returns
        -------
        pd.DataFrame
            DataFrame with source, target, and flow strength
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not computed.")
        
        flows = []
        
        for si in source_idx:
            # Get transition probabilities to target cells
            probs = self.transition_matrix[si, target_idx]
            
            # Get top k targets
            top_indices = np.argsort(probs)[-top_k:][::-1]
            
            for idx in top_indices:
                if probs[idx] > 0:
                    flows.append({
                        'source': si,
                        'target': target_idx[idx],
                        'probability': probs[idx]
                    })
        
        return pd.DataFrame(flows)
    
    def interpolate_trajectory(
        self,
        start_cell: int,
        end_cell: int,
        n_steps: int = 10
    ) -> np.ndarray:
        """
        Interpolate trajectory between two cells using OT geodesic.
        
        Parameters
        ----------
        start_cell : int
            Index of start cell
        end_cell : int
            Index of end cell
        n_steps : int
            Number of interpolation steps
            
        Returns
        -------
        np.ndarray
            Interpolated positions (n_steps x n_dims)
        """
        embedding = self.state_space.embedding
        
        start_pos = embedding[start_cell]
        end_pos = embedding[end_cell]
        
        # Linear interpolation in embedding space
        # (For more sophisticated geodesic, would need to solve OT at each step)
        t_values = np.linspace(0, 1, n_steps)
        trajectory = np.array([
            (1 - t) * start_pos + t * end_pos
            for t in t_values
        ])
        
        return trajectory
    
    def store_in_adata(self, adata) -> None:
        """
        Store computed results in AnnData object.
        
        Parameters
        ----------
        adata : AnnData
            AnnData object to store results in
        """
        # Store transition matrix
        if self.transition_matrix is not None:
            adata.obsp["metabolic_transition"] = csr_matrix(self.transition_matrix)
        
        # Store fate probabilities
        if self.fate_probabilities is not None:
            for col in self.fate_probabilities.columns:
                adata.obs[f"fate_{col}"] = self.fate_probabilities[col].values
            
            # Store fate entropy
            probs = self.fate_probabilities.values
            adata.obs["fate_entropy"] = entropy(probs, axis=1)
    
    def summary(self) -> str:
        """Return summary of the OT analysis."""
        lines = ["MetabolicOT Summary:"]
        
        if self.transition_matrix is not None:
            n_cells = self.transition_matrix.shape[0]
            sparsity = (self.transition_matrix > 1e-6).sum() / (n_cells * n_cells)
            lines.append(f"  - Transition matrix: {n_cells} x {n_cells}")
            lines.append(f"  - Transition sparsity: {sparsity:.2%}")
        
        if self.transport_plans:
            lines.append(f"  - Transport plans computed: {len(self.transport_plans)}")
        
        if self.fate_probabilities is not None:
            n_fates = self.fate_probabilities.shape[1]
            lines.append(f"  - Fate states: {n_fates}")
        
        if self.wasserstein_distances:
            lines.append(f"  - Wasserstein distances computed: {len(self.wasserstein_distances)}")
        
        if len(lines) == 1:
            lines.append("  - Not fitted yet")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.summary()
