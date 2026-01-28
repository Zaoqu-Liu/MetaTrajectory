"""
Network-Constrained Distance Module

Provides biologically meaningful distance metrics based on metabolic
network topology, rather than simple Euclidean distance.

Key concept: Two cells are "metabolically close" if they have similar
activity in connected pathways, not just similar average expression.

Distance types:
1. Network-weighted distance: Uses pathway connectivity
2. Flux-based distance: Based on metabolic flux constraints
3. Hierarchical distance: Considers metabolic module organization
4. Diffusion distance: Captures network propagation dynamics

References:
- Menche et al. (2015) Network-based disease module discovery
- Cowen et al. (2017) Network propagation
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist, squareform, pdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
from typing import Dict, List, Optional, Tuple, Union
import warnings


class MetabolicNetworkDistance:
    """
    Compute biologically meaningful distances using metabolic network topology.
    
    This class provides multiple network-aware distance metrics that capture
    the biological relationships between metabolic states better than
    simple Euclidean distance.
    
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        Metabolic pathway network
    pathway_scores : pd.DataFrame
        Pathway activity scores (cells x pathways)
        
    Attributes
    ----------
    pathway_distances : np.ndarray
        Pairwise pathway distances based on network
    diffusion_matrix : np.ndarray
        Network diffusion kernel
    """
    
    def __init__(
        self,
        network: Union[nx.Graph, nx.DiGraph],
        pathway_scores: pd.DataFrame
    ):
        self.network = network
        self.pathway_scores = pathway_scores
        
        # Extract pathways that are in both network and scores
        network_nodes = set(network.nodes())
        score_columns = set(pathway_scores.columns)
        self.common_pathways = sorted(network_nodes & score_columns)
        
        if len(self.common_pathways) < 2:
            warnings.warn(
                f"Only {len(self.common_pathways)} pathways in common between "
                "network and scores. Network distance may not be meaningful."
            )
        
        # Compute pathway-pathway distances
        self.pathway_distance_matrix = self._compute_pathway_distances()
        
        # Compute diffusion kernel
        self.diffusion_kernel = self._compute_diffusion_kernel()
    
    def _compute_pathway_distances(self) -> np.ndarray:
        """
        Compute pairwise distances between pathways based on network topology.
        
        Uses shortest path length, normalized by network diameter.
        
        Returns
        -------
        np.ndarray
            Pathway x Pathway distance matrix
        """
        n_pathways = len(self.common_pathways)
        distances = np.zeros((n_pathways, n_pathways))
        
        # Convert to undirected for shortest path
        if self.network.is_directed():
            G = self.network.to_undirected()
        else:
            G = self.network
        
        # Compute shortest paths
        for i, p1 in enumerate(self.common_pathways):
            for j, p2 in enumerate(self.common_pathways):
                if i == j:
                    distances[i, j] = 0
                else:
                    try:
                        path_length = nx.shortest_path_length(G, p1, p2)
                    except nx.NetworkXNoPath:
                        path_length = len(self.common_pathways)  # Max distance
                    distances[i, j] = path_length
        
        # Normalize by diameter
        diameter = distances.max()
        if diameter > 0:
            distances = distances / diameter
        
        return distances
    
    def _compute_diffusion_kernel(self, beta: float = 1.0) -> np.ndarray:
        """
        Compute diffusion kernel on pathway network.
        
        K = exp(-beta * L) where L is the graph Laplacian.
        
        Parameters
        ----------
        beta : float
            Diffusion time parameter
            
        Returns
        -------
        np.ndarray
            Pathway x Pathway diffusion kernel
        """
        n_pathways = len(self.common_pathways)
        
        if n_pathways < 2:
            return np.eye(n_pathways)
        
        # Build subgraph with common pathways
        subgraph = self.network.subgraph(self.common_pathways)
        
        # Get adjacency matrix
        A = nx.adjacency_matrix(subgraph, nodelist=self.common_pathways).toarray()
        
        # Degree matrix
        D = np.diag(A.sum(axis=1))
        
        # Laplacian
        L = D - A
        
        # Diffusion kernel
        K = expm(-beta * L)
        
        # Normalize rows to sum to 1
        K = K / (K.sum(axis=1, keepdims=True) + 1e-10)
        
        return K
    
    def network_weighted_distance(
        self,
        cells1: Optional[np.ndarray] = None,
        cells2: Optional[np.ndarray] = None,
        weight_by_connectivity: bool = True
    ) -> np.ndarray:
        """
        Compute network-weighted distance between cells.
        
        The distance between two cells considers not just the difference
        in pathway scores, but also the network relationships between
        differentially active pathways.
        
        d(c1, c2) = sqrt(sum_ij w_ij * (s1_i - s2_i) * (s1_j - s2_j))
        
        where w_ij reflects pathway connectivity.
        
        Parameters
        ----------
        cells1 : np.ndarray, optional
            Indices of first cell group (default: all cells)
        cells2 : np.ndarray, optional
            Indices of second cell group (default: same as cells1)
        weight_by_connectivity : bool
            Whether to weight by pathway connectivity
            
        Returns
        -------
        np.ndarray
            Distance matrix (n_cells1 x n_cells2)
        """
        # Get relevant scores
        scores = self.pathway_scores[self.common_pathways].values
        
        if cells1 is None:
            scores1 = scores
        else:
            scores1 = scores[cells1]
        
        if cells2 is None:
            scores2 = scores1
        else:
            scores2 = scores[cells2]
        
        n1, n2 = scores1.shape[0], scores2.shape[0]
        n_pathways = len(self.common_pathways)
        
        if weight_by_connectivity:
            # Weight matrix based on pathway connectivity
            # Higher weight for connected pathways
            W = 1 - self.pathway_distance_matrix
            W = (W + W.T) / 2  # Ensure symmetry
            np.fill_diagonal(W, 1)
        else:
            W = np.eye(n_pathways)
        
        # Compute Mahalanobis-like distance
        distances = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                diff = scores1[i] - scores2[j]
                # Quadratic form: d^T W d
                distances[i, j] = np.sqrt(np.abs(diff @ W @ diff))
        
        return distances
    
    def diffusion_distance(
        self,
        cells1: Optional[np.ndarray] = None,
        cells2: Optional[np.ndarray] = None,
        t: float = 1.0
    ) -> np.ndarray:
        """
        Compute diffusion distance between cells.
        
        First "smooths" pathway scores using network diffusion,
        then computes Euclidean distance in the diffused space.
        
        This captures the idea that similar activation of connected
        pathways should result in smaller distance.
        
        Parameters
        ----------
        cells1, cells2 : np.ndarray, optional
            Cell indices
        t : float
            Diffusion time
            
        Returns
        -------
        np.ndarray
            Distance matrix
        """
        scores = self.pathway_scores[self.common_pathways].values
        
        # Diffuse scores through network
        # Each cell's score vector is smoothed based on network structure
        K = self._compute_diffusion_kernel(beta=t)
        diffused_scores = scores @ K
        
        if cells1 is None:
            scores1 = diffused_scores
        else:
            scores1 = diffused_scores[cells1]
        
        if cells2 is None:
            scores2 = diffused_scores
        else:
            scores2 = diffused_scores[cells2]
        
        return cdist(scores1, scores2, metric='euclidean')
    
    def flux_constrained_distance(
        self,
        cells1: Optional[np.ndarray] = None,
        cells2: Optional[np.ndarray] = None,
        flux_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute distance respecting metabolic flux constraints.
        
        This distance emphasizes differences in rate-limiting steps
        and connected pathway activities that would affect flux.
        
        Parameters
        ----------
        cells1, cells2 : np.ndarray, optional
            Cell indices
        flux_weights : dict, optional
            Pathway -> importance weight for flux
            
        Returns
        -------
        np.ndarray
            Distance matrix
        """
        scores = self.pathway_scores[self.common_pathways].values
        n_pathways = len(self.common_pathways)
        
        # Default flux weights based on network centrality
        if flux_weights is None:
            # Use betweenness centrality as proxy for flux importance
            subgraph = self.network.subgraph(self.common_pathways)
            centrality = nx.betweenness_centrality(subgraph)
            
            flux_weights = np.array([
                centrality.get(p, 0.5) for p in self.common_pathways
            ])
            # Normalize
            flux_weights = flux_weights / (flux_weights.sum() + 1e-10)
        else:
            flux_weights = np.array([
                flux_weights.get(p, 0.5) for p in self.common_pathways
            ])
            flux_weights = flux_weights / (flux_weights.sum() + 1e-10)
        
        # Weight scores
        weighted_scores = scores * flux_weights[np.newaxis, :]
        
        if cells1 is None:
            scores1 = weighted_scores
        else:
            scores1 = weighted_scores[cells1]
        
        if cells2 is None:
            scores2 = weighted_scores
        else:
            scores2 = weighted_scores[cells2]
        
        return cdist(scores1, scores2, metric='euclidean')
    
    def hierarchical_distance(
        self,
        cells1: Optional[np.ndarray] = None,
        cells2: Optional[np.ndarray] = None,
        module_assignment: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Compute hierarchical distance respecting metabolic module organization.
        
        Metabolic pathways are organized hierarchically (e.g., central carbon
        metabolism contains glycolysis, TCA, PPP). This distance considers
        both within-module and between-module differences.
        
        Parameters
        ----------
        cells1, cells2 : np.ndarray, optional
            Cell indices
        module_assignment : dict, optional
            Pathway -> Module assignment
            
        Returns
        -------
        np.ndarray
            Distance matrix
        """
        scores = self.pathway_scores[self.common_pathways].values
        
        # Default module assignment based on network communities
        if module_assignment is None:
            subgraph = self.network.subgraph(self.common_pathways)
            # Detect communities
            try:
                communities = nx.community.louvain_communities(
                    subgraph.to_undirected() if subgraph.is_directed() else subgraph
                )
            except:
                # Fall back to single module
                communities = [set(self.common_pathways)]
            
            module_assignment = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    module_assignment[node] = f"Module_{i}"
        
        # Get unique modules
        modules = sorted(set(module_assignment.values()))
        
        # Compute module-level scores
        module_scores = np.zeros((scores.shape[0], len(modules)))
        
        for i, module in enumerate(modules):
            module_pathways = [
                self.common_pathways.index(p) 
                for p in self.common_pathways 
                if module_assignment.get(p) == module
            ]
            if module_pathways:
                module_scores[:, i] = scores[:, module_pathways].mean(axis=1)
        
        if cells1 is None:
            scores1 = module_scores
        else:
            scores1 = module_scores[cells1]
        
        if cells2 is None:
            scores2 = module_scores
        else:
            scores2 = module_scores[cells2]
        
        return cdist(scores1, scores2, metric='euclidean')
    
    def combined_distance(
        self,
        cells1: Optional[np.ndarray] = None,
        cells2: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Compute combined distance using multiple metrics.
        
        Combines network-weighted, diffusion, and flux-based distances
        for a more robust metabolic distance measure.
        
        Parameters
        ----------
        cells1, cells2 : np.ndarray, optional
            Cell indices
        weights : dict, optional
            Metric -> weight
            
        Returns
        -------
        np.ndarray
            Combined distance matrix
        """
        if weights is None:
            weights = {
                'network': 0.4,
                'diffusion': 0.3,
                'flux': 0.3
            }
        
        # Compute individual distances
        d_network = self.network_weighted_distance(cells1, cells2)
        d_diffusion = self.diffusion_distance(cells1, cells2)
        d_flux = self.flux_constrained_distance(cells1, cells2)
        
        # Normalize each to [0, 1]
        def normalize(d):
            d_min, d_max = d.min(), d.max()
            if d_max > d_min:
                return (d - d_min) / (d_max - d_min)
            return d
        
        d_network = normalize(d_network)
        d_diffusion = normalize(d_diffusion)
        d_flux = normalize(d_flux)
        
        # Weighted combination
        combined = (
            weights.get('network', 0.4) * d_network +
            weights.get('diffusion', 0.3) * d_diffusion +
            weights.get('flux', 0.3) * d_flux
        )
        
        return combined
    
    def get_optimal_transport_cost(
        self,
        distance_type: str = "combined",
        **kwargs
    ) -> np.ndarray:
        """
        Get cost matrix for optimal transport using network distance.
        
        Parameters
        ----------
        distance_type : str
            'euclidean', 'network', 'diffusion', 'flux', 'combined'
        **kwargs
            Arguments for distance computation
            
        Returns
        -------
        np.ndarray
            Cost matrix for OT
        """
        scores = self.pathway_scores[self.common_pathways].values
        
        if distance_type == "euclidean":
            cost = cdist(scores, scores, metric='euclidean')
        elif distance_type == "network":
            cost = self.network_weighted_distance(**kwargs)
        elif distance_type == "diffusion":
            cost = self.diffusion_distance(**kwargs)
        elif distance_type == "flux":
            cost = self.flux_constrained_distance(**kwargs)
        elif distance_type == "combined":
            cost = self.combined_distance(**kwargs)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")
        
        # Square for transport cost (Wasserstein-2)
        cost = cost ** 2
        
        # Normalize
        cost = cost / (cost.max() + 1e-10)
        
        return cost


def build_pathway_network_from_kb(
    pathways: Dict[str, List[str]],
    gene_overlap_threshold: float = 0.1
) -> nx.Graph:
    """
    Build pathway network based on gene overlap.
    
    Two pathways are connected if they share genes, with edge weight
    proportional to the Jaccard similarity of their gene sets.
    
    Parameters
    ----------
    pathways : dict
        Pathway -> gene list
    gene_overlap_threshold : float
        Minimum Jaccard similarity for an edge
        
    Returns
    -------
    nx.Graph
        Pathway network
    """
    G = nx.Graph()
    
    pathway_names = list(pathways.keys())
    G.add_nodes_from(pathway_names)
    
    # Compute pairwise Jaccard similarity
    for i, p1 in enumerate(pathway_names):
        genes1 = set(pathways[p1])
        for j, p2 in enumerate(pathway_names[i+1:], i+1):
            genes2 = set(pathways[p2])
            
            intersection = len(genes1 & genes2)
            union = len(genes1 | genes2)
            
            if union > 0:
                jaccard = intersection / union
                
                if jaccard >= gene_overlap_threshold:
                    G.add_edge(p1, p2, weight=jaccard)
    
    return G
