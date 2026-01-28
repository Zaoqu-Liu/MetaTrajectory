"""
Utility functions for MetaTrajectory.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, Optional, List, Dict, Tuple
import warnings


def check_adata(adata, require_umap: bool = False, require_velocity: bool = False):
    """
    Validate AnnData object has required attributes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    require_umap : bool
        Whether to require UMAP coordinates in obsm
    require_velocity : bool
        Whether to require velocity layer
        
    Raises
    ------
    ValueError
        If required attributes are missing
    """
    if adata.n_obs == 0:
        raise ValueError("AnnData object has no cells.")
    if adata.n_vars == 0:
        raise ValueError("AnnData object has no genes.")
    
    if require_umap:
        if "X_umap" not in adata.obsm:
            raise ValueError(
                "UMAP coordinates not found. Run `sc.tl.umap(adata)` first."
            )
    
    if require_velocity:
        if "velocity" not in adata.layers:
            raise ValueError(
                "Velocity layer not found. Run `scv.tl.velocity(adata)` first."
            )


def get_expression_matrix(
    adata,
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract expression matrix from AnnData.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    layer : str, optional
        Layer to use. If None, uses .X
    genes : list of str, optional
        Subset of genes to extract
        
    Returns
    -------
    np.ndarray
        Expression matrix (cells x genes)
    """
    if layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"Layer '{layer}' not found in adata.layers")
        X = adata.layers[layer]
    else:
        X = adata.X
    
    # Convert sparse to dense if needed
    if sparse.issparse(X):
        X = X.toarray()
    
    # Subset genes if specified
    if genes is not None:
        gene_mask = adata.var_names.isin(genes)
        X = X[:, gene_mask]
    
    return np.asarray(X)


def zscore_normalize(
    X: np.ndarray,
    axis: int = 0,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Z-score normalization.
    
    Parameters
    ----------
    X : np.ndarray
        Input matrix
    axis : int
        Axis along which to normalize (0=columns, 1=rows)
    eps : float
        Small constant to avoid division by zero
        
    Returns
    -------
    np.ndarray
        Z-score normalized matrix
    """
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    return (X - mean) / (std + eps)


def compute_knn_graph(
    X: np.ndarray,
    k: int = 30,
    metric: str = "euclidean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbor graph.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples x n_features)
    k : int
        Number of neighbors
    metric : str
        Distance metric
        
    Returns
    -------
    indices : np.ndarray
        KNN indices (n_samples x k)
    distances : np.ndarray
        KNN distances (n_samples x k)
    """
    from sklearn.neighbors import NearestNeighbors
    
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    return indices, distances


def compute_transition_matrix(
    distances: np.ndarray,
    sigma: Optional[float] = None
) -> np.ndarray:
    """
    Compute transition probability matrix from distances.
    
    Uses Gaussian kernel: P[i,j] = exp(-d[i,j]^2 / (2*sigma^2))
    
    Parameters
    ----------
    distances : np.ndarray
        Distance matrix (n x n) or KNN distances (n x k)
    sigma : float, optional
        Kernel bandwidth. If None, uses median distance.
        
    Returns
    -------
    np.ndarray
        Row-normalized transition matrix
    """
    if sigma is None:
        sigma = np.median(distances[distances > 0])
    
    # Gaussian kernel
    P = np.exp(-distances**2 / (2 * sigma**2))
    
    # Row normalize
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / (row_sums + 1e-10)
    
    return P


def entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-10) -> np.ndarray:
    """
    Compute Shannon entropy.
    
    Parameters
    ----------
    p : np.ndarray
        Probability distribution(s)
    axis : int
        Axis along which to compute entropy
    eps : float
        Small constant to avoid log(0)
        
    Returns
    -------
    np.ndarray
        Entropy values
    """
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def softmax(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax.
    
    Parameters
    ----------
    X : np.ndarray
        Input array
    axis : int
        Axis along which to compute softmax
        
    Returns
    -------
    np.ndarray
        Softmax probabilities
    """
    X_max = np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X - X_max)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)


def cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between rows of A and B.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix (n x d)
    B : np.ndarray
        Matrix (m x d)
        
    Returns
    -------
    np.ndarray
        Similarity matrix (n x m)
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-10)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-10)
    return A_norm @ B_norm.T


class MetaTrajectoryError(Exception):
    """Base exception for MetaTrajectory."""
    pass


class DataValidationError(MetaTrajectoryError):
    """Raised when input data validation fails."""
    pass


class ConvergenceError(MetaTrajectoryError):
    """Raised when optimization fails to converge."""
    pass
