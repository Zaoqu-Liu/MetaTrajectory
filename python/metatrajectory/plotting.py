"""
Visualization Module

Provides publication-quality visualizations for metabolic trajectory analysis.

This module includes:
- Metabolic landscape plots
- Transport flow visualizations
- Fate probability plots
- Driver gene bar plots
- Pathway heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
import warnings

# Default color palettes
METABOLIC_PALETTE = [
    "#282A62", "#692F7C", "#B43970",
    "#D96558", "#EFA143", "#F6C63C"
]

FATE_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087",
    "#3C5488", "#F39B7F", "#8491B4"
]

DRIVER_PALETTE = {
    "up": "#E64B35",
    "down": "#4DBBD5"
}


def plot_metabolic_landscape(
    adata,
    color: str = "axis_Glycolysis_vs_OXPHOS",
    embedding_key: str = "X_umap",
    title: str = "Metabolic Landscape",
    cmap: str = "RdBu_r",
    size: float = 10,
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (8, 8),
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Plot metabolic landscape on UMAP/embedding.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding and metabolic scores
    color : str
        Column in adata.obs to color by
    embedding_key : str
        Key in adata.obsm for embedding coordinates
    title : str
        Plot title
    cmap : str
        Colormap name
    size : float
        Point size
    alpha : float
        Point transparency
    figsize : tuple
        Figure size
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
    ax : plt.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    plt.Axes or None
        Axes object if show=False
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get embedding
    if embedding_key not in adata.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in adata.obsm")
    
    embedding = adata.obsm[embedding_key]
    
    # Get color values
    if color in adata.obs.columns:
        values = adata.obs[color].values
    elif color in adata.var_names:
        values = adata[:, color].X.toarray().flatten() if hasattr(adata[:, color].X, 'toarray') else adata[:, color].X.flatten()
    else:
        raise ValueError(f"Color '{color}' not found in adata.obs or var_names")
    
    # Plot
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        cmap=cmap,
        s=size,
        alpha=alpha,
        edgecolors='none'
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(color.replace('_', ' '), fontsize=10)
    
    # Formatting
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return ax


def plot_transport_flow(
    adata,
    transport_matrix: np.ndarray,
    embedding_key: str = "X_umap",
    top_k: int = 3,
    min_weight: float = 0.01,
    arrow_scale: float = 1.0,
    title: str = "Transport Flow",
    figsize: Tuple[float, float] = (10, 10),
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Plot optimal transport flow on embedding.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding
    transport_matrix : np.ndarray
        Transport/transition probability matrix
    embedding_key : str
        Key in adata.obsm for embedding
    top_k : int
        Number of top transitions per cell to show
    min_weight : float
        Minimum transition weight to display
    arrow_scale : float
        Scale factor for arrows
    title : str
        Plot title
    figsize : tuple
        Figure size
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
    ax : plt.Axes, optional
        Existing axes to plot on
        
    Returns
    -------
    plt.Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]
    
    # Collect flow vectors
    flows = []
    weights = []
    
    for i in range(n_cells):
        # Get top k transitions
        trans_probs = transport_matrix[i, :]
        top_indices = np.argsort(trans_probs)[-top_k:]
        
        for j in top_indices:
            if i != j and trans_probs[j] > min_weight:
                flows.append([
                    embedding[i, 0], embedding[i, 1],
                    embedding[j, 0], embedding[j, 1]
                ])
                weights.append(trans_probs[j])
    
    if len(flows) > 0:
        flows = np.array(flows)
        weights = np.array(weights)
        
        # Normalize weights for visualization
        weights_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        
        # Plot arrows
        for k in range(len(flows)):
            ax.annotate(
                '',
                xy=(flows[k, 2], flows[k, 3]),
                xytext=(flows[k, 0], flows[k, 1]),
                arrowprops=dict(
                    arrowstyle='->',
                    color=plt.cm.viridis(weights_norm[k]),
                    alpha=0.3 + 0.4 * weights_norm[k],
                    lw=0.5 + weights_norm[k] * arrow_scale
                )
            )
    
    # Plot cells
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c='lightgray',
        s=10,
        alpha=0.5,
        edgecolors='none'
    )
    
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return ax


def plot_fate_probabilities(
    adata,
    fate_probs: pd.DataFrame,
    embedding_key: str = "X_umap",
    title: str = "Fate Probabilities",
    figsize: Tuple[float, float] = (12, 4),
    show: bool = True,
    save: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot fate probabilities as separate panels.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding
    fate_probs : pd.DataFrame
        Cell x Fate probability matrix
    embedding_key : str
        Key in adata.obsm for embedding
    title : str
        Overall title
    figsize : tuple
        Figure size
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure or None
    """
    embedding = adata.obsm[embedding_key]
    n_fates = fate_probs.shape[1]
    
    fig, axes = plt.subplots(1, n_fates, figsize=figsize)
    if n_fates == 1:
        axes = [axes]
    
    for i, fate in enumerate(fate_probs.columns):
        ax = axes[i]
        
        probs = fate_probs[fate].values
        
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=probs,
            cmap='Purples',
            s=10,
            alpha=0.8,
            edgecolors='none',
            vmin=0,
            vmax=1
        )
        
        ax.set_title(fate, fontsize=12)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Probability')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return fig


def plot_fate_pie(
    adata,
    fate_probs: pd.DataFrame,
    embedding_key: str = "X_umap",
    sample_frac: float = 0.1,
    pie_radius: float = 0.02,
    title: str = "Fate Composition",
    figsize: Tuple[float, float] = (10, 10),
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Plot pie charts showing fate composition for sampled cells.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data with embedding
    fate_probs : pd.DataFrame
        Cell x Fate probability matrix
    embedding_key : str
        Key in adata.obsm for embedding
    sample_frac : float
        Fraction of cells to show pie charts for
    pie_radius : float
        Radius of pie charts (relative to plot range)
    title : str
        Plot title
    figsize : tuple
        Figure size
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
    ax : plt.Axes, optional
        Existing axes
        
    Returns
    -------
    plt.Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    embedding = adata.obsm[embedding_key]
    n_cells = embedding.shape[0]
    n_fates = fate_probs.shape[1]
    
    # Sample cells
    n_sample = max(1, int(n_cells * sample_frac))
    sample_idx = np.random.choice(n_cells, n_sample, replace=False)
    
    # Get plot range for scaling
    x_range = embedding[:, 0].max() - embedding[:, 0].min()
    y_range = embedding[:, 1].max() - embedding[:, 1].min()
    radius = pie_radius * min(x_range, y_range)
    
    # Colors for fates
    colors = FATE_PALETTE[:n_fates]
    
    # Plot background cells
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c='lightgray',
        s=5,
        alpha=0.3,
        edgecolors='none'
    )
    
    # Plot pie charts
    for idx in sample_idx:
        probs = fate_probs.iloc[idx].values
        
        # Only show if entropy is high enough (interesting cells)
        if probs.max() < 0.9:  # Not fully committed
            x, y = embedding[idx, 0], embedding[idx, 1]
            
            # Create pie wedges
            theta1 = 0
            for j, prob in enumerate(probs):
                if prob > 0.05:  # Only show significant portions
                    theta2 = theta1 + prob * 360
                    wedge = plt.matplotlib.patches.Wedge(
                        (x, y), radius,
                        theta1, theta2,
                        facecolor=colors[j],
                        edgecolor='white',
                        linewidth=0.5,
                        alpha=0.8
                    )
                    ax.add_patch(wedge)
                    theta1 = theta2
    
    ax.set_xlabel("UMAP 1", fontsize=12)
    ax.set_ylabel("UMAP 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    
    # Legend
    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor=colors[i], label=fate)
        for i, fate in enumerate(fate_probs.columns)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return ax


def plot_driver_genes(
    drivers: pd.DataFrame,
    top_n: int = 15,
    title: str = "Metabolic Driver Genes",
    figsize: Tuple[float, float] = (8, 6),
    show: bool = True,
    save: Optional[str] = None,
    ax: Optional[plt.Axes] = None
) -> Optional[plt.Axes]:
    """
    Plot bar chart of driver genes.
    
    Parameters
    ----------
    drivers : pd.DataFrame
        Driver analysis results with 'gene' and 'log2_fc' columns
    top_n : int
        Number of top drivers to show
    title : str
        Plot title
    figsize : tuple
        Figure size
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
    ax : plt.Axes, optional
        Existing axes
        
    Returns
    -------
    plt.Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Get top drivers
    top_drivers = drivers.head(top_n).copy()
    
    if len(top_drivers) == 0:
        ax.text(0.5, 0.5, "No significant drivers found",
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Sort by log2_fc for display
    top_drivers = top_drivers.sort_values('log2_fc')
    
    # Colors based on direction
    colors = [DRIVER_PALETTE['up'] if fc > 0 else DRIVER_PALETTE['down'] 
              for fc in top_drivers['log2_fc']]
    
    # Create bar plot
    y_pos = np.arange(len(top_drivers))
    bars = ax.barh(y_pos, top_drivers['log2_fc'], color=colors, alpha=0.8)
    
    # Add gene names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_drivers['gene'])
    
    # Mark rate-limiting enzymes
    if 'is_rate_limiting' in top_drivers.columns:
        for i, (_, row) in enumerate(top_drivers.iterrows()):
            if row['is_rate_limiting']:
                ax.get_yticklabels()[i].set_fontweight('bold')
                ax.get_yticklabels()[i].set_color('#8B0000')
    
    # Add significance markers
    if 'p_adjusted' in top_drivers.columns:
        for i, (_, row) in enumerate(top_drivers.iterrows()):
            if row['p_adjusted'] < 0.001:
                marker = '***'
            elif row['p_adjusted'] < 0.01:
                marker = '**'
            elif row['p_adjusted'] < 0.05:
                marker = '*'
            else:
                marker = ''
            
            if marker:
                x_pos = row['log2_fc'] + 0.05 * np.sign(row['log2_fc'])
                ax.text(x_pos, i, marker, va='center', fontsize=8)
    
    # Formatting
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Log2 Fold Change", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor=DRIVER_PALETTE['up'], label='Upregulated'),
        plt.matplotlib.patches.Patch(facecolor=DRIVER_PALETTE['down'], label='Downregulated'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return ax


def plot_pathway_heatmap(
    cluster_profiles: pd.DataFrame,
    title: str = "Metabolic Pathway Profiles",
    figsize: Tuple[float, float] = (12, 8),
    cmap: str = "RdBu_r",
    show: bool = True,
    save: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot heatmap of pathway activity across clusters.
    
    Parameters
    ----------
    cluster_profiles : pd.DataFrame
        Cluster x Pathway score matrix
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure or None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cluster_profiles,
        cmap=cmap,
        center=0,
        annot=False,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Z-score', 'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Metabolic Pathway", fontsize=12)
    ax.set_ylabel("Cluster", fontsize=12)
    
    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return fig


def plot_wasserstein_distances(
    distances: pd.DataFrame,
    title: str = "Wasserstein Distances",
    figsize: Tuple[float, float] = (8, 8),
    cmap: str = "Blues",
    show: bool = True,
    save: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot heatmap of Wasserstein distances between groups.
    
    Parameters
    ----------
    distances : pd.DataFrame
        Pairwise distance matrix
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    show : bool
        Whether to show the plot
    save : str, optional
        Path to save figure
        
    Returns
    -------
    plt.Figure or None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        distances,
        cmap=cmap,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Wasserstein Distance', 'shrink': 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return fig


def plot_metabolic_trajectory(
    adata,
    state_space,
    metabolic_ot,
    embedding_key: str = "X_umap",
    color_by: str = "fate_entropy",
    figsize: Tuple[float, float] = (16, 5),
    show: bool = True,
    save: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create comprehensive metabolic trajectory visualization.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data
    state_space : MetabolicStateSpace
        Fitted state space
    metabolic_ot : MetabolicOT
        Fitted OT model
    embedding_key : str
        Embedding to use
    color_by : str
        Variable to color by in first panel
    figsize : tuple
        Figure size
    show : bool
        Whether to show
    save : str, optional
        Path to save
        
    Returns
    -------
    plt.Figure or None
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Metabolic landscape
    plot_metabolic_landscape(
        adata,
        color=color_by,
        embedding_key=embedding_key,
        title="Metabolic Landscape",
        show=False,
        ax=axes[0]
    )
    
    # Panel 2: Transport flow
    if metabolic_ot.transition_matrix is not None:
        plot_transport_flow(
            adata,
            metabolic_ot.transition_matrix,
            embedding_key=embedding_key,
            title="Transport Flow",
            show=False,
            ax=axes[1]
        )
    else:
        axes[1].text(0.5, 0.5, "Transition matrix not computed",
                    ha='center', va='center', transform=axes[1].transAxes)
    
    # Panel 3: Pathway scores
    if state_space.pathway_scores is not None:
        # Show distribution of a key axis
        if state_space.axis_scores is not None and len(state_space.axis_scores.columns) > 0:
            axis = state_space.axis_scores.columns[0]
            axes[2].hist(
                state_space.axis_scores[axis],
                bins=50,
                color=METABOLIC_PALETTE[2],
                alpha=0.7,
                edgecolor='white'
            )
            axes[2].set_xlabel(axis.replace('_', ' '))
            axes[2].set_ylabel("Number of Cells")
            axes[2].set_title("Metabolic Axis Distribution")
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        return None
    
    return fig
