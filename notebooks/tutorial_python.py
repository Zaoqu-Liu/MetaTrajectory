"""
MetaTrajectory Python Tutorial
==============================

This tutorial demonstrates how to use MetaTrajectory for optimal transport-based
metabolic trajectory analysis in single-cell RNA-seq data.

Key concepts:
1. We use optimal transport instead of pseudo-dynamics (mathematically rigorous)
2. We leverage metabolic network topology for biologically meaningful distances
3. We integrate with (not replace) RNA velocity results
"""

# %% [markdown]
# # MetaTrajectory Tutorial
# 
# ## Overview
# 
# MetaTrajectory is a mathematically rigorous framework for analyzing metabolic 
# state transitions in single-cell RNA-seq data using optimal transport theory.
# 
# **Key advantages over pseudo-time methods:**
# 1. No assumption of continuous trajectories
# 2. Probabilistic interpretation of fate transitions
# 3. Handles branching naturally
# 4. Wasserstein distance as a meaningful metric

# %%
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add path if running from notebook directory
import sys
sys.path.insert(0, '../python')

# Import MetaTrajectory
from metatrajectory import (
    MetabolicKnowledgeBase,
    MetabolicStateSpace,
    MetabolicOT,
    DriverAnalysis,
    plot_metabolic_landscape,
    plot_driver_genes
)

print("MetaTrajectory imported successfully!")

# %% [markdown]
# ## Step 1: Build Metabolic Knowledge Base
# 
# The knowledge base contains:
# - Curated metabolic pathway gene sets (from KEGG)
# - Rate-limiting enzyme definitions
# - Metabolic network structure
# - Metabolic axis definitions (e.g., glycolysis vs OXPHOS)

# %%
# Initialize and load knowledge base
kb = MetabolicKnowledgeBase()
kb.load_kegg_pathways()

# Print summary
print(kb.summary())
print("\nAvailable pathways:", list(kb.pathways.keys()))
print("\nMetabolic axes:", list(kb.metabolic_axes.keys()))

# %% [markdown]
# ## Step 2: Create Synthetic Data
# 
# For this tutorial, we create synthetic data with clear metabolic structure.
# In practice, you would use your own AnnData object from scanpy.

# %%
import anndata as ad

# Create synthetic data with metabolic structure
np.random.seed(42)
n_cells = 300

# Get metabolic genes from knowledge base
metabolic_genes = kb.get_all_genes()[:100]
n_genes = len(metabolic_genes)

# Base expression
X = np.random.rand(n_cells, n_genes) * 5

# Create three metabolic states:
# State 1: High glycolysis (cells 0-99)
glycolysis_genes = kb.get_pathway_genes("Glycolysis")
glycolysis_idx = [i for i, g in enumerate(metabolic_genes) if g in glycolysis_genes]
X[:100, glycolysis_idx] *= 3

# State 2: High OXPHOS (cells 100-199)
oxphos_genes = kb.get_pathway_genes("OXPHOS")
oxphos_idx = [i for i, g in enumerate(metabolic_genes) if g in oxphos_genes]
X[100:200, oxphos_idx] *= 3

# State 3: High TCA (cells 200-299)
tca_genes = kb.get_pathway_genes("TCA_Cycle")
tca_idx = [i for i, g in enumerate(metabolic_genes) if g in tca_genes]
X[200:300, tca_idx] *= 3

# Create AnnData
adata = ad.AnnData(X)
adata.var_names = metabolic_genes
adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
adata.obs['group'] = ['Glycolytic']*100 + ['Oxidative']*100 + ['TCA_high']*100

print(f"Created synthetic data: {adata.shape}")
print(f"Groups: {adata.obs['group'].value_counts().to_dict()}")

# %% [markdown]
# ## Step 3: Build Metabolic State Space
# 
# This step:
# 1. Computes pathway-level activity scores
# 2. Constructs metabolic axes (biological trade-offs)
# 3. Creates low-dimensional embedding

# %%
# Build metabolic state space
mss = MetabolicStateSpace(kb)
mss.fit(adata, scoring_method="mean", n_components=10)

print(mss.summary())

# Examine pathway scores
print("\nPathway scores (first 5 cells):")
print(mss.pathway_scores.head())

# Examine axis scores
print("\nAxis scores (first 5 cells):")
print(mss.axis_scores.head())

# %% [markdown]
# ## Step 4: Optimal Transport Analysis
# 
# This is the core innovation of MetaTrajectory:
# - We compute transport plans between cell populations
# - This gives us transition probabilities without assuming pseudo-time
# - Wasserstein distance quantifies metabolic differences

# %%
# Initialize OT model
mot = MetabolicOT(mss)

# Compute cost matrix (metabolic distances)
cost = mot.compute_cost_matrix(metric="euclidean")
print(f"Cost matrix shape: {cost.shape}")

# Compute transition matrix using kNN
trans = mot.compute_transition_matrix(k_neighbors=30)
print(f"Transition matrix shape: {trans.shape}")

# Identify terminal states
terminal_states = mss.identify_terminal_states(n_states=3, method="extremes")
print(f"Terminal states: {terminal_states}")

# Compute fate probabilities
fate_probs = mot.compute_fate_probabilities(terminal_states, n_iterations=50)
print(f"Fate probabilities shape: {fate_probs.shape}")
print("\nFate probability distribution:")
print(fate_probs.describe())

# %% [markdown]
# ## Step 5: Compute Wasserstein Distances
# 
# Wasserstein distance quantifies how different two metabolic states are.
# Unlike Euclidean distance, it considers the "cost" of transformation.

# %%
# Compute pairwise Wasserstein distances between groups
group_labels = adata.obs['group'].values
W_dist = mot.compute_pairwise_wasserstein(group_labels)

print("Pairwise Wasserstein Distances:")
print(W_dist.round(3))

# %% [markdown]
# ## Step 6: Identify Driver Genes
# 
# Driver analysis identifies metabolic enzymes that distinguish different fates.
# Rate-limiting enzymes are prioritized as they control metabolic flux.

# %%
# Initialize driver analysis
da = DriverAnalysis(mot, mss, kb)
da.fit(adata, min_cells=30)

print(da.summary())

# Get top transition-correlated drivers
if da.transition_drivers is not None and len(da.transition_drivers) > 0:
    print("\nTop transition-correlated genes:")
    print(da.transition_drivers.head(10)[['gene', 'correlation', 'p_adjusted', 'is_rate_limiting']])

# %% [markdown]
# ## Step 7: Direct Comparison Between Groups
# 
# We can also directly compare metabolic drivers between two cell populations.

# %%
# Compare glycolytic vs oxidative cells
glycolytic_idx = np.where(adata.obs['group'] == 'Glycolytic')[0]
oxidative_idx = np.where(adata.obs['group'] == 'Oxidative')[0]

drivers = da.find_driver_genes(glycolytic_idx, oxidative_idx, top_n=15)

print("Driver genes: Glycolytic vs Oxidative")
print(drivers[['gene', 'log2_fc', 'p_adjusted', 'is_rate_limiting', 'pathways']])

# %% [markdown]
# ## Summary
# 
# In this tutorial, we demonstrated:
# 
# 1. **Building the knowledge base** with metabolic pathways and network structure
# 2. **Computing pathway scores** and metabolic axes
# 3. **Optimal transport analysis** for trajectory inference
# 4. **Wasserstein distances** to quantify metabolic differences
# 5. **Driver gene identification** to find key metabolic regulators
# 
# ### Key differences from pseudo-time methods:
# 
# | Feature | Pseudo-time | MetaTrajectory (OT) |
# |---------|-------------|---------------------|
# | Direction | Assumed | Not required |
# | Topology | Single path | Any |
# | Probabilities | No | Yes (fate probs) |
# | Distance metric | Graph | Wasserstein |
# | Mathematical foundation | Heuristic | Optimal transport |
# 
# ### Best practices:
# 
# 1. Use MetaTrajectory for **metabolic interpretation**, not as a replacement for RNA velocity
# 2. Focus on **rate-limiting enzymes** as they control metabolic flux
# 3. Validate computational findings with **biological knowledge**
# 4. Use **Wasserstein distance** to quantify metabolic differences between conditions

print("\nTutorial completed successfully!")
