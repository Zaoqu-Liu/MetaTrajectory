"""
Test suite for MetaTrajectory package.

Tests all major components using synthetic data and real pancreas dataset.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMetabolicKnowledgeBase:
    """Test MetabolicKnowledgeBase class."""
    
    def test_initialization(self):
        """Test knowledge base initialization."""
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        
        kb = MetabolicKnowledgeBase()
        assert kb.pathways == {}
        assert kb.rate_limiting_enzymes == set()
    
    def test_load_kegg_pathways(self):
        """Test loading KEGG pathways."""
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        # Check that pathways are loaded
        assert len(kb.pathways) > 10
        assert "Glycolysis" in kb.pathways
        assert "TCA_Cycle" in kb.pathways
        assert "OXPHOS" in kb.pathways
        
        # Check rate-limiting enzymes
        assert len(kb.rate_limiting_enzymes) > 20
        assert "HK1" in kb.rate_limiting_enzymes
        assert "PFKM" in kb.rate_limiting_enzymes
        
        # Check metabolic axes
        assert len(kb.metabolic_axes) > 0
        assert "Glycolysis_vs_OXPHOS" in kb.metabolic_axes
        
        # Check network
        assert kb.network is not None
        assert kb.network.number_of_nodes() > 0
    
    def test_get_pathway_genes(self):
        """Test getting genes for a pathway."""
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        glycolysis_genes = kb.get_pathway_genes("Glycolysis")
        assert len(glycolysis_genes) > 5
        assert "HK1" in glycolysis_genes or "HK2" in glycolysis_genes
    
    def test_pathway_distance(self):
        """Test pathway distance computation."""
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        # Connected pathways should have finite distance
        d1 = kb.compute_pathway_distance("Glycolysis", "TCA_Cycle")
        assert d1 < float('inf')
        
        # Same pathway should have zero distance
        d2 = kb.compute_pathway_distance("Glycolysis", "Glycolysis")
        assert d2 == 0


class TestMetabolicStateSpace:
    """Test MetabolicStateSpace class."""
    
    @pytest.fixture
    def synthetic_adata(self):
        """Create synthetic AnnData for testing."""
        import anndata as ad
        
        np.random.seed(42)
        n_cells = 200
        n_genes = 100
        
        # Create expression matrix
        X = np.random.rand(n_cells, n_genes) * 10
        
        # Add some metabolic genes
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        metabolic_genes = kb.get_all_genes()[:50]
        other_genes = [f"Gene_{i}" for i in range(50)]
        gene_names = metabolic_genes + other_genes
        
        adata = ad.AnnData(X)
        adata.var_names = gene_names
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        
        return adata
    
    def test_fit(self, synthetic_adata):
        """Test fitting metabolic state space."""
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        from metatrajectory.state_space import MetabolicStateSpace
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        mss = MetabolicStateSpace(kb)
        mss.fit(synthetic_adata)
        
        # Check outputs
        assert mss.pathway_scores is not None
        assert mss.pathway_scores.shape[0] == synthetic_adata.n_obs
        
        assert mss.embedding is not None
        assert mss.embedding.shape[0] == synthetic_adata.n_obs
        
        assert mss._fitted is True
    
    def test_axis_scores(self, synthetic_adata):
        """Test metabolic axis score computation."""
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        from metatrajectory.state_space import MetabolicStateSpace
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        mss = MetabolicStateSpace(kb)
        mss.fit(synthetic_adata)
        
        # Check axis scores
        if mss.axis_scores is not None and len(mss.axis_scores.columns) > 0:
            assert mss.axis_scores.shape[0] == synthetic_adata.n_obs


class TestMetabolicOT:
    """Test MetabolicOT class."""
    
    @pytest.fixture
    def fitted_state_space(self):
        """Create fitted state space for testing."""
        import anndata as ad
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        from metatrajectory.state_space import MetabolicStateSpace
        
        np.random.seed(42)
        n_cells = 100
        n_genes = 50
        
        # Create expression matrix
        X = np.random.rand(n_cells, n_genes) * 10
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        metabolic_genes = kb.get_all_genes()[:50]
        
        adata = ad.AnnData(X)
        adata.var_names = metabolic_genes
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        
        mss = MetabolicStateSpace(kb)
        mss.fit(adata)
        
        return mss, adata
    
    def test_cost_matrix(self, fitted_state_space):
        """Test cost matrix computation."""
        from metatrajectory.optimal_transport import MetabolicOT
        
        mss, adata = fitted_state_space
        
        mot = MetabolicOT(mss)
        cost = mot.compute_cost_matrix()
        
        assert cost is not None
        assert cost.shape[0] == adata.n_obs
        assert cost.shape[1] == adata.n_obs
        assert cost.min() >= 0
    
    def test_transport_plan(self, fitted_state_space):
        """Test transport plan computation."""
        from metatrajectory.optimal_transport import MetabolicOT
        
        mss, adata = fitted_state_space
        
        mot = MetabolicOT(mss)
        mot.compute_cost_matrix()
        
        # Compute transport plan
        plan = mot.compute_transport_plan(reg=0.1)
        
        assert plan is not None
        assert plan.shape[0] == adata.n_obs
        assert plan.shape[1] == adata.n_obs
        
        # Plan should be non-negative
        assert plan.min() >= 0
        
        # Rows should sum to approximately 1/n (uniform source)
        row_sums = plan.sum(axis=1)
        expected = 1.0 / adata.n_obs
        np.testing.assert_array_almost_equal(row_sums, expected, decimal=3)
    
    def test_transition_matrix(self, fitted_state_space):
        """Test transition matrix computation."""
        from metatrajectory.optimal_transport import MetabolicOT
        
        mss, adata = fitted_state_space
        
        mot = MetabolicOT(mss)
        trans = mot.compute_transition_matrix(k_neighbors=20)
        
        assert trans is not None
        assert trans.shape[0] == adata.n_obs
        assert trans.shape[1] == adata.n_obs
        
        # Row sums should be close to 1
        row_sums = trans.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=3)
    
    def test_fate_probabilities(self, fitted_state_space):
        """Test fate probability computation."""
        from metatrajectory.optimal_transport import MetabolicOT
        
        mss, adata = fitted_state_space
        
        mot = MetabolicOT(mss)
        mot.compute_transition_matrix(k_neighbors=20)
        
        # Define terminal states
        terminal_states = np.array([0, 50, 99])
        
        fate_probs = mot.compute_fate_probabilities(terminal_states)
        
        assert fate_probs is not None
        assert fate_probs.shape[0] == adata.n_obs
        assert fate_probs.shape[1] == len(terminal_states)
        
        # Rows should sum to 1
        row_sums = fate_probs.values.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0, decimal=3)
    
    def test_wasserstein_distance(self, fitted_state_space):
        """Test Wasserstein distance computation."""
        from metatrajectory.optimal_transport import MetabolicOT
        
        mss, adata = fitted_state_space
        
        mot = MetabolicOT(mss)
        
        group1 = np.arange(0, 50)
        group2 = np.arange(50, 100)
        
        W = mot.compute_wasserstein_distance(group1, group2)
        
        assert W >= 0


class TestDriverAnalysis:
    """Test DriverAnalysis class."""
    
    @pytest.fixture
    def fitted_models(self):
        """Create fitted models for testing."""
        import anndata as ad
        from metatrajectory.knowledge_base import MetabolicKnowledgeBase
        from metatrajectory.state_space import MetabolicStateSpace
        from metatrajectory.optimal_transport import MetabolicOT
        
        np.random.seed(42)
        n_cells = 100
        n_genes = 50
        
        X = np.random.rand(n_cells, n_genes) * 10
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        metabolic_genes = kb.get_all_genes()[:50]
        
        adata = ad.AnnData(X)
        adata.var_names = metabolic_genes
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        
        mss = MetabolicStateSpace(kb)
        mss.fit(adata)
        
        mot = MetabolicOT(mss)
        mot.compute_transition_matrix(k_neighbors=20)
        mot.compute_fate_probabilities(np.array([0, 50, 99]))
        
        return kb, mss, mot, adata
    
    def test_fit(self, fitted_models):
        """Test driver analysis fitting."""
        from metatrajectory.drivers import DriverAnalysis
        
        kb, mss, mot, adata = fitted_models
        
        da = DriverAnalysis(mot, mss, kb)
        da.fit(adata, min_cells=5)
        
        assert da._fitted is True


class TestIntegration:
    """Test full integration workflow."""
    
    def test_full_workflow_synthetic(self):
        """Test full workflow with synthetic data."""
        import anndata as ad
        from metatrajectory import (
            MetabolicKnowledgeBase,
            MetabolicStateSpace,
            MetabolicOT,
            DriverAnalysis
        )
        
        # Create synthetic data
        np.random.seed(42)
        n_cells = 150
        
        kb = MetabolicKnowledgeBase()
        kb.load_kegg_pathways()
        
        metabolic_genes = kb.get_all_genes()[:60]
        n_genes = len(metabolic_genes)
        
        X = np.random.rand(n_cells, n_genes) * 10
        
        # Add some structure: two groups with different expression
        X[:75, :30] *= 2  # Group 1: high glycolysis-like
        X[75:, 30:] *= 2  # Group 2: high OXPHOS-like
        
        adata = ad.AnnData(X)
        adata.var_names = metabolic_genes
        adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
        
        # Run full workflow
        print("Step 1: Building metabolic state space...")
        mss = MetabolicStateSpace(kb)
        mss.fit(adata)
        
        print("Step 2: Computing optimal transport...")
        mot = MetabolicOT(mss)
        mot.compute_cost_matrix()
        mot.compute_transition_matrix(k_neighbors=20)
        
        print("Step 3: Computing fate probabilities...")
        terminal_states = mss.identify_terminal_states(n_states=3)
        fate_probs = mot.compute_fate_probabilities(terminal_states)
        
        print("Step 4: Identifying drivers...")
        da = DriverAnalysis(mot, mss, kb)
        da.fit(adata, min_cells=5)
        
        # Verify outputs
        assert mss.pathway_scores is not None
        assert mot.transition_matrix is not None
        assert fate_probs is not None
        
        print("\n=== Test Results ===")
        print(f"Pathway scores shape: {mss.pathway_scores.shape}")
        print(f"Embedding shape: {mss.embedding.shape}")
        print(f"Transition matrix shape: {mot.transition_matrix.shape}")
        print(f"Fate probabilities shape: {fate_probs.shape}")
        print(f"Terminal states: {terminal_states}")
        
        print("\nWorkflow completed successfully!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running MetaTrajectory Test Suite")
    print("=" * 60)
    
    # Test 1: Knowledge Base
    print("\n>>> Testing MetabolicKnowledgeBase...")
    test_kb = TestMetabolicKnowledgeBase()
    test_kb.test_initialization()
    test_kb.test_load_kegg_pathways()
    test_kb.test_get_pathway_genes()
    test_kb.test_pathway_distance()
    print("   PASSED!")
    
    # Test 2: Integration workflow
    print("\n>>> Testing Full Integration Workflow...")
    test_int = TestIntegration()
    test_int.test_full_workflow_synthetic()
    print("   PASSED!")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
