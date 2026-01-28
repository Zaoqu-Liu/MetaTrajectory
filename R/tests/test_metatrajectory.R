# MetaTrajectory R Package Tests
# Run with: source("tests/test_metatrajectory.R")

cat("=" , rep("=", 58), "=\n", sep = "")
cat("Running MetaTrajectory R Package Tests\n")
cat("=" , rep("=", 58), "=\n", sep = "")

# Source all R files
for (f in list.files("R", pattern = "\\.R$", full.names = TRUE)) {
  source(f)
}

# Test 1: Knowledge Base
cat("\n>>> Testing build_metabolic_graph()...\n")
kb <- build_metabolic_graph()

stopifnot(length(kb$pathways) > 10)
stopifnot("Glycolysis" %in% names(kb$pathways))
stopifnot("TCA_Cycle" %in% names(kb$pathways))
stopifnot(length(kb$rate_limiting) > 20)
stopifnot(!is.null(kb$network))
stopifnot(length(kb$metabolic_axes) > 0)

cat("   - Pathways loaded:", length(kb$pathways), "\n")
cat("   - Rate-limiting enzymes:", length(kb$rate_limiting), "\n")
cat("   - Metabolic axes:", length(kb$metabolic_axes), "\n")
cat("   PASSED!\n")

# Test 2: Get all genes
cat("\n>>> Testing get_all_metabolic_genes()...\n")
all_genes <- get_all_metabolic_genes(kb)
stopifnot(length(all_genes) > 100)
cat("   - Total unique genes:", length(all_genes), "\n")
cat("   PASSED!\n")

# Test 3: Pathway for gene
cat("\n>>> Testing get_pathway_for_gene()...\n")
pathways_for_hk1 <- get_pathway_for_gene(kb, "HK1")
stopifnot(length(pathways_for_hk1) >= 1)
stopifnot("Glycolysis" %in% pathways_for_hk1)
cat("   - HK1 pathways:", paste(pathways_for_hk1, collapse = ", "), "\n")
cat("   PASSED!\n")

# Test 4: Pathway distance
cat("\n>>> Testing compute_pathway_distance()...\n")
d1 <- compute_pathway_distance(kb, "Glycolysis", "TCA_Cycle")
d2 <- compute_pathway_distance(kb, "Glycolysis", "Glycolysis")
stopifnot(d1 < Inf)
stopifnot(d2 == 0)
cat("   - Glycolysis to TCA_Cycle:", d1, "\n")
cat("   - Glycolysis to Glycolysis:", d2, "\n")
cat("   PASSED!\n")

# Test 5: Create synthetic data and test state space
cat("\n>>> Testing metabolic state space with synthetic data...\n")

# Create synthetic expression matrix
set.seed(42)
n_cells <- 100
metabolic_genes <- get_all_metabolic_genes(kb)[1:50]

expr <- matrix(
  runif(n_cells * length(metabolic_genes)) * 10,
  nrow = length(metabolic_genes),
  ncol = n_cells
)
rownames(expr) <- metabolic_genes
colnames(expr) <- paste0("Cell_", 1:n_cells)

# Add structure
expr[1:20, 1:50] <- expr[1:20, 1:50] * 2  # Group 1: high first genes
expr[21:50, 51:100] <- expr[21:50, 51:100] * 2  # Group 2: high second genes

cat("   - Synthetic data shape:", nrow(expr), "genes x", ncol(expr), "cells\n")

# Test pathway scores
pathway_scores <- compute_pathway_scores(expr, kb, method = "mean", min_genes = 3)
cat("   - Pathway scores shape:", nrow(pathway_scores), "x", ncol(pathway_scores), "\n")
stopifnot(nrow(pathway_scores) == n_cells)
cat("   PASSED!\n")

# Test 6: Metabolic state space
cat("\n>>> Testing metabolic_state_space()...\n")
mss <- metabolic_state_space(expr, kb, n_pcs = 10)
stopifnot(!is.null(mss$pathway_scores))
stopifnot(!is.null(mss$embedding))
cat("   - Pathway scores:", nrow(mss$pathway_scores), "x", ncol(mss$pathway_scores), "\n")
cat("   - Embedding:", nrow(mss$embedding), "x", ncol(mss$embedding), "\n")
cat("   PASSED!\n")

# Test 7: Terminal states
cat("\n>>> Testing identify_terminal_states()...\n")
terminal_states <- identify_terminal_states(mss, n_states = 3)
stopifnot(length(terminal_states) >= 2)
cat("   - Terminal states identified:", terminal_states, "\n")
cat("   PASSED!\n")

# Test 8: Transition matrix
cat("\n>>> Testing compute_transition_matrix()...\n")
trans <- compute_transition_matrix(mss, k = 20)
stopifnot(nrow(trans) == n_cells)
stopifnot(ncol(trans) == n_cells)

row_sums <- rowSums(trans)
stopifnot(all(abs(row_sums - 1) < 0.01))
cat("   - Transition matrix:", nrow(trans), "x", ncol(trans), "\n")
cat("   - Row sums (should be ~1):", round(mean(row_sums), 3), "\n")
cat("   PASSED!\n")

# Test 9: Fate probabilities
cat("\n>>> Testing compute_fate_probabilities()...\n")
fate_probs <- compute_fate_probabilities(trans, terminal_states)
stopifnot(nrow(fate_probs) == n_cells)
stopifnot(ncol(fate_probs) == length(terminal_states))

fate_row_sums <- rowSums(fate_probs)
cat("   - Fate probabilities:", nrow(fate_probs), "x", ncol(fate_probs), "\n")
cat("   - Row sums (should be ~1):", round(mean(fate_row_sums), 3), "\n")
cat("   PASSED!\n")

# Test 10: Bifurcation detection
cat("\n>>> Testing identify_bifurcation()...\n")
bifurcation_cells <- identify_bifurcation(fate_probs, entropy_threshold = 0.5)
cat("   - Bifurcation cells:", length(bifurcation_cells), "\n")
cat("   PASSED!\n")

# Test 11: Optimal transport
cat("\n>>> Testing optimal_transport()...\n")
source_points <- mss$embedding[1:50, ]
target_points <- mss$embedding[51:100, ]

# Check if transport package is available
if (requireNamespace("transport", quietly = TRUE)) {
  ot_result <- optimal_transport(source_points, target_points)
  stopifnot(!is.null(ot_result$plan))
  stopifnot(nrow(ot_result$plan) == 50)
  stopifnot(ncol(ot_result$plan) == 50)
  cat("   - Transport plan:", nrow(ot_result$plan), "x", ncol(ot_result$plan), "\n")
  cat("   - Wasserstein distance:", round(ot_result$wasserstein, 4), "\n")
  cat("   PASSED!\n")
} else {
  cat("   - transport package not available, skipping OT test\n")
}

# Test 12: Driver genes
cat("\n>>> Testing find_driver_genes()...\n")
group1_cells <- colnames(expr)[1:50]
group2_cells <- colnames(expr)[51:100]
drivers <- find_driver_genes(expr, group1_cells, group2_cells, kb)
cat("   - Drivers found:", nrow(drivers), "\n")
if (nrow(drivers) > 0) {
  sig_drivers <- sum(drivers$p_adjusted < 0.05, na.rm = TRUE)
  cat("   - Significant drivers (p<0.05):", sig_drivers, "\n")
}
cat("   PASSED!\n")

cat("\n")
cat("=" , rep("=", 58), "=\n", sep = "")
cat("ALL TESTS PASSED!\n")
cat("=" , rep("=", 58), "=\n", sep = "")
