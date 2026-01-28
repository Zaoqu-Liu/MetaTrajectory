#' @title Advanced Pathway Scoring Methods
#' @description Multiple pathway scoring methods for single-cell metabolic analysis.
#' @name scoring
NULL

#' Compute Pathway Scores with Multiple Methods
#'
#' @description
#' Compute pathway activity scores using various methods:
#' - mean: Simple mean of gene expression
#' - vision: Z-score normalized signature scoring
#' - weighted: Rate-limiting enzyme weighted average
#' - aucell: Rank-based AUCell-like scoring
#'
#' @param seurat_obj Seurat object
#' @param kb Knowledge base from build_metabolic_graph()
#' @param method Scoring method: "mean", "vision", "weighted", "aucell"
#' @param assay Expression assay to use
#' @param slot Data slot: "data" or "counts"
#' @param min_genes Minimum genes per pathway
#'
#' @return Matrix of pathway scores (cells x pathways)
#'
#' @export
compute_pathway_scores_advanced <- function(seurat_obj, 
                                            kb,
                                            method = "vision",
                                            assay = "RNA",
                                            slot = "data",
                                            min_genes = 3) {
  
  # Get expression matrix
  if (slot == "data") {
    expr <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = "data")
  } else {
    expr <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = "counts")
  }
  
  expr <- as.matrix(expr)
  genes <- rownames(expr)
  cells <- colnames(expr)
  
  pathways <- kb$pathways
  rate_limiting <- kb$rate_limiting
  
  # Compute scores based on method
  if (method == "mean") {
    scores <- compute_mean_scores(expr, genes, pathways, min_genes)
  } else if (method == "vision") {
    scores <- compute_vision_scores(expr, genes, pathways, min_genes)
  } else if (method == "weighted") {
    scores <- compute_weighted_scores(expr, genes, pathways, rate_limiting, min_genes)
  } else if (method == "aucell") {
    scores <- compute_aucell_scores(expr, genes, pathways, min_genes)
  } else {
    stop(paste("Unknown method:", method))
  }
  
  rownames(scores) <- cells
  return(scores)
}

#' Compute Mean Scores
#' @keywords internal
compute_mean_scores <- function(expr, genes, pathways, min_genes) {
  n_cells <- ncol(expr)
  scores <- matrix(NA, nrow = n_cells, ncol = length(pathways))
  colnames(scores) <- names(pathways)
  
  for (i in seq_along(pathways)) {
    pathway_name <- names(pathways)[i]
    pathway_genes <- pathways[[i]]
    
    available <- pathway_genes[pathway_genes %in% genes]
    
    if (length(available) >= min_genes) {
      pathway_expr <- expr[available, , drop = FALSE]
      scores[, i] <- colMeans(pathway_expr)
    }
  }
  
  # Remove NA columns
  scores <- scores[, !apply(is.na(scores), 2, all), drop = FALSE]
  return(scores)
}

#' Compute VISION-like Scores
#' @keywords internal  
compute_vision_scores <- function(expr, genes, pathways, min_genes) {
  n_cells <- ncol(expr)
  
  # Normalize library size
  lib_sizes <- colSums(expr)
  lib_sizes[lib_sizes == 0] <- 1
  expr_norm <- t(t(expr) / lib_sizes) * median(lib_sizes)
  
  # Log transform
  expr_log <- log1p(expr_norm)
  
  # Z-score genes
  expr_z <- t(scale(t(expr_log)))
  expr_z[is.na(expr_z)] <- 0
  
  scores <- matrix(NA, nrow = n_cells, ncol = length(pathways))
  colnames(scores) <- names(pathways)
  
  for (i in seq_along(pathways)) {
    pathway_name <- names(pathways)[i]
    pathway_genes <- pathways[[i]]
    
    available <- pathway_genes[pathway_genes %in% genes]
    
    if (length(available) >= min_genes) {
      pathway_expr <- expr_z[available, , drop = FALSE]
      # Trimmed mean (remove outliers)
      scores[, i] <- apply(pathway_expr, 2, function(x) {
        x <- sort(x)
        n <- length(x)
        trim <- max(1, floor(n * 0.1))
        if (n > 2 * trim) {
          mean(x[(trim + 1):(n - trim)])
        } else {
          mean(x)
        }
      })
    }
  }
  
  scores <- scores[, !apply(is.na(scores), 2, all), drop = FALSE]
  
  # Z-score normalize final scores
  scores <- scale(scores)
  scores[is.na(scores)] <- 0
  
  return(scores)
}

#' Compute Weighted Scores
#' @keywords internal
compute_weighted_scores <- function(expr, genes, pathways, rate_limiting, min_genes,
                                    rate_limiting_weight = 3.0) {
  n_cells <- ncol(expr)
  
  # Normalize
  lib_sizes <- colSums(expr) + 1
  expr_norm <- log1p(t(t(expr) / lib_sizes) * 10000)
  
  scores <- matrix(NA, nrow = n_cells, ncol = length(pathways))
  colnames(scores) <- names(pathways)
  
  for (i in seq_along(pathways)) {
    pathway_name <- names(pathways)[i]
    pathway_genes <- pathways[[i]]
    
    available <- pathway_genes[pathway_genes %in% genes]
    
    if (length(available) >= min_genes) {
      # Compute weights
      weights <- ifelse(available %in% rate_limiting, rate_limiting_weight, 1.0)
      weights <- weights / sum(weights)
      
      pathway_expr <- expr_norm[available, , drop = FALSE]
      scores[, i] <- colSums(pathway_expr * weights)
    }
  }
  
  scores <- scores[, !apply(is.na(scores), 2, all), drop = FALSE]
  scores <- scale(scores)
  scores[is.na(scores)] <- 0
  
  return(scores)
}

#' Compute AUCell-like Scores
#' @keywords internal
compute_aucell_scores <- function(expr, genes, pathways, min_genes, 
                                  n_top = 500) {
  n_cells <- ncol(expr)
  n_genes <- nrow(expr)
  
  n_top <- min(n_top, n_genes)
  
  # Rank genes for each cell (higher expression = lower rank)
  rankings <- apply(expr, 2, function(x) rank(-x, ties.method = "first"))
  
  scores <- matrix(NA, nrow = n_cells, ncol = length(pathways))
  colnames(scores) <- names(pathways)
  
  for (i in seq_along(pathways)) {
    pathway_name <- names(pathways)[i]
    pathway_genes <- pathways[[i]]
    
    available <- pathway_genes[pathway_genes %in% genes]
    
    if (length(available) >= min_genes) {
      gene_idx <- match(available, genes)
      n_pathway <- length(gene_idx)
      
      for (cell in seq_len(n_cells)) {
        pathway_ranks <- rankings[gene_idx, cell]
        
        # Compute AUC
        sorted_ranks <- sort(pathway_ranks)
        auc <- sum(pmax(0, n_top - sorted_ranks + 1) / n_top)
        
        scores[cell, i] <- auc / n_pathway
      }
    }
  }
  
  scores <- scores[, !apply(is.na(scores), 2, all), drop = FALSE]
  return(scores)
}
