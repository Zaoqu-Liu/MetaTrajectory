#' @title Metabolic State Space
#' @description Construct metabolic state space from single-cell data.
#' @name state_space
NULL

#' Compute Pathway Scores
#'
#' @description
#' Calculate metabolic pathway activity scores for each cell.
#'
#' @param seurat_obj Seurat object or expression matrix.
#' @param kb Metabolic knowledge base from build_metabolic_graph().
#' @param assay Character. Seurat assay to use (default "RNA").
#' @param slot Character. Slot to use (default "data").
#' @param method Character. Scoring method: "mean" or "zscore".
#' @param min_genes Integer. Minimum genes required per pathway.
#' @param scale Logical. Whether to z-score normalize scores.
#'
#' @return A matrix of cells x pathways.
#'
#' @examples
#' \dontrun{
#' kb <- build_metabolic_graph()
#' scores <- compute_pathway_scores(seurat_obj, kb)
#' }
#'
#' @export
compute_pathway_scores <- function(
    seurat_obj,
    kb,
    assay = "RNA",
    slot = "data",
    method = c("mean", "zscore"),
    min_genes = 3,
    scale = TRUE
) {
  method <- match.arg(method)
  
  # Extract expression matrix
  if (inherits(seurat_obj, "Seurat")) {
    expr <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = slot)
  } else if (is.matrix(seurat_obj) || inherits(seurat_obj, "dgCMatrix")) {
    expr <- seurat_obj
  } else {
    stop("seurat_obj must be a Seurat object or expression matrix")
  }
  
  genes_available <- rownames(expr)
  cells <- colnames(expr)
  
  # Score each pathway
  scores <- lapply(names(kb$pathways), function(pw) {
    genes <- kb$pathways[[pw]]
    genes_use <- intersect(genes, genes_available)
    
    if (length(genes_use) < min_genes) {
      warning(sprintf("Pathway '%s' has only %d genes, skipping", pw, length(genes_use)))
      return(rep(NA_real_, length(cells)))
    }
    
    sub_expr <- as.matrix(expr[genes_use, , drop = FALSE])
    
    if (method == "mean") {
      colMeans(sub_expr, na.rm = TRUE)
    } else {
      z <- t(scale(t(sub_expr)))
      colMeans(z, na.rm = TRUE)
    }
  })
  
  score_mat <- do.call(rbind, scores)
  rownames(score_mat) <- names(kb$pathways)
  colnames(score_mat) <- cells
  
  # Transpose to cells x pathways
  score_mat <- t(score_mat)
  
  # Remove empty pathways
  valid <- colSums(!is.na(score_mat)) > 0
  score_mat <- score_mat[, valid, drop = FALSE]
  
  # Scale if requested
  if (scale && ncol(score_mat) > 0) {
    score_mat <- scale(score_mat)
  }
  
  score_mat
}

#' Compute Metabolic Axis Scores
#'
#' @description
#' Calculate metabolic axis scores representing biological trade-offs.
#'
#' @param seurat_obj Seurat object or expression matrix.
#' @param kb Metabolic knowledge base.
#' @param assay Character. Seurat assay to use.
#' @param slot Character. Slot to use.
#'
#' @return A matrix of cells x axes.
#'
#' @export
compute_axis_scores <- function(
    seurat_obj,
    kb,
    assay = "RNA",
    slot = "data"
) {
  # Extract expression
  if (inherits(seurat_obj, "Seurat")) {
    expr <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = slot)
  } else {
    expr <- seurat_obj
  }
  
  genes_available <- rownames(expr)
  cells <- colnames(expr)
  
  axis_scores <- sapply(names(kb$metabolic_axes), function(axis_name) {
    axis_def <- kb$metabolic_axes[[axis_name]]
    
    pos_genes <- intersect(axis_def$positive, genes_available)
    neg_genes <- intersect(axis_def$negative, genes_available)
    
    if (length(pos_genes) < 2 || length(neg_genes) < 2) {
      warning(sprintf("Axis '%s' has insufficient genes", axis_name))
      return(rep(NA_real_, length(cells)))
    }
    
    pos_expr <- colMeans(as.matrix(expr[pos_genes, , drop = FALSE]), na.rm = TRUE)
    neg_expr <- colMeans(as.matrix(expr[neg_genes, , drop = FALSE]), na.rm = TRUE)
    
    pos_expr - neg_expr
  })
  
  axis_scores <- as.matrix(axis_scores)
  rownames(axis_scores) <- cells
  
  # Z-score normalize
  axis_scores <- scale(axis_scores)
  
  axis_scores
}

#' Build Metabolic State Space
#'
#' @description
#' Construct complete metabolic state space including scores and embedding.
#'
#' @param seurat_obj Seurat object.
#' @param kb Metabolic knowledge base.
#' @param n_pcs Integer. Number of PCA components.
#' @param assay Character. Seurat assay to use.
#' @param slot Character. Slot to use.
#'
#' @return A list containing:
#' \describe{
#'   \item{pathway_scores}{Cell x Pathway score matrix}
#'   \item{axis_scores}{Cell x Axis score matrix}
#'   \item{embedding}{PCA embedding}
#'   \item{pca}{PCA object}
#' }
#'
#' @export
metabolic_state_space <- function(
    seurat_obj,
    kb,
    n_pcs = 20,
    assay = "RNA",
    slot = "data"
) {
  # Compute pathway scores
  pathway_scores <- compute_pathway_scores(
    seurat_obj, kb,
    assay = assay, slot = slot,
    method = "mean", min_genes = 3, scale = TRUE
  )
  
  # Compute axis scores
  axis_scores <- compute_axis_scores(
    seurat_obj, kb,
    assay = assay, slot = slot
  )
  
  # PCA embedding
  n_pcs <- min(n_pcs, ncol(pathway_scores), nrow(pathway_scores) - 1)
  
  pca <- stats::prcomp(pathway_scores, center = FALSE, scale. = FALSE)
  embedding <- pca$x[, seq_len(n_pcs), drop = FALSE]
  
  list(
    pathway_scores = pathway_scores,
    axis_scores = axis_scores,
    embedding = embedding,
    pca = pca,
    n_cells = nrow(pathway_scores),
    n_pathways = ncol(pathway_scores)
  )
}

#' Identify Terminal States
#'
#' @description
#' Identify potential terminal metabolic states based on axis extremes.
#'
#' @param mss Metabolic state space from metabolic_state_space().
#' @param n_states Integer. Number of terminal states to identify.
#'
#' @return Integer vector of cell indices.
#'
#' @export
identify_terminal_states <- function(mss, n_states = 3) {
  if (is.null(mss$axis_scores) || ncol(mss$axis_scores) == 0) {
    # Fall back to embedding extremes
    terminal_idx <- c()
    for (i in seq_len(min(ncol(mss$embedding), n_states))) {
      terminal_idx <- c(terminal_idx, which.max(mss$embedding[, i]))
      terminal_idx <- c(terminal_idx, which.min(mss$embedding[, i]))
    }
    return(unique(terminal_idx)[seq_len(min(n_states, length(unique(terminal_idx))))])
  }
  
  terminal_idx <- c()
  
  for (axis in colnames(mss$axis_scores)) {
    scores <- mss$axis_scores[, axis]
    if (!all(is.na(scores))) {
      terminal_idx <- c(terminal_idx, which.max(scores))
      terminal_idx <- c(terminal_idx, which.min(scores))
    }
  }
  
  unique(terminal_idx)[seq_len(min(n_states, length(unique(terminal_idx))))]
}

#' Get Metabolic Distance Matrix
#'
#' @description
#' Compute pairwise metabolic distances between cells.
#'
#' @param mss Metabolic state space.
#' @param method Character. Distance method.
#'
#' @return Distance matrix.
#'
#' @export
compute_metabolic_distance <- function(mss, method = "euclidean") {
  as.matrix(stats::dist(mss$embedding, method = method))
}
