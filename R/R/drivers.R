#' @title Driver Gene Analysis
#' @description Identify metabolic driver genes for fate decisions.
#' @name drivers
NULL

#' Find Driver Genes
#'
#' @description
#' Identify genes driving metabolic state transitions between two cell groups.
#'
#' @param seurat_obj Seurat object.
#' @param group1_cells Character vector of cell names in group 1.
#' @param group2_cells Character vector of cell names in group 2.
#' @param kb Metabolic knowledge base.
#' @param assay Character. Seurat assay to use.
#' @param slot Character. Slot to use.
#' @param test Character. Statistical test: "wilcox" or "t".
#'
#' @return Data frame with driver analysis results.
#'
#' @examples
#' \dontrun{
#' kb <- build_metabolic_graph()
#' drivers <- find_driver_genes(seurat_obj, cells1, cells2, kb)
#' }
#'
#' @export
find_driver_genes <- function(
    seurat_obj,
    group1_cells,
    group2_cells,
    kb,
    assay = "RNA",
    slot = "data",
    test = c("wilcox", "t")
) {
  test <- match.arg(test)
  
  # Get expression
  if (inherits(seurat_obj, "Seurat")) {
    expr <- Seurat::GetAssayData(seurat_obj, assay = assay, slot = slot)
  } else {
    expr <- seurat_obj
  }
  
  # Get metabolic genes
  metabolic_genes <- get_all_metabolic_genes(kb)
  available_genes <- intersect(metabolic_genes, rownames(expr))
  
  if (length(available_genes) == 0) {
    warning("No metabolic genes found in expression data")
    return(data.frame())
  }
  
  # Get expression for each group
  expr1 <- as.matrix(expr[available_genes, group1_cells, drop = FALSE])
  expr2 <- as.matrix(expr[available_genes, group2_cells, drop = FALSE])
  
  # Test each gene
  results <- lapply(available_genes, function(gene) {
    x1 <- expr1[gene, ]
    x2 <- expr2[gene, ]
    
    # Skip if no variance
  # Skip if both groups have no variance (use tolerance for float comparison)
  if (stats::sd(x1) < 1e-10 && stats::sd(x2) < 1e-10) {
    return(NULL)
  }
    
    # Statistical test
    if (test == "wilcox") {
      test_result <- tryCatch({
        stats::wilcox.test(x1, x2)
      }, error = function(e) list(p.value = NA))
    } else {
      test_result <- tryCatch({
        stats::t.test(x1, x2)
      }, error = function(e) list(p.value = NA))
    }
    
    # Effect size (log fold change)
    mean1 <- mean(x1) + 1e-10
    mean2 <- mean(x2) + 1e-10
    # log2 fold change: group1 / group2 (consistent with Python)
    log_fc <- log2(mean1 / mean2)
    
    # Get pathways for this gene
    pathways <- get_pathway_for_gene(kb, gene)
    
    data.frame(
      gene = gene,
      log2_fc = log_fc,
      p_value = test_result$p.value,
      mean_group1 = mean1,
      mean_group2 = mean2,
      is_rate_limiting = gene %in% kb$rate_limiting,
      pathways = paste(pathways[1:min(3, length(pathways))], collapse = ", "),
      stringsAsFactors = FALSE
    )
  })
  
  results <- do.call(rbind, results[!sapply(results, is.null)])
  
  if (nrow(results) == 0) {
    return(data.frame())
  }
  
  # Multiple testing correction
  results$p_adjusted <- stats::p.adjust(results$p_value, method = "BH")
  
  # Driver score
  results$driver_score <- -log10(results$p_adjusted + 1e-300) * abs(results$log2_fc)
  
  # Sort by driver score
  results <- results[order(-results$driver_score), ]
  rownames(results) <- NULL
  
  results
}

#' Find Fate Drivers
#'
#' @description
#' Identify genes driving fate decisions based on fate probabilities.
#'
#' @param seurat_obj Seurat object.
#' @param fate_probs Fate probability matrix.
#' @param kb Metabolic knowledge base.
#' @param fate_threshold Numeric. Threshold for fate commitment.
#' @param min_cells Integer. Minimum cells per fate.
#' @param assay Character. Seurat assay to use.
#'
#' @return Data frame with fate driver analysis.
#'
#' @export
find_fate_drivers <- function(
    seurat_obj,
    fate_probs,
    kb,
    fate_threshold = 0.7,
    min_cells = 10,
    assay = "RNA"
) {
  fates <- colnames(fate_probs)
  all_results <- list()
  
  # Compare each pair of fates
  for (i in seq_along(fates)) {
    for (j in seq_along(fates)) {
      if (i >= j) next
      
      fate1 <- fates[i]
      fate2 <- fates[j]
      
      # Get cells committed to each fate
      cells1 <- rownames(fate_probs)[fate_probs[, fate1] > fate_threshold]
      cells2 <- rownames(fate_probs)[fate_probs[, fate2] > fate_threshold]
      
      if (length(cells1) < min_cells || length(cells2) < min_cells) {
        next
      }
      
      # Find drivers
      drivers <- find_driver_genes(
        seurat_obj, cells1, cells2, kb,
        assay = assay
      )
      
      if (nrow(drivers) > 0) {
        drivers$fate1 <- fate1
        drivers$fate2 <- fate2
        all_results[[paste(fate1, fate2, sep = "_vs_")]] <- drivers
      }
    }
  }
  
  if (length(all_results) == 0) {
    return(data.frame())
  }
  
  do.call(rbind, all_results)
}

#' Get Top Drivers
#'
#' @description
#' Get top driver genes with filtering options.
#'
#' @param drivers Data frame from find_driver_genes or find_fate_drivers.
#' @param top_n Integer. Number of top drivers to return.
#' @param only_rate_limiting Logical. Only return rate-limiting enzymes.
#' @param max_pval Numeric. Maximum adjusted p-value.
#'
#' @return Filtered data frame.
#'
#' @export
get_top_drivers <- function(
    drivers,
    top_n = 20,
    only_rate_limiting = FALSE,
    max_pval = 0.05
) {
  if (nrow(drivers) == 0) {
    return(drivers)
  }
  
  # Filter by p-value
  drivers <- drivers[drivers$p_adjusted < max_pval, ]
  
  # Filter by rate-limiting
  if (only_rate_limiting) {
    drivers <- drivers[drivers$is_rate_limiting, ]
  }
  
  # Return top N
  if (nrow(drivers) > top_n) {
    drivers <- drivers[1:top_n, ]
  }
  
  drivers
}

#' Compute Pathway Enrichment
#'
#' @description
#' Compute pathway enrichment for driver genes.
#'
#' @param drivers Data frame of driver genes.
#' @param kb Metabolic knowledge base.
#' @param pval_cutoff Numeric. P-value cutoff for significant drivers.
#'
#' @return Data frame with enrichment results.
#'
#' @export
compute_pathway_enrichment <- function(drivers, kb, pval_cutoff = 0.05) {
  # Get significant drivers
  sig_drivers <- unique(drivers$gene[drivers$p_adjusted < pval_cutoff])
  
  if (length(sig_drivers) == 0) {
    return(data.frame())
  }
  
  total_genes <- length(get_all_metabolic_genes(kb))
  
  results <- lapply(names(kb$pathways), function(pw) {
    pw_genes <- kb$pathways[[pw]]
    
    overlap <- length(intersect(sig_drivers, pw_genes))
    
    if (overlap == 0) {
      return(NULL)
    }
    
    # Expected by chance
    expected <- length(sig_drivers) * length(pw_genes) / total_genes
    
    # Enrichment ratio
    enrichment <- overlap / (expected + 1e-10)
    
    # Hypergeometric test
    pval <- stats::phyper(
      overlap - 1,
      length(pw_genes),
      total_genes - length(pw_genes),
      length(sig_drivers),
      lower.tail = FALSE
    )
    
    data.frame(
      pathway = pw,
      n_drivers = overlap,
      n_pathway_genes = length(pw_genes),
      enrichment = enrichment,
      p_value = pval,
      driver_genes = paste(intersect(sig_drivers, pw_genes)[1:min(5, overlap)], collapse = ", "),
      stringsAsFactors = FALSE
    )
  })
  
  results <- do.call(rbind, results[!sapply(results, is.null)])
  
  if (nrow(results) > 0) {
    results$p_adjusted <- stats::p.adjust(results$p_value, method = "BH")
    results <- results[order(-results$enrichment), ]
  }
  
  results
}

#' Integrate with RNA Velocity
#'
#' @description
#' Integrate metabolic analysis with RNA velocity results.
#'
#' @param seurat_obj Seurat object with velocity results.
#' @param kb Metabolic knowledge base.
#' @param velocity_slot Character. Name of velocity slot.
#'
#' @return Data frame with pathway velocity contributions.
#'
#' @export
integrate_velocity <- function(seurat_obj, kb, velocity_slot = "velocity") {
  # Check for velocity
  if (!velocity_slot %in% names(seurat_obj@assays)) {
    warning("Velocity assay not found")
    return(data.frame())
  }
  
  velocity <- Seurat::GetAssayData(seurat_obj, assay = velocity_slot, slot = "data")
  
  # Compute pathway-level velocity
  pathway_velocities <- lapply(names(kb$pathways), function(pw) {
    genes <- kb$pathways[[pw]]
    available <- intersect(genes, rownames(velocity))
    
    if (length(available) < 3) {
      return(NULL)
    }
    
    pw_vel <- as.matrix(velocity[available, , drop = FALSE])
    
    data.frame(
      pathway = pw,
      mean_velocity = mean(colMeans(pw_vel, na.rm = TRUE), na.rm = TRUE),
      n_genes = length(available),
      direction = ifelse(mean(colMeans(pw_vel, na.rm = TRUE), na.rm = TRUE) > 0, "up", "down"),
      stringsAsFactors = FALSE
    )
  })
  
  do.call(rbind, pathway_velocities[!sapply(pathway_velocities, is.null)])
}
