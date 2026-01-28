#' @title Visualization Functions
#' @description Publication-quality visualizations for metabolic trajectory analysis.
#' @name plotting
NULL

# Default palettes
.metabolic_palette <- c(
  "#282A62", "#692F7C", "#B43970",
  "#D96558", "#EFA143", "#F6C63C"
)

.fate_palette <- c(
  "#E64B35", "#4DBBD5", "#00A087",
  "#3C5488", "#F39B7F", "#8491B4"
)

#' Plot Metabolic Landscape
#'
#' @description
#' Plot metabolic scores on UMAP embedding.
#'
#' @param seurat_obj Seurat object with UMAP.
#' @param score Numeric vector of scores or column name in metadata.
#' @param reduction Character. Reduction to use (default "umap").
#' @param title Character. Plot title.
#' @param palette Character vector of colors.
#'
#' @return A ggplot object.
#'
#' @examples
#' \dontrun{
#' p <- plot_metabolic_landscape(seurat_obj, "axis_Glycolysis_vs_OXPHOS")
#' print(p)
#' }
#'
#' @export
plot_metabolic_landscape <- function(
    seurat_obj,
    score,
    reduction = "umap",
    title = "Metabolic Landscape",
    palette = NULL
) {
  if (is.null(palette)) {
    palette <- .metabolic_palette
  }
  
  # Get embedding
  if (inherits(seurat_obj, "Seurat")) {
    embedding <- Seurat::Embeddings(seurat_obj, reduction = reduction)
    
    # Get score
    if (is.character(score) && score %in% colnames(seurat_obj@meta.data)) {
      score_values <- seurat_obj@meta.data[[score]]
    } else if (is.numeric(score)) {
      score_values <- score
    } else {
      stop("score must be numeric vector or column name in metadata")
    }
  } else {
    stop("seurat_obj must be a Seurat object")
  }
  
  df <- data.frame(
    UMAP_1 = embedding[, 1],
    UMAP_2 = embedding[, 2],
    score = score_values
  )
  
  ggplot2::ggplot(df, ggplot2::aes(x = UMAP_1, y = UMAP_2)) +
    ggplot2::geom_point(
      ggplot2::aes(color = score),
      size = 1.5,
      alpha = 0.8
    ) +
    ggplot2::scale_color_gradientn(colors = palette) +
    ggplot2::theme_classic() +
    ggplot2::labs(
      title = title,
      color = "Score",
      x = "UMAP 1",
      y = "UMAP 2"
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5)
    )
}

#' Plot Transport Flow
#'
#' @description
#' Visualize optimal transport flow on embedding.
#'
#' @param seurat_obj Seurat object with embedding.
#' @param trans_matrix Transition matrix.
#' @param reduction Character. Reduction to use.
#' @param top_k Integer. Top transitions per cell.
#' @param min_weight Numeric. Minimum weight to show.
#' @param title Character. Plot title.
#'
#' @return A ggplot object.
#'
#' @export
plot_transport_flow <- function(
    seurat_obj,
    trans_matrix,
    reduction = "umap",
    top_k = 3,
    min_weight = 0.01,
    title = "Transport Flow"
) {
  # Get embedding
  embedding <- Seurat::Embeddings(seurat_obj, reduction = reduction)
  n_cells <- nrow(embedding)
  
  # Collect flows
  flows <- data.frame()
  
  for (i in seq_len(n_cells)) {
    trans_probs <- trans_matrix[i, ]
    top_idx <- order(-trans_probs)[1:top_k]
    
    for (j in top_idx) {
      if (i != j && trans_probs[j] > min_weight) {
        flows <- rbind(flows, data.frame(
          x_start = embedding[i, 1],
          y_start = embedding[i, 2],
          x_end = embedding[j, 1],
          y_end = embedding[j, 2],
          weight = trans_probs[j]
        ))
      }
    }
  }
  
  df_cells <- data.frame(
    UMAP_1 = embedding[, 1],
    UMAP_2 = embedding[, 2]
  )
  
  p <- ggplot2::ggplot() +
    ggplot2::geom_point(
      data = df_cells,
      ggplot2::aes(x = UMAP_1, y = UMAP_2),
      color = "lightgray",
      size = 1,
      alpha = 0.5
    ) +
    ggplot2::theme_classic() +
    ggplot2::labs(
      title = title,
      x = "UMAP 1",
      y = "UMAP 2"
    )
  
  if (nrow(flows) > 0) {
    p <- p +
      ggplot2::geom_segment(
        data = flows,
        ggplot2::aes(
          x = x_start, y = y_start,
          xend = x_end, yend = y_end,
          alpha = weight
        ),
        arrow = ggplot2::arrow(length = ggplot2::unit(0.1, "cm")),
        color = "#3C5488"
      ) +
      ggplot2::scale_alpha_continuous(range = c(0.1, 0.6), guide = "none")
  }
  
  p
}

#' Plot Fate Probabilities
#'
#' @description
#' Plot fate probability distributions.
#'
#' @param seurat_obj Seurat object.
#' @param fate_probs Fate probability matrix.
#' @param reduction Character. Reduction to use.
#' @param palette Character vector of colors.
#'
#' @return A ggplot object.
#'
#' @export
plot_fate_probabilities <- function(
    seurat_obj,
    fate_probs,
    reduction = "umap",
    palette = NULL
) {
  if (is.null(palette)) {
    palette <- .fate_palette
  }
  
  embedding <- Seurat::Embeddings(seurat_obj, reduction = reduction)
  n_fates <- ncol(fate_probs)
  
  # Create list of plots
  plots <- lapply(seq_len(n_fates), function(i) {
    fate <- colnames(fate_probs)[i]
    
    df <- data.frame(
      UMAP_1 = embedding[, 1],
      UMAP_2 = embedding[, 2],
      prob = fate_probs[, i]
    )
    
    ggplot2::ggplot(df, ggplot2::aes(x = UMAP_1, y = UMAP_2)) +
      ggplot2::geom_point(
        ggplot2::aes(color = prob),
        size = 1,
        alpha = 0.8
      ) +
      ggplot2::scale_color_gradient(
        low = "lightgray",
        high = palette[i],
        limits = c(0, 1)
      ) +
      ggplot2::theme_classic() +
      ggplot2::labs(
        title = fate,
        color = "Probability",
        x = "UMAP 1",
        y = "UMAP 2"
      ) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(hjust = 0.5)
      )
  })
  
  # Combine plots
  if (n_fates <= 3) {
    cowplot::plot_grid(plotlist = plots, nrow = 1)
  } else {
    cowplot::plot_grid(plotlist = plots, ncol = 3)
  }
}

#' Plot Driver Genes
#'
#' @description
#' Create bar plot of driver genes.
#'
#' @param drivers Data frame from find_driver_genes.
#' @param top_n Integer. Number of top drivers.
#' @param title Character. Plot title.
#'
#' @return A ggplot object.
#'
#' @export
plot_driver_genes <- function(
    drivers,
    top_n = 15,
    title = "Metabolic Driver Genes"
) {
  if (nrow(drivers) == 0) {
    return(ggplot2::ggplot() + ggplot2::theme_void() +
             ggplot2::annotate("text", x = 0.5, y = 0.5,
                               label = "No significant drivers"))
  }
  
  # Get top drivers
  top_drivers <- drivers[1:min(top_n, nrow(drivers)), ]
  top_drivers <- top_drivers[order(top_drivers$log2_fc), ]
  
  top_drivers$gene <- factor(top_drivers$gene, levels = top_drivers$gene)
  top_drivers$direction <- ifelse(top_drivers$log2_fc > 0, "up", "down")
  
  ggplot2::ggplot(
    top_drivers,
    ggplot2::aes(x = log2_fc, y = gene, fill = direction)
  ) +
    ggplot2::geom_bar(stat = "identity", alpha = 0.8) +
    ggplot2::scale_fill_manual(
      values = c("up" = "#E64B35", "down" = "#4DBBD5"),
      name = "Direction"
    ) +
    ggplot2::geom_vline(xintercept = 0, color = "black", linewidth = 0.5) +
    ggplot2::theme_classic() +
    ggplot2::labs(
      title = title,
      x = "Log2 Fold Change",
      y = ""
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5),
      axis.text.y = ggplot2::element_text(
        face = ifelse(top_drivers$is_rate_limiting, "bold", "plain")
      )
    )
}

#' Plot Pathway Heatmap
#'
#' @description
#' Create heatmap of pathway scores across clusters.
#'
#' @param pathway_scores Pathway score matrix (cells x pathways).
#' @param clusters Factor vector of cluster assignments.
#' @param title Character. Plot title.
#'
#' @return A ggplot object.
#'
#' @export
plot_pathway_heatmap <- function(
    pathway_scores,
    clusters,
    title = "Metabolic Pathway Profiles"
) {
  # Compute cluster means
  cluster_profiles <- stats::aggregate(
    pathway_scores,
    by = list(cluster = clusters),
    FUN = mean,
    na.rm = TRUE
  )
  
  rownames(cluster_profiles) <- cluster_profiles$cluster
  cluster_profiles$cluster <- NULL
  
  # Z-score normalize
  cluster_profiles <- scale(cluster_profiles)
  
  # Convert to long format
  df_long <- tidyr::pivot_longer(
    tibble::rownames_to_column(as.data.frame(cluster_profiles), "cluster"),
    cols = -cluster,
    names_to = "pathway",
    values_to = "score"
  )
  
  ggplot2::ggplot(
    df_long,
    ggplot2::aes(x = pathway, y = cluster, fill = score)
  ) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_gradient2(
      low = "#3C5488", mid = "white", high = "#E64B35",
      midpoint = 0,
      name = "Z-score"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = title,
      x = "Metabolic Pathway",
      y = "Cluster"
    ) +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1),
      plot.title = ggplot2::element_text(hjust = 0.5)
    )
}

#' Plot Wasserstein Distances
#'
#' @description
#' Plot heatmap of Wasserstein distances between groups.
#'
#' @param distances Distance matrix or data frame.
#' @param title Character. Plot title.
#'
#' @return A ggplot object.
#'
#' @export
plot_wasserstein_heatmap <- function(
    distances,
    title = "Wasserstein Distances"
) {
  # Convert to long format
  if (is.matrix(distances)) {
    distances <- as.data.frame(distances)
  }
  
  df_long <- tidyr::pivot_longer(
    tibble::rownames_to_column(distances, "group1"),
    cols = -group1,
    names_to = "group2",
    values_to = "distance"
  )
  
  ggplot2::ggplot(
    df_long,
    ggplot2::aes(x = group1, y = group2, fill = distance)
  ) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(
      ggplot2::aes(label = round(distance, 2)),
      color = "white",
      size = 3
    ) +
    ggplot2::scale_fill_gradient(
      low = "lightblue", high = "darkblue",
      name = "Distance"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::coord_fixed() +
    ggplot2::labs(
      title = title,
      x = "",
      y = ""
    ) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(hjust = 0.5)
    )
}

#' Create Summary Figure
#'
#' @description
#' Create comprehensive metabolic trajectory summary figure.
#'
#' @param seurat_obj Seurat object.
#' @param mss Metabolic state space.
#' @param fate_probs Fate probabilities (optional).
#' @param drivers Driver genes (optional).
#' @param reduction Character. Reduction to use.
#'
#' @return A combined ggplot object.
#'
#' @export
plot_metatrajectory_summary <- function(
    seurat_obj,
    mss,
    fate_probs = NULL,
    drivers = NULL,
    reduction = "umap"
) {
  plots <- list()
  
  # Panel 1: Metabolic landscape (first axis)
  if (!is.null(mss$axis_scores) && ncol(mss$axis_scores) > 0) {
    axis_name <- colnames(mss$axis_scores)[1]
    seurat_obj@meta.data$metabolic_axis <- mss$axis_scores[, 1]
    plots[[1]] <- plot_metabolic_landscape(
      seurat_obj, "metabolic_axis",
      reduction = reduction,
      title = "Metabolic Landscape"
    )
  }
  
  # Panel 2: Fate probabilities
  if (!is.null(fate_probs)) {
    seurat_obj@meta.data$fate_entropy <- apply(fate_probs, 1, function(p) {
      p <- p[p > 0]
      -sum(p * log(p + 1e-10))
    })
    plots[[2]] <- plot_metabolic_landscape(
      seurat_obj, "fate_entropy",
      reduction = reduction,
      title = "Fate Uncertainty"
    )
  }
  
  # Panel 3: Driver genes
  if (!is.null(drivers) && nrow(drivers) > 0) {
    plots[[3]] <- plot_driver_genes(drivers, top_n = 10)
  }
  
  if (length(plots) > 0) {
    cowplot::plot_grid(plotlist = plots, ncol = min(3, length(plots)))
  } else {
    message("No plots generated")
    NULL
  }
}
