#' @title Optimal Transport
#' @description Optimal transport-based trajectory analysis.
#' @name optimal_transport
NULL

#' Compute Optimal Transport
#'
#' @description
#' Compute optimal transport plan between source and target distributions
#' using the transport package.
#'
#' @param source_points Matrix of source point coordinates.
#' @param target_points Matrix of target point coordinates.
#' @param source_weights Numeric vector of source weights (optional).
#' @param target_weights Numeric vector of target weights (optional).
#' @param method Character. Transport method: "networkflow" or "revsimplex".
#'
#' @return A list containing:
#' \describe{
#'   \item{plan}{Transport plan matrix}
#'   \item{cost}{Total transport cost}
#'   \item{wasserstein}{Wasserstein distance}
#' }
#'
#' @examples
#' \dontrun{
#' source <- matrix(rnorm(100), ncol = 2)
#' target <- matrix(rnorm(100), ncol = 2)
#' ot_result <- optimal_transport(source, target)
#' }
#'
#' @export
optimal_transport <- function(
    source_points,
    target_points,
    source_weights = NULL,
    target_weights = NULL,
    method = "networkflow"
) {
  n_source <- nrow(source_points)
  n_target <- nrow(target_points)
  
  # Set uniform weights if not provided
  if (is.null(source_weights)) {
    source_weights <- rep(1 / n_source, n_source)
  }
  if (is.null(target_weights)) {
    target_weights <- rep(1 / n_target, n_target)
  }
  
  # Normalize weights
  source_weights <- source_weights / sum(source_weights)
  target_weights <- target_weights / sum(target_weights)
  
  # Compute cost matrix (squared Euclidean distance)
  cost_matrix <- as.matrix(stats::dist(rbind(source_points, target_points)))
  cost_matrix <- cost_matrix[1:n_source, (n_source + 1):(n_source + n_target)]^2
  
  # Normalize cost matrix for numerical stability (consistent with Python)
  cost_max <- max(cost_matrix)
  if (cost_max > 0) {
    cost_matrix <- cost_matrix / cost_max
  }
  
  # Use transport package
  if (requireNamespace("transport", quietly = TRUE)) {
    # Create weight objects
    source_wpp <- transport::wpp(source_points, source_weights)
    target_wpp <- transport::wpp(target_points, target_weights)
    
    # Compute transport
    result <- transport::transport(source_wpp, target_wpp, method = method)
    
    # Convert to plan matrix
    plan <- matrix(0, nrow = n_source, ncol = n_target)
    for (i in seq_len(nrow(result))) {
      plan[result$from[i], result$to[i]] <- result$mass[i]
    }
    
    # Compute Wasserstein distance
    W <- transport::wasserstein(source_wpp, target_wpp, p = 2)
    
    list(
      plan = plan,
      cost = sum(plan * cost_matrix),
      wasserstein = W
    )
  } else {
    # Fallback: simple nearest neighbor matching
    warning("transport package not available, using simple matching")
    
    plan <- matrix(0, nrow = n_source, ncol = n_target)
    for (i in seq_len(n_source)) {
      nearest <- which.min(cost_matrix[i, ])
      plan[i, nearest] <- source_weights[i]
    }
    
    list(
      plan = plan,
      cost = sum(plan * cost_matrix),
      wasserstein = NA
    )
  }
}

#' Compute Transition Matrix
#'
#' @description
#' Compute cell-to-cell transition probability matrix using kNN.
#'
#' @param mss Metabolic state space.
#' @param k Integer. Number of nearest neighbors.
#' @param sigma Numeric. Kernel bandwidth (optional).
#'
#' @return Transition probability matrix.
#'
#' @export
compute_transition_matrix <- function(mss, k = 30, sigma = NULL) {
  embedding <- mss$embedding
  n_cells <- nrow(embedding)
  
  # Compute distance matrix
  D <- as.matrix(stats::dist(embedding))
  
  # Determine sigma
  if (is.null(sigma)) {
    # Use median of k-th nearest neighbor distances
    k_distances <- apply(D, 1, function(x) sort(x)[k + 1])
    sigma <- stats::median(k_distances)
  }
  
  # Compute transition matrix using Gaussian kernel
  trans <- matrix(0, nrow = n_cells, ncol = n_cells)
  
  for (i in seq_len(n_cells)) {
    # Get k nearest neighbors
    nn_idx <- order(D[i, ])[2:(k + 1)]  # Exclude self
    
    for (j in nn_idx) {
      trans[i, j] <- exp(-D[i, j]^2 / (2 * sigma^2))
    }
  }
  
  # Row normalize
  row_sums <- rowSums(trans)
  trans <- trans / (row_sums + 1e-10)
  
  trans
}

#' Trajectory Inference
#'
#' @description
#' Infer metabolic trajectory using optimal transport between time points.
#'
#' @param mss Metabolic state space.
#' @param time_labels Factor or character vector of time point labels.
#' @param method Character. Transport method.
#'
#' @return A list containing transport plans between consecutive time points.
#'
#' @export
trajectory_inference <- function(mss, time_labels, method = "networkflow") {
  time_points <- sort(unique(time_labels))
  embedding <- mss$embedding
  
  transport_plans <- list()
  
  for (i in seq_len(length(time_points) - 1)) {
    t1 <- time_points[i]
    t2 <- time_points[i + 1]
    
    source_idx <- which(time_labels == t1)
    target_idx <- which(time_labels == t2)
    
    source_points <- embedding[source_idx, , drop = FALSE]
    target_points <- embedding[target_idx, , drop = FALSE]
    
    ot_result <- optimal_transport(
      source_points, target_points,
      method = method
    )
    
    transport_plans[[paste(t1, t2, sep = "_to_")]] <- list(
      plan = ot_result$plan,
      source_idx = source_idx,
      target_idx = target_idx,
      wasserstein = ot_result$wasserstein
    )
  }
  
  transport_plans
}

#' Compute Fate Probabilities
#'
#' @description
#' Compute probability of each cell reaching terminal states.
#'
#' @param trans_matrix Transition probability matrix.
#' @param terminal_states Integer vector of terminal state cell indices.
#' @param n_iter Integer. Number of iterations for power method.
#'
#' @return Matrix of cells x terminal states.
#'
#' @export
compute_fate_probabilities <- function(trans_matrix, terminal_states, n_iter = 100) {
  n_cells <- nrow(trans_matrix)
  n_terminals <- length(terminal_states)
  
  # Make terminal states absorbing
  P <- trans_matrix
  for (t in terminal_states) {
    P[t, ] <- 0
    P[t, t] <- 1
  }
  
  # Compute fate probabilities
  fate_probs <- matrix(0, nrow = n_cells, ncol = n_terminals)
  
  for (i in seq_along(terminal_states)) {
    terminal <- terminal_states[i]
    
    # Initialize with indicator
    prob <- rep(0, n_cells)
    prob[terminal] <- 1
    
    # Power iteration
    for (iter in seq_len(n_iter)) {
      prob <- t(P) %*% prob
      prob[terminal] <- 1
    }
    
    fate_probs[, i] <- as.numeric(prob)
  }
  
  # Normalize rows
  row_sums <- rowSums(fate_probs)
  fate_probs <- fate_probs / (row_sums + 1e-10)
  
  colnames(fate_probs) <- paste0("Terminal_", seq_along(terminal_states))
  
  fate_probs
}

#' Identify Bifurcation Cells
#'
#' @description
#' Identify cells at bifurcation points based on fate entropy.
#'
#' @param fate_probs Fate probability matrix.
#' @param entropy_threshold Numeric. Threshold for bifurcation (0-1).
#'
#' @return Integer vector of bifurcation cell indices.
#'
#' @export
identify_bifurcation <- function(fate_probs, entropy_threshold = 0.8) {
  # Compute entropy for each cell
  n_fates <- ncol(fate_probs)
  max_entropy <- log(n_fates)
  
  cell_entropy <- apply(fate_probs, 1, function(p) {
    p <- p[p > 0]
    -sum(p * log(p))
  })
  
  # Normalize
  normalized_entropy <- cell_entropy / max_entropy
  
  # Identify high-entropy cells
  which(normalized_entropy > entropy_threshold)
}

#' Compute Wasserstein Distance
#'
#' @description
#' Compute Wasserstein distance between two cell populations.
#'
#' @param mss Metabolic state space.
#' @param group1_idx Integer vector of first group indices.
#' @param group2_idx Integer vector of second group indices.
#'
#' @return Numeric Wasserstein distance.
#'
#' @export
wasserstein_distance <- function(mss, group1_idx, group2_idx) {
  emb1 <- mss$embedding[group1_idx, , drop = FALSE]
  emb2 <- mss$embedding[group2_idx, , drop = FALSE]
  
  ot_result <- optimal_transport(emb1, emb2)
  ot_result$wasserstein
}
