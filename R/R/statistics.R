#' @title Statistical Methods
#' @description Rigorous statistical methods for driver gene analysis.
#' @name statistics
NULL

#' Calculate Cohen's d Effect Size
#'
#' @description
#' Calculate Cohen's d with 95% confidence interval.
#'
#' @param group1 Numeric vector for group 1
#' @param group2 Numeric vector for group 2
#'
#' @return List with d, ci_lower, ci_upper
#'
#' @export
cohens_d <- function(group1, group2) {
  n1 <- length(group1)
  n2 <- length(group2)
  mean1 <- mean(group1)
  mean2 <- mean(group2)
  var1 <- var(group1)
  var2 <- var(group2)
  
  # Pooled standard deviation
  pooled_var <- ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
  pooled_sd <- sqrt(pooled_var)
  
  d <- (mean1 - mean2) / (pooled_sd + 1e-10)
  
  # Standard error
  se_d <- sqrt((n1 + n2) / (n1 * n2) + d^2 / (2 * (n1 + n2)))
  
  # 95% CI
  ci_lower <- d - 1.96 * se_d
  ci_upper <- d + 1.96 * se_d
  
  list(d = d, ci_lower = ci_lower, ci_upper = ci_upper, se = se_d)
}

#' Calculate Hedges' g Effect Size
#'
#' @description
#' Bias-corrected Cohen's d, recommended for small samples.
#'
#' @param group1 Numeric vector for group 1
#' @param group2 Numeric vector for group 2
#'
#' @return List with g, ci_lower, ci_upper
#'
#' @export
hedges_g <- function(group1, group2) {
  cd <- cohens_d(group1, group2)
  
  n1 <- length(group1)
  n2 <- length(group2)
  df <- n1 + n2 - 2
  
  # Correction factor
  correction <- 1 - 3 / (4 * df - 1)
  
  g <- cd$d * correction
  ci_lower <- cd$ci_lower * correction
  ci_upper <- cd$ci_upper * correction
  
  list(g = g, ci_lower = ci_lower, ci_upper = ci_upper)
}

#' Calculate Cliff's Delta (Non-parametric)
#'
#' @description
#' Non-parametric effect size measure.
#' Range: [-1, 1], 0 = no effect.
#'
#' @param group1 Numeric vector for group 1
#' @param group2 Numeric vector for group 2
#'
#' @return List with delta, ci_lower, ci_upper
#'
#' @export
cliffs_delta <- function(group1, group2) {
  n1 <- length(group1)
  n2 <- length(group2)
  
  # Count dominance
  greater <- 0
  less <- 0
  
  for (x1 in group1) {
    for (x2 in group2) {
      if (x1 > x2) greater <- greater + 1
      else if (x1 < x2) less <- less + 1
    }
  }
  
  delta <- (greater - less) / (n1 * n2)
  
  # Bootstrap CI
  n_boot <- 1000
  boot_deltas <- numeric(n_boot)
  
  for (b in seq_len(n_boot)) {
    idx1 <- sample(n1, n1, replace = TRUE)
    idx2 <- sample(n2, n2, replace = TRUE)
    boot_g1 <- group1[idx1]
    boot_g2 <- group2[idx2]
    
    g <- 0
    l <- 0
    for (x1 in boot_g1) {
      for (x2 in boot_g2) {
        if (x1 > x2) g <- g + 1
        else if (x1 < x2) l <- l + 1
      }
    }
    boot_deltas[b] <- (g - l) / (n1 * n2)
  }
  
  ci_lower <- quantile(boot_deltas, 0.025)
  ci_upper <- quantile(boot_deltas, 0.975)
  
  list(delta = delta, ci_lower = ci_lower, ci_upper = ci_upper)
}

#' Benjamini-Hochberg FDR Correction
#'
#' @description
#' Apply Benjamini-Hochberg false discovery rate correction.
#'
#' @param p_values Numeric vector of p-values
#' @param alpha Target FDR level (default 0.05)
#'
#' @return List with adjusted p-values and significance indicators
#'
#' @export
benjamini_hochberg <- function(p_values, alpha = 0.05) {
  n <- length(p_values)
  
  # Handle NA
  valid <- !is.na(p_values)
  adjusted <- rep(NA, n)
  
  if (sum(valid) == 0) {
    return(list(adjusted = adjusted, significant = rep(FALSE, n)))
  }
  
  valid_p <- p_values[valid]
  n_valid <- length(valid_p)
  
  # Sort
  sorted_idx <- order(valid_p)
  sorted_p <- valid_p[sorted_idx]
  
  # Adjust
  adjusted_sorted <- numeric(n_valid)
  adjusted_sorted[n_valid] <- sorted_p[n_valid]
  
  for (i in (n_valid - 1):1) {
    adjusted_sorted[i] <- min(
      adjusted_sorted[i + 1],
      sorted_p[i] * n_valid / i
    )
  }
  
  # Unsort
  adjusted_valid <- numeric(n_valid)
  adjusted_valid[sorted_idx] <- adjusted_sorted
  adjusted[valid] <- adjusted_valid
  
  # Clip to [0, 1]
  adjusted <- pmin(pmax(adjusted, 0), 1)
  
  significant <- adjusted < alpha
  significant[is.na(significant)] <- FALSE
  
  list(adjusted = adjusted, significant = significant)
}

#' Global FDR Correction for Results Table
#'
#' @description
#' Apply global FDR correction to a results data frame.
#'
#' @param results Data frame with p-values
#' @param p_col Name of p-value column
#' @param alpha Target FDR level
#'
#' @return Data frame with adjusted p-values
#'
#' @export
global_fdr_correction <- function(results, p_col = "p_value", alpha = 0.05) {
  p_values <- results[[p_col]]
  
  correction <- benjamini_hochberg(p_values, alpha)
  
  results$p_adjusted <- correction$adjusted
  results$significant <- correction$significant
  
  results
}

#' Rigorous Differential Test
#'
#' @description
#' Perform differential test with effect size and uncertainty.
#'
#' @param group1 Expression values for group 1
#' @param group2 Expression values for group 2
#' @param test Test type: "wilcox" or "ttest"
#' @param effect_size_type "hedges_g", "cohens_d", or "cliffs_delta"
#'
#' @return List with test results
#'
#' @export
rigorous_differential_test <- function(group1, group2, 
                                       test = "wilcox",
                                       effect_size_type = "hedges_g") {
  n1 <- length(group1)
  n2 <- length(group2)
  
  if (n1 < 3 || n2 < 3) {
    return(list(
      statistic = NA,
      p_value = 1,
      effect_size = 0,
      ci_lower = 0,
      ci_upper = 0,
      n_group1 = n1,
      n_group2 = n2
    ))
  }
  
  # Perform test
  if (test == "wilcox") {
    test_result <- tryCatch({
      stats::wilcox.test(group1, group2, alternative = "two.sided")
    }, error = function(e) list(statistic = NA, p.value = 1))
  } else {
    test_result <- tryCatch({
      stats::t.test(group1, group2, var.equal = FALSE)
    }, error = function(e) list(statistic = NA, p.value = 1))
  }
  
  # Effect size
  if (effect_size_type == "hedges_g") {
    es <- hedges_g(group1, group2)
    effect_size <- es$g
    ci_lower <- es$ci_lower
    ci_upper <- es$ci_upper
  } else if (effect_size_type == "cohens_d") {
    es <- cohens_d(group1, group2)
    effect_size <- es$d
    ci_lower <- es$ci_lower
    ci_upper <- es$ci_upper
  } else if (effect_size_type == "cliffs_delta") {
    es <- cliffs_delta(group1, group2)
    effect_size <- es$delta
    ci_lower <- es$ci_lower
    ci_upper <- es$ci_upper
  } else {
    effect_size <- log2((mean(group1) + 1e-10) / (mean(group2) + 1e-10))
    ci_lower <- effect_size - 0.5
    ci_upper <- effect_size + 0.5
  }
  
  list(
    statistic = as.numeric(test_result$statistic),
    p_value = test_result$p.value,
    effect_size = effect_size,
    effect_size_type = effect_size_type,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    n_group1 = n1,
    n_group2 = n2
  )
}

#' Power Analysis for Two-Sample Test
#'
#' @description
#' Compute statistical power for two-sample test.
#'
#' @param n1 Sample size group 1
#' @param n2 Sample size group 2
#' @param effect_size Expected Cohen's d
#' @param alpha Significance level
#'
#' @return Power (0-1)
#'
#' @export
power_two_sample <- function(n1, n2, effect_size, alpha = 0.05) {
  # Effective sample size
  n_eff <- 2 / (1/n1 + 1/n2)
  
  # Non-centrality parameter
  ncp <- effect_size * sqrt(n_eff / 2)
  
  # Degrees of freedom
  df <- n1 + n2 - 2
  
  # Critical value
  t_crit <- qt(1 - alpha / 2, df)
  
  # Power
  power <- 1 - pt(t_crit, df, ncp) + pt(-t_crit, df, ncp)
  
  power
}
