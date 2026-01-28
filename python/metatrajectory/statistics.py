"""
Statistical Methods Module

Provides rigorous statistical methods for MetaTrajectory analysis:
1. Multiple testing correction (global FDR)
2. Effect size calculations (Cohen's d, Hedges' g, Glass's delta)
3. Uncertainty quantification (bootstrap CI, Bayesian)
4. Power analysis
5. Sensitivity analysis

These methods ensure scientific rigor in driver gene identification
and trajectory inference.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    mannwhitneyu, ttest_ind, wilcoxon, spearmanr, pearsonr,
    norm, t as t_dist, bootstrap
)
from typing import Dict, List, Optional, Tuple, Union, Callable
import warnings
from dataclasses import dataclass
from enum import Enum


class EffectSizeType(Enum):
    """Effect size types."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFFS_DELTA = "cliffs_delta"  # Non-parametric
    LOG2_FC = "log2_fc"
    RANK_BISERIAL = "rank_biserial"


@dataclass
class StatisticalResult:
    """Container for statistical test results with full uncertainty."""
    
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: str
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_group1: int
    n_group2: int
    power: Optional[float] = None
    bayes_factor: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'statistic': self.statistic,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'effect_size_type': self.effect_size_type,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_level': self.ci_level,
            'n_group1': self.n_group1,
            'n_group2': self.n_group2,
            'power': self.power,
            'bayes_factor': self.bayes_factor
        }


class EffectSizeCalculator:
    """
    Calculate effect sizes with confidence intervals.
    
    Provides multiple effect size measures appropriate for different
    data distributions and experimental designs.
    """
    
    @staticmethod
    def cohens_d(
        group1: np.ndarray,
        group2: np.ndarray,
        pooled: bool = True
    ) -> Tuple[float, float, float]:
        """
        Calculate Cohen's d with confidence interval.
        
        d = (mean1 - mean2) / pooled_std
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Data arrays
        pooled : bool
            Use pooled standard deviation
            
        Returns
        -------
        d, ci_lower, ci_upper
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        if pooled:
            # Pooled standard deviation
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            pooled_std = np.sqrt(pooled_var)
        else:
            pooled_std = np.sqrt((var1 + var2) / 2)
        
        d = (mean1 - mean2) / (pooled_std + 1e-10)
        
        # Standard error of d (Hedges & Olkin formula)
        se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        
        # 95% CI
        ci_lower = d - 1.96 * se_d
        ci_upper = d + 1.96 * se_d
        
        return d, ci_lower, ci_upper
    
    @staticmethod
    def hedges_g(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate Hedges' g (bias-corrected Cohen's d).
        
        Recommended for small samples (n < 20).
        
        Returns
        -------
        g, ci_lower, ci_upper
        """
        d, ci_l, ci_u = EffectSizeCalculator.cohens_d(group1, group2)
        
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        
        # Correction factor (approaches 1 for large samples)
        correction = 1 - 3 / (4 * df - 1)
        
        g = d * correction
        ci_lower = ci_l * correction
        ci_upper = ci_u * correction
        
        return g, ci_lower, ci_upper
    
    @staticmethod
    def cliffs_delta(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate Cliff's delta (non-parametric effect size).
        
        Measures the probability that a random value from group1
        is greater than a random value from group2.
        
        Range: [-1, 1]
        - 0: no effect
        - 1: all group1 > all group2
        - -1: all group1 < all group2
        
        Returns
        -------
        delta, ci_lower, ci_upper
        """
        n1, n2 = len(group1), len(group2)
        
        # Count dominance
        greater = 0
        less = 0
        
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    greater += 1
                elif x1 < x2:
                    less += 1
        
        delta = (greater - less) / (n1 * n2)
        
        # Variance estimation (Cliff, 1996)
        # Simplified bootstrap CI
        n_boot = 1000
        boot_deltas = []
        
        for _ in range(n_boot):
            idx1 = np.random.choice(n1, n1, replace=True)
            idx2 = np.random.choice(n2, n2, replace=True)
            boot_g1 = group1[idx1]
            boot_g2 = group2[idx2]
            
            g, l = 0, 0
            for x1 in boot_g1:
                for x2 in boot_g2:
                    if x1 > x2:
                        g += 1
                    elif x1 < x2:
                        l += 1
            boot_deltas.append((g - l) / (n1 * n2))
        
        ci_lower = np.percentile(boot_deltas, 2.5)
        ci_upper = np.percentile(boot_deltas, 97.5)
        
        return delta, ci_lower, ci_upper
    
    @staticmethod
    def rank_biserial(
        group1: np.ndarray,
        group2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate rank-biserial correlation from Mann-Whitney U.
        
        This is the effect size for Mann-Whitney U test.
        
        Returns
        -------
        r, ci_lower, ci_upper
        """
        n1, n2 = len(group1), len(group2)
        
        try:
            U, _ = mannwhitneyu(group1, group2, alternative='two-sided')
        except:
            return 0.0, -1.0, 1.0
        
        # Rank-biserial correlation
        r = 1 - (2 * U) / (n1 * n2)
        
        # Bootstrap CI
        boot_rs = []
        for _ in range(1000):
            idx1 = np.random.choice(n1, n1, replace=True)
            idx2 = np.random.choice(n2, n2, replace=True)
            try:
                U_boot, _ = mannwhitneyu(group1[idx1], group2[idx2], alternative='two-sided')
                boot_rs.append(1 - (2 * U_boot) / (n1 * n2))
            except:
                pass
        
        if len(boot_rs) > 10:
            ci_lower = np.percentile(boot_rs, 2.5)
            ci_upper = np.percentile(boot_rs, 97.5)
        else:
            ci_lower, ci_upper = -1.0, 1.0
        
        return r, ci_lower, ci_upper


class MultipleTestingCorrection:
    """
    Multiple testing correction methods.
    
    Provides various FDR and FWER correction methods with proper
    implementation for large-scale testing.
    """
    
    @staticmethod
    def benjamini_hochberg(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Hochberg FDR correction.
        
        Parameters
        ----------
        p_values : np.ndarray
            Raw p-values
        alpha : float
            Target FDR level
            
        Returns
        -------
        adjusted_p : np.ndarray
            Adjusted p-values
        significant : np.ndarray
            Boolean array of significant results
        """
        n = len(p_values)
        
        # Handle NaN
        valid = ~np.isnan(p_values)
        adjusted = np.full(n, np.nan)
        
        if np.sum(valid) == 0:
            return adjusted, np.zeros(n, dtype=bool)
        
        valid_p = p_values[valid]
        n_valid = len(valid_p)
        
        # Sort
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]
        
        # Adjust
        adjusted_sorted = np.zeros(n_valid)
        for i in range(n_valid - 1, -1, -1):
            if i == n_valid - 1:
                adjusted_sorted[i] = sorted_p[i]
            else:
                adjusted_sorted[i] = min(
                    adjusted_sorted[i + 1],
                    sorted_p[i] * n_valid / (i + 1)
                )
        
        # Unsort
        adjusted_valid = np.zeros(n_valid)
        adjusted_valid[sorted_idx] = adjusted_sorted
        adjusted[valid] = adjusted_valid
        
        # Clip to [0, 1]
        adjusted = np.clip(adjusted, 0, 1)
        
        significant = adjusted < alpha
        
        return adjusted, significant
    
    @staticmethod
    def benjamini_yekutieli(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Benjamini-Yekutieli FDR correction.
        
        Controls FDR under arbitrary dependence between tests.
        More conservative than BH.
        
        Returns
        -------
        adjusted_p, significant
        """
        n = len(p_values)
        valid = ~np.isnan(p_values)
        adjusted = np.full(n, np.nan)
        
        if np.sum(valid) == 0:
            return adjusted, np.zeros(n, dtype=bool)
        
        valid_p = p_values[valid]
        n_valid = len(valid_p)
        
        # BY correction factor
        c_n = np.sum(1.0 / np.arange(1, n_valid + 1))
        
        sorted_idx = np.argsort(valid_p)
        sorted_p = valid_p[sorted_idx]
        
        adjusted_sorted = np.zeros(n_valid)
        for i in range(n_valid - 1, -1, -1):
            if i == n_valid - 1:
                adjusted_sorted[i] = sorted_p[i] * c_n
            else:
                adjusted_sorted[i] = min(
                    adjusted_sorted[i + 1],
                    sorted_p[i] * n_valid * c_n / (i + 1)
                )
        
        adjusted_valid = np.zeros(n_valid)
        adjusted_valid[sorted_idx] = adjusted_sorted
        adjusted[valid] = np.clip(adjusted_valid, 0, 1)
        
        return adjusted, adjusted < alpha
    
    @staticmethod
    def storey_q(
        p_values: np.ndarray,
        lambda_: float = 0.5
    ) -> np.ndarray:
        """
        Storey's q-value estimation.
        
        Estimates the proportion of true nulls (pi0) for more
        powerful FDR control.
        
        Parameters
        ----------
        p_values : np.ndarray
            Raw p-values
        lambda_ : float
            Threshold for pi0 estimation
            
        Returns
        -------
        q_values : np.ndarray
        """
        valid = ~np.isnan(p_values)
        n = np.sum(valid)
        
        if n == 0:
            return np.full(len(p_values), np.nan)
        
        p = p_values[valid]
        
        # Estimate pi0
        pi0 = np.sum(p > lambda_) / (n * (1 - lambda_))
        pi0 = min(pi0, 1.0)
        
        # Compute q-values
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        
        q_sorted = np.zeros(n)
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                q_sorted[i] = pi0 * sorted_p[i]
            else:
                q_sorted[i] = min(
                    q_sorted[i + 1],
                    pi0 * n * sorted_p[i] / (i + 1)
                )
        
        q = np.zeros(n)
        q[sorted_idx] = q_sorted
        
        result = np.full(len(p_values), np.nan)
        result[valid] = np.clip(q, 0, 1)
        
        return result


class UncertaintyQuantification:
    """
    Uncertainty quantification methods.
    
    Provides bootstrap, Bayesian, and analytical methods for
    quantifying uncertainty in trajectory and driver analysis.
    """
    
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: Callable,
        n_bootstrap: int = 10000,
        ci_level: float = 0.95,
        method: str = "percentile"
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval.
        
        Parameters
        ----------
        data : np.ndarray
            Data array
        statistic : callable
            Function to compute statistic
        n_bootstrap : int
            Number of bootstrap samples
        ci_level : float
            Confidence level (e.g., 0.95)
        method : str
            'percentile', 'bca', or 'basic'
            
        Returns
        -------
        point_estimate, ci_lower, ci_upper
        """
        n = len(data)
        point = statistic(data)
        
        # Bootstrap samples
        boot_stats = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            boot_sample = data[np.random.choice(n, n, replace=True)]
            boot_stats[i] = statistic(boot_sample)
        
        alpha = 1 - ci_level
        
        if method == "percentile":
            ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
            ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
            
        elif method == "basic":
            ci_lower = 2 * point - np.percentile(boot_stats, 100 * (1 - alpha / 2))
            ci_upper = 2 * point - np.percentile(boot_stats, 100 * alpha / 2)
            
        elif method == "bca":
            # Bias-corrected and accelerated
            # Bias correction
            z0 = norm.ppf(np.mean(boot_stats < point))
            
            # Acceleration (jackknife estimate)
            jackknife_stats = np.zeros(n)
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_stats[i] = statistic(jack_sample)
            
            jack_mean = np.mean(jackknife_stats)
            a = np.sum((jack_mean - jackknife_stats) ** 3) / (
                6 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5) + 1e-10
            )
            
            # Adjusted percentiles
            z_alpha_lower = norm.ppf(alpha / 2)
            z_alpha_upper = norm.ppf(1 - alpha / 2)
            
            alpha_lower = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
            alpha_upper = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
            
            ci_lower = np.percentile(boot_stats, 100 * alpha_lower)
            ci_upper = np.percentile(boot_stats, 100 * alpha_upper)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return point, ci_lower, ci_upper
    
    @staticmethod
    def bayesian_credible_interval(
        group1: np.ndarray,
        group2: np.ndarray,
        prior_mean: float = 0,
        prior_var: float = 1,
        ci_level: float = 0.95
    ) -> Tuple[float, float, float, float]:
        """
        Bayesian credible interval for difference of means.
        
        Uses conjugate Normal-Normal model with known variance.
        
        Parameters
        ----------
        group1, group2 : np.ndarray
            Data arrays
        prior_mean : float
            Prior mean for difference
        prior_var : float
            Prior variance
        ci_level : float
            Credible level
            
        Returns
        -------
        posterior_mean, ci_lower, ci_upper, bayes_factor
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Observed difference
        diff = mean1 - mean2
        diff_var = var1 / n1 + var2 / n2
        
        # Posterior (conjugate update)
        posterior_var = 1 / (1 / prior_var + 1 / diff_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + diff / diff_var)
        
        # Credible interval
        alpha = 1 - ci_level
        ci_lower = norm.ppf(alpha / 2, posterior_mean, np.sqrt(posterior_var))
        ci_upper = norm.ppf(1 - alpha / 2, posterior_mean, np.sqrt(posterior_var))
        
        # Bayes factor (Savage-Dickey ratio)
        # BF01 = posterior at 0 / prior at 0
        posterior_at_0 = norm.pdf(0, posterior_mean, np.sqrt(posterior_var))
        prior_at_0 = norm.pdf(0, prior_mean, np.sqrt(prior_var))
        bf01 = posterior_at_0 / (prior_at_0 + 1e-10)
        bf10 = 1 / (bf01 + 1e-10)  # Evidence for alternative
        
        return posterior_mean, ci_lower, ci_upper, bf10


class PowerAnalysis:
    """
    Statistical power analysis for single-cell studies.
    """
    
    @staticmethod
    def power_two_sample(
        n1: int,
        n2: int,
        effect_size: float,
        alpha: float = 0.05
    ) -> float:
        """
        Compute power for two-sample test.
        
        Parameters
        ----------
        n1, n2 : int
            Sample sizes
        effect_size : float
            Expected Cohen's d
        alpha : float
            Significance level
            
        Returns
        -------
        power : float
        """
        # Pooled sample size
        n_eff = 2 / (1/n1 + 1/n2)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n_eff / 2)
        
        # Critical value
        df = n1 + n2 - 2
        t_crit = t_dist.ppf(1 - alpha / 2, df)
        
        # Power
        power = 1 - t_dist.cdf(t_crit, df, ncp) + t_dist.cdf(-t_crit, df, ncp)
        
        return power
    
    @staticmethod
    def required_sample_size(
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
        ratio: float = 1.0
    ) -> Tuple[int, int]:
        """
        Compute required sample size for desired power.
        
        Parameters
        ----------
        effect_size : float
            Expected Cohen's d
        power : float
            Desired power
        alpha : float
            Significance level
        ratio : float
            n2/n1 ratio
            
        Returns
        -------
        n1, n2 : int
        """
        # Normal approximation for large samples
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)
        
        # Sample size formula
        n1 = int(np.ceil((1 + 1/ratio) * ((z_alpha + z_beta) / effect_size) ** 2))
        n2 = int(np.ceil(n1 * ratio))
        
        return n1, n2


def rigorous_differential_test(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = "auto",
    effect_size_type: str = "hedges_g",
    ci_level: float = 0.95
) -> StatisticalResult:
    """
    Perform rigorous differential test with effect size and uncertainty.
    
    This is the recommended function for driver gene identification.
    
    Parameters
    ----------
    group1, group2 : np.ndarray
        Expression values for two groups
    test : str
        'ttest', 'mann_whitney', or 'auto' (selects based on normality)
    effect_size_type : str
        Type of effect size to compute
    ci_level : float
        Confidence level for intervals
        
    Returns
    -------
    StatisticalResult
        Complete statistical result with uncertainty
    """
    n1, n2 = len(group1), len(group2)
    
    # Check minimum sample size
    if n1 < 3 or n2 < 3:
        return StatisticalResult(
            statistic=np.nan, p_value=1.0, effect_size=0.0,
            effect_size_type=effect_size_type, ci_lower=0.0, ci_upper=0.0,
            ci_level=ci_level, n_group1=n1, n_group2=n2
        )
    
    # Select test
    if test == "auto":
        # Shapiro-Wilk test for normality
        if n1 >= 3 and n2 >= 3:
            try:
                _, p_norm1 = stats.shapiro(group1[:min(50, n1)])
                _, p_norm2 = stats.shapiro(group2[:min(50, n2)])
                test = "ttest" if (p_norm1 > 0.05 and p_norm2 > 0.05) else "mann_whitney"
            except:
                test = "mann_whitney"
        else:
            test = "mann_whitney"
    
    # Perform test
    try:
        if test == "ttest":
            stat, pval = ttest_ind(group1, group2, equal_var=False)
        else:
            stat, pval = mannwhitneyu(group1, group2, alternative='two-sided')
    except Exception:
        stat, pval = np.nan, 1.0
    
    # Compute effect size
    esc = EffectSizeCalculator()
    
    if effect_size_type == "hedges_g":
        es, ci_l, ci_u = esc.hedges_g(group1, group2)
    elif effect_size_type == "cohens_d":
        es, ci_l, ci_u = esc.cohens_d(group1, group2)
    elif effect_size_type == "cliffs_delta":
        es, ci_l, ci_u = esc.cliffs_delta(group1, group2)
    elif effect_size_type == "rank_biserial":
        es, ci_l, ci_u = esc.rank_biserial(group1, group2)
    else:
        es = np.log2((np.mean(group1) + 1e-10) / (np.mean(group2) + 1e-10))
        ci_l, ci_u = es - 0.5, es + 0.5
    
    # Compute power
    power = PowerAnalysis.power_two_sample(n1, n2, abs(es) if es else 0.5)
    
    # Bayes factor
    try:
        _, _, _, bf = UncertaintyQuantification.bayesian_credible_interval(
            group1, group2
        )
    except:
        bf = None
    
    return StatisticalResult(
        statistic=stat,
        p_value=pval,
        effect_size=es,
        effect_size_type=effect_size_type,
        ci_lower=ci_l,
        ci_upper=ci_u,
        ci_level=ci_level,
        n_group1=n1,
        n_group2=n2,
        power=power,
        bayes_factor=bf
    )


def global_fdr_correction(
    results_df: pd.DataFrame,
    p_value_col: str = "p_value",
    method: str = "bh",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Apply global FDR correction across all tests.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Results table with p-values
    p_value_col : str
        Column name for p-values
    method : str
        'bh' (Benjamini-Hochberg), 'by' (Benjamini-Yekutieli), or 'storey'
    alpha : float
        Target FDR level
        
    Returns
    -------
    pd.DataFrame
        Results with adjusted p-values and significance flags
    """
    df = results_df.copy()
    p_values = df[p_value_col].values
    
    mtc = MultipleTestingCorrection()
    
    if method == "bh":
        adjusted, significant = mtc.benjamini_hochberg(p_values, alpha)
    elif method == "by":
        adjusted, significant = mtc.benjamini_yekutieli(p_values, alpha)
    elif method == "storey":
        adjusted = mtc.storey_q(p_values)
        significant = adjusted < alpha
    else:
        raise ValueError(f"Unknown method: {method}")
    
    df["p_adjusted"] = adjusted
    df["significant"] = significant
    df["fdr_method"] = method
    
    return df
