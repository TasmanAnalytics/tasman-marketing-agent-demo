"""Analysis utilities for Phase 2 deep analytics agents.

Provides:
- Stratified sampling with configurable strata
- Categorical encoding (one-hot, target, ordinal)
- Statistical functions (CI, bootstrap, multiple testing correction)
- Outlier handling (winsorize, clip)
- Feature preprocessing
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.utils import resample
import warnings


class StratifiedSampler:
    """Stratified sampling to cap row counts while preserving distribution."""

    def __init__(
        self,
        max_rows: int,
        strata_cols: List[str],
        min_stratum_size: int = 100,
        random_state: int = 42
    ):
        """
        Initialize stratified sampler.

        Args:
            max_rows: Maximum rows to sample
            strata_cols: Columns to stratify by (in priority order)
            min_stratum_size: Minimum rows per stratum
            random_state: Random seed for reproducibility
        """
        self.max_rows = max_rows
        self.strata_cols = strata_cols
        self.min_stratum_size = min_stratum_size
        self.random_state = random_state
        self.sample_ratio_ = None
        self.strata_distribution_ = None

    def fit_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stratified sampling if needed.

        Args:
            df: Input DataFrame

        Returns:
            Sampled DataFrame with metadata
        """
        n = len(df)

        # No sampling needed
        if n <= self.max_rows:
            self.sample_ratio_ = 1.0
            self.strata_distribution_ = self._get_strata_dist(df)
            return df

        # Find valid strata columns
        valid_strata = [col for col in self.strata_cols if col in df.columns]

        if not valid_strata:
            # Fall back to random sampling
            sampled = df.sample(n=self.max_rows, random_state=self.random_state)
            self.sample_ratio_ = self.max_rows / n
            self.strata_distribution_ = None
            return sampled

        # Create strata key
        df = df.copy()
        df['_strata_key'] = df[valid_strata].astype(str).agg('_'.join, axis=1)

        # Count strata
        strata_counts = df['_strata_key'].value_counts()
        total_strata = len(strata_counts)

        # Calculate proportional samples per stratum
        target_per_stratum = self.max_rows / total_strata

        # Sample from each stratum
        sampled_dfs = []
        for stratum, count in strata_counts.items():
            stratum_df = df[df['_strata_key'] == stratum]

            # Sample size for this stratum (at least min_stratum_size if possible)
            sample_size = max(
                min(int(target_per_stratum), count),
                min(self.min_stratum_size, count)
            )

            if sample_size < count:
                stratum_sample = stratum_df.sample(
                    n=sample_size,
                    random_state=self.random_state
                )
            else:
                stratum_sample = stratum_df

            sampled_dfs.append(stratum_sample)

        # Combine samples
        sampled = pd.concat(sampled_dfs, ignore_index=True)

        # If still over max_rows, random downsample
        if len(sampled) > self.max_rows:
            sampled = sampled.sample(n=self.max_rows, random_state=self.random_state)

        # Drop helper column
        sampled = sampled.drop(columns=['_strata_key'])

        self.sample_ratio_ = len(sampled) / n
        self.strata_distribution_ = self._get_strata_dist(sampled, valid_strata)

        return sampled

    def _get_strata_dist(
        self,
        df: pd.DataFrame,
        strata_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get distribution of strata."""
        if strata_cols is None:
            strata_cols = [col for col in self.strata_cols if col in df.columns]

        if not strata_cols:
            return {}

        dist = {}
        for col in strata_cols:
            dist[col] = df[col].value_counts().to_dict()

        return dist

    def get_metadata(self) -> Dict[str, Any]:
        """Get sampling metadata."""
        return {
            'sample_ratio': self.sample_ratio_,
            'strata_distribution': self.strata_distribution_,
            'max_rows': self.max_rows,
            'strata_cols': self.strata_cols
        }


class CategoricalEncoder:
    """Encode categorical features with cardinality limits."""

    def __init__(
        self,
        method: str = 'onehot',
        max_categories: int = 20,
        min_frequency: float = 0.01
    ):
        """
        Initialize categorical encoder.

        Args:
            method: Encoding method (onehot | target | ordinal)
            max_categories: Max categories (top K + 'other')
            min_frequency: Minimum category frequency (0-1)
        """
        self.method = method
        self.max_categories = max_categories
        self.min_frequency = min_frequency
        self.encoders_ = {}
        self.category_maps_ = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Encode categorical columns.

        Args:
            df: Input DataFrame
            cat_cols: Categorical columns to encode
            target: Target variable (required for target encoding)

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        for col in cat_cols:
            if col not in df.columns:
                continue

            # Get top categories
            value_counts = df[col].value_counts()
            freq = value_counts / len(df)

            # Filter by frequency and limit
            valid_cats = freq[freq >= self.min_frequency].head(self.max_categories).index.tolist()

            # Map to 'other' for rare/unseen
            df[f'{col}_original'] = df[col]
            df[col] = df[col].apply(lambda x: x if x in valid_cats else 'other')

            self.category_maps_[col] = valid_cats

            if self.method == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col, f'{col}_original'])
                self.encoders_[col] = list(dummies.columns)

            elif self.method == 'target' and target is not None:
                # Target encoding (mean of target per category)
                target_means = df.groupby(col)[target.name].mean()
                df[f'{col}_encoded'] = df[col].map(target_means)
                df = df.drop(columns=[col, f'{col}_original'])
                self.encoders_[col] = target_means.to_dict()

            elif self.method == 'ordinal':
                # Ordinal encoding
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                df = df.drop(columns=[col, f'{col}_original'])
                self.encoders_[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        return df


def winsorize(
    series: pd.Series,
    limits: Tuple[float, float] = (0.01, 0.99)
) -> pd.Series:
    """
    Winsorize (clip) outliers at specified percentiles.

    Args:
        series: Input series
        limits: Lower and upper percentile limits (0-1)

    Returns:
        Winsorized series
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower=lower, upper=upper).copy()


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: Input data
        statistic: Function to compute statistic (default: mean)
        n_iterations: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        random_state: Random seed

    Returns:
        (point_estimate, lower_ci, upper_ci)
    """
    np.random.seed(random_state)

    estimates = []
    n = len(data)

    for _ in range(n_iterations):
        sample = resample(data, n_samples=n, random_state=None)
        estimates.append(statistic(sample))

    point_estimate = statistic(data)
    alpha = 1 - confidence_level
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return point_estimate, lower, upper


def benjamini_hochberg(
    pvalues: List[float],
    alpha: float = 0.1
) -> Tuple[List[bool], List[float]]:
    """
    Benjamini-Hochberg FDR correction for multiple testing.

    Args:
        pvalues: List of p-values
        alpha: FDR rate (default: 0.1)

    Returns:
        (rejected, adjusted_pvalues) where rejected[i] indicates if H0 rejected
    """
    m = len(pvalues)

    # Sort p-values
    sorted_indices = np.argsort(pvalues)
    sorted_pvalues = np.array(pvalues)[sorted_indices]

    # Compute BH threshold
    ranks = np.arange(1, m + 1)
    bh_threshold = (ranks / m) * alpha

    # Find largest k where p[k] <= (k/m)*alpha
    comparisons = sorted_pvalues <= bh_threshold

    if np.any(comparisons):
        max_k = np.where(comparisons)[0].max()
        # Reject H0 for indices 0..max_k
        rejected_sorted = np.zeros(m, dtype=bool)
        rejected_sorted[:max_k + 1] = True
    else:
        rejected_sorted = np.zeros(m, dtype=bool)

    # Unsort
    rejected = np.zeros(m, dtype=bool)
    rejected[sorted_indices] = rejected_sorted

    # Adjusted p-values (q-values)
    adjusted = np.minimum(1, sorted_pvalues * m / ranks)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]  # Enforce monotonicity

    adjusted_pvalues = np.zeros(m)
    adjusted_pvalues[sorted_indices] = adjusted

    return rejected.tolist(), adjusted_pvalues.tolist()


def proportion_test(
    successes1: int,
    trials1: int,
    successes2: int,
    trials2: int,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Two-sample z-test for proportions.

    Args:
        successes1: Successes in group 1
        trials1: Trials in group 1
        successes2: Successes in group 2
        trials2: Trials in group 2
        alternative: 'two-sided' | 'greater' | 'less'

    Returns:
        Dict with p_value, z_stat, p1, p2, diff, relative_lift
    """
    # Handle edge cases
    if trials1 == 0 or trials2 == 0:
        p1 = successes1 / trials1 if trials1 > 0 else 0
        p2 = successes2 / trials2 if trials2 > 0 else 0
        return {
            'p_value': 1.0,
            'z_stat': 0.0,
            'p1': p1,
            'p2': p2,
            'diff': p1 - p2,
            'relative_lift': 0.0
        }

    p1 = successes1 / trials1
    p2 = successes2 / trials2

    # Pooled proportion
    p_pool = (successes1 + successes2) / (trials1 + trials2)

    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1/trials1 + 1/trials2))

    if se == 0:
        return {
            'p_value': 1.0,
            'z_stat': 0.0,
            'p1': p1,
            'p2': p2,
            'diff': 0.0,
            'relative_lift': 0.0
        }

    # Z-statistic
    z_stat = (p1 - p2) / se

    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)
    else:  # less
        p_value = stats.norm.cdf(z_stat)

    diff = p1 - p2
    relative_lift = (p1 / p2 - 1) if p2 > 0 else 0.0

    return {
        'p_value': p_value,
        'z_stat': z_stat,
        'p1': p1,
        'p2': p2,
        'diff': diff,
        'relative_lift': relative_lift
    }


def welch_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Welch's t-test for unequal variances.

    Args:
        group1: Data from group 1
        group2: Data from group 2
        alternative: 'two-sided' | 'greater' | 'less'

    Returns:
        Dict with p_value, t_stat, mean1, mean2, diff, cohens_d
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    n1 = len(group1)
    n2 = len(group2)

    # Welch's t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(var1/n1 + var2/n2)

    # Degrees of freedom (Welch-Satterthwaite)
    df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df)
    else:  # less
        p_value = stats.t.cdf(t_stat, df)

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    diff = mean1 - mean2

    return {
        'p_value': p_value,
        't_stat': t_stat,
        'df': df,
        'mean1': mean1,
        'mean2': mean2,
        'diff': diff,
        'cohens_d': cohens_d
    }


def levene_test(groups: List[np.ndarray], alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Levene's test for equality of variances.

    Args:
        groups: List of data arrays
        alpha: Significance level

    Returns:
        (equal_variances, p_value)
    """
    stat, p_value = stats.levene(*groups)
    equal_variances = p_value > alpha
    return equal_variances, p_value


def compute_confidence_interval(
    data: np.ndarray,
    confidence_level: float = 0.95,
    method: str = 'normal'
) -> Tuple[float, float]:
    """
    Compute confidence interval for mean.

    Args:
        data: Input data
        confidence_level: Confidence level (default: 0.95)
        method: 'normal' or 'bootstrap'

    Returns:
        (lower_ci, upper_ci)
    """
    if method == 'bootstrap':
        _, lower, upper = bootstrap_ci(data, np.mean, confidence_level=confidence_level)
        return lower, upper

    # Normal approximation
    mean = np.mean(data)
    se = stats.sem(data)
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha/2)

    lower = mean - z * se
    upper = mean + z * se

    return lower, upper


def standardize_features(
    df: pd.DataFrame,
    cols: List[str],
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric features (z-score normalization).

    Args:
        df: Input DataFrame
        cols: Columns to standardize
        scaler: Pre-fitted scaler (optional)

    Returns:
        (standardized_df, fitted_scaler)
    """
    df = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])

    return df, scaler


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using IQR method.

    Args:
        series: Input series
        multiplier: IQR multiplier (default: 1.5)

    Returns:
        Boolean series indicating outliers
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = (series < lower_bound) | (series > upper_bound)
    return outliers.copy()


def format_pvalue(p: float, precision: int = 3) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001"
    elif p > 0.999:
        return "> 0.999"
    else:
        return f"{p:.{precision}f}"


def format_effect_size(effect: float, ci: Tuple[float, float], precision: int = 3) -> str:
    """Format effect size with confidence interval."""
    return f"{effect:.{precision}f} [{ci[0]:.{precision}f}, {ci[1]:.{precision}f}]"
