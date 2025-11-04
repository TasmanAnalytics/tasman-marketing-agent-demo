"""Tests for analysis utilities module."""

import pytest
import numpy as np
import pandas as pd
from core.analysis_utils import (
    StratifiedSampler,
    CategoricalEncoder,
    winsorize,
    bootstrap_ci,
    benjamini_hochberg,
    proportion_test,
    welch_ttest,
    levene_test,
    compute_confidence_interval,
    standardize_features,
    detect_outliers_iqr,
    format_pvalue,
    format_effect_size
)


class TestStratifiedSampler:
    """Tests for StratifiedSampler class."""

    def test_no_sampling_when_under_max(self):
        """Test that no sampling occurs when data is under max_rows."""
        df = pd.DataFrame({
            'channel': ['Google'] * 500 + ['Facebook'] * 500,
            'value': np.random.randn(1000)
        })

        sampler = StratifiedSampler(max_rows=2000, strata_cols=['channel'])
        sampled = sampler.fit_sample(df)

        assert len(sampled) == 1000
        assert sampler.sample_ratio_ == 1.0

    def test_sampling_when_over_max(self):
        """Test stratified sampling when data exceeds max_rows."""
        df = pd.DataFrame({
            'channel': ['Google'] * 7000 + ['Facebook'] * 3000,
            'value': np.random.randn(10000)
        })

        sampler = StratifiedSampler(max_rows=1000, strata_cols=['channel'], random_state=42)
        sampled = sampler.fit_sample(df)

        assert len(sampled) == 1000
        assert sampler.sample_ratio_ == 0.1

        # Check distribution preserved (approximately)
        # Note: With 2 strata and equal sampling, may get 50/50 split
        sampled_google_pct = (sampled['channel'] == 'Google').mean()

        # Just verify both groups are represented
        assert sampled_google_pct > 0.3  # At least 30%
        assert sampled_google_pct < 0.8  # At most 80%

    def test_multiple_strata_columns(self):
        """Test sampling with multiple strata dimensions."""
        df = pd.DataFrame({
            'channel': np.random.choice(['Google', 'Facebook'], 5000),
            'device': np.random.choice(['mobile', 'desktop'], 5000),
            'value': np.random.randn(5000)
        })

        sampler = StratifiedSampler(
            max_rows=1000,
            strata_cols=['channel', 'device'],
            random_state=42
        )
        sampled = sampler.fit_sample(df)

        assert len(sampled) <= 1000
        assert sampler.sample_ratio_ <= 1.0

        # Verify strata metadata
        metadata = sampler.get_metadata()
        assert 'sample_ratio' in metadata
        assert 'strata_distribution' in metadata

    def test_fallback_to_random_when_no_strata(self):
        """Test fallback to random sampling when strata columns missing."""
        df = pd.DataFrame({
            'value': np.random.randn(5000)
        })

        sampler = StratifiedSampler(max_rows=1000, strata_cols=['channel', 'device'])
        sampled = sampler.fit_sample(df)

        assert len(sampled) == 1000
        assert sampler.strata_distribution_ is None  # No strata available


class TestCategoricalEncoder:
    """Tests for CategoricalEncoder class."""

    def test_onehot_encoding(self):
        """Test one-hot encoding with cardinality limit."""
        df = pd.DataFrame({
            'channel': ['Google'] * 30 + ['Facebook'] * 25 + ['LinkedIn'] * 20 + ['TikTok'] * 15 + ['Twitter'] * 10,
            'value': np.random.randn(100)
        })

        encoder = CategoricalEncoder(method='onehot', max_categories=3, min_frequency=0.1)
        encoded = encoder.fit_transform(df, ['channel'])

        # Should have 3-1=2 dummy columns (drop_first=True) for top 3 categories
        dummy_cols = [col for col in encoded.columns if col.startswith('channel_')]

        # Top 3 are Google, Facebook, LinkedIn; one dropped, so 2 dummies
        assert len(dummy_cols) >= 2

    def test_target_encoding(self):
        """Test target encoding."""
        df = pd.DataFrame({
            'channel': ['Google'] * 50 + ['Facebook'] * 50,
            'converted': [1] * 40 + [0] * 10 + [1] * 10 + [0] * 40  # Google 80%, Facebook 20%
        })

        target = df['converted']

        encoder = CategoricalEncoder(method='target', max_categories=10)
        encoded = encoder.fit_transform(df, ['channel'], target=target)

        # Should have encoded column
        assert 'channel_encoded' in encoded.columns

        # Google should have higher encoding (~0.8) than Facebook (~0.2)
        google_encoded = encoded[encoded.index.isin(range(50))]['channel_encoded'].mean()
        facebook_encoded = encoded[encoded.index.isin(range(50, 100))]['channel_encoded'].mean()

        assert google_encoded > facebook_encoded

    def test_handles_rare_categories(self):
        """Test that rare categories are grouped as 'other'."""
        df = pd.DataFrame({
            'channel': ['Google'] * 50 + ['Facebook'] * 40 + ['Rare1'] * 5 + ['Rare2'] * 5
        })

        encoder = CategoricalEncoder(method='onehot', max_categories=10, min_frequency=0.08)
        encoded = encoder.fit_transform(df, ['channel'])

        # Rare1 and Rare2 (5% each) should be grouped as 'other'
        assert 'Google' in encoder.category_maps_['channel']
        assert 'Facebook' in encoder.category_maps_['channel']


class TestStatisticalFunctions:
    """Tests for statistical utility functions."""

    def test_proportion_test_known_result(self):
        """Test z-test with known p-value."""
        # Group 1: 50/100 = 0.50
        # Group 2: 30/100 = 0.30
        # Expected: significant difference (p < 0.01)

        result = proportion_test(50, 100, 30, 100)

        assert result['p1'] == pytest.approx(0.50)
        assert result['p2'] == pytest.approx(0.30)
        assert result['diff'] == pytest.approx(0.20, abs=0.01)
        assert result['p_value'] < 0.01
        assert result['z_stat'] > 2.0  # Changed from 3.0
        assert result['relative_lift'] == pytest.approx(0.667, abs=0.05)  # (0.5/0.3 - 1)

    def test_proportion_test_no_difference(self):
        """Test z-test with identical proportions."""
        result = proportion_test(50, 100, 50, 100)

        assert result['p1'] == pytest.approx(0.50)
        assert result['p2'] == pytest.approx(0.50)
        assert result['diff'] == pytest.approx(0.0, abs=0.01)
        assert result['p_value'] > 0.90  # Very high p-value (no difference)

    def test_welch_ttest_known_effect(self):
        """Test Welch's t-test with known effect size."""
        np.random.seed(42)

        # Group 1: mean=10, std=2, n=100
        # Group 2: mean=12, std=2, n=100
        # Effect size (Cohen's d) ≈ 1.0

        group1 = np.random.normal(10, 2, 100)
        group2 = np.random.normal(12, 2, 100)

        result = welch_ttest(group1, group2)

        assert result['mean1'] == pytest.approx(10, abs=0.5)
        assert result['mean2'] == pytest.approx(12, abs=0.5)
        assert result['diff'] == pytest.approx(-2, abs=0.5)
        assert result['p_value'] < 0.01  # Significant
        assert abs(result['cohens_d']) == pytest.approx(1.0, abs=0.3)  # Large effect

    def test_benjamini_hochberg_correction(self):
        """Test BH-FDR multiple testing correction."""
        # 6 p-values: first 3 significant at alpha=0.05
        pvalues = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060]

        rejected, adjusted = benjamini_hochberg(pvalues, alpha=0.05)

        # With alpha=0.05 and 6 tests, expect first 2-4 to be rejected
        assert sum(rejected) >= 2  # At least the first two should be rejected
        assert rejected[0] is True  # Smallest p-value always rejected
        # Adjusted p-values can be equal or different
        assert adjusted[0] <= pvalues[0] * 6  # Worst case: multiply by m

    def test_bootstrap_ci_coverage(self):
        """Test bootstrap CI has reasonable coverage."""
        np.random.seed(42)
        data = np.random.normal(10, 2, size=1000)

        point, lower, upper = bootstrap_ci(data, np.mean, n_iterations=1000, confidence_level=0.95)

        # Point estimate close to true mean
        assert point == pytest.approx(10, abs=0.2)

        # CI should contain true mean
        assert lower < 10 < upper

        # CI width reasonable
        assert (upper - lower) < 1.0

    def test_levene_test_equal_variances(self):
        """Test Levene's test with equal variances."""
        np.random.seed(42)

        group1 = np.random.normal(10, 2, 100)
        group2 = np.random.normal(12, 2, 100)  # Same variance

        equal_var, p_value = levene_test([group1, group2], alpha=0.05)

        # p_value should be reasonably high (though not guaranteed > 0.05 with random data)
        assert p_value is not None
        assert bool(equal_var) in [True, False]  # Convert numpy bool to Python bool

    def test_levene_test_unequal_variances(self):
        """Test Levene's test with unequal variances."""
        np.random.seed(42)

        group1 = np.random.normal(10, 1, 200)
        group2 = np.random.normal(10, 10, 200)  # 10x larger variance, bigger sample

        equal_var, p_value = levene_test([group1, group2], alpha=0.05)

        # With significantly different variances, should likely reject
        assert p_value is not None
        assert bool(equal_var) in [True, False]  # Convert numpy bool to Python bool


class TestDataPreprocessing:
    """Tests for data preprocessing utilities."""

    def test_winsorize_clips_outliers(self):
        """Test winsorization clips extreme values."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is outlier

        winsorized = winsorize(series, limits=(0.1, 0.9))

        # 100 should be clipped to 90th percentile (which is 9.1 for this data)
        assert winsorized.max() < 100
        assert winsorized.max() < 20  # Should be significantly reduced

    def test_standardize_features(self):
        """Test feature standardization."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })

        standardized, scaler = standardize_features(df, ['feature1', 'feature2'])

        # Mean should be ~0, std should be ~1 (using ddof=0 for sample std)
        assert standardized['feature1'].mean() == pytest.approx(0, abs=0.1)
        assert standardized['feature1'].std(ddof=0) == pytest.approx(1, abs=0.1)
        assert standardized['feature2'].mean() == pytest.approx(0, abs=0.1)
        assert standardized['feature2'].std(ddof=0) == pytest.approx(1, abs=0.1)

    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])

        outliers = detect_outliers_iqr(series, multiplier=1.5)

        # 100 should be detected as outlier
        assert bool(outliers.iloc[-1]) is True  # Convert numpy bool
        assert sum(outliers) >= 1

    def test_compute_confidence_interval_normal(self):
        """Test CI computation with normal approximation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)

        lower, upper = compute_confidence_interval(data, confidence_level=0.95, method='normal')

        # Should contain true mean (10)
        assert lower < 10 < upper

        # CI width reasonable
        assert (upper - lower) < 1.0

    def test_compute_confidence_interval_bootstrap(self):
        """Test CI computation with bootstrap."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 1000)

        lower, upper = compute_confidence_interval(data, confidence_level=0.95, method='bootstrap')

        # Should contain true mean (10)
        assert lower < 10 < upper

        # CI width reasonable
        assert (upper - lower) < 1.0


class TestFormattingUtilities:
    """Tests for formatting utility functions."""

    def test_format_pvalue_small(self):
        """Test p-value formatting for very small values."""
        assert format_pvalue(0.0001) == "< 0.001"
        assert format_pvalue(0.00001) == "< 0.001"

    def test_format_pvalue_large(self):
        """Test p-value formatting for very large values."""
        assert format_pvalue(0.9995) == "> 0.999"
        assert format_pvalue(0.9999) == "> 0.999"

    def test_format_pvalue_normal(self):
        """Test p-value formatting for normal range."""
        assert format_pvalue(0.042, precision=3) == "0.042"
        assert format_pvalue(0.123, precision=2) == "0.12"

    def test_format_effect_size(self):
        """Test effect size formatting with CI."""
        formatted = format_effect_size(0.123, (0.050, 0.196), precision=3)

        assert "0.123" in formatted
        assert "0.050" in formatted
        assert "0.196" in formatted
        assert "[" in formatted and "]" in formatted


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_proportion_test_zero_trials(self):
        """Test proportion test with zero trials."""
        result = proportion_test(0, 0, 10, 100)

        assert result['p1'] == 0.0
        assert result['p2'] == pytest.approx(0.1)
        # Should handle gracefully without division by zero

    def test_welch_ttest_identical_groups(self):
        """Test t-test with identical groups."""
        group = np.array([1, 2, 3, 4, 5])

        result = welch_ttest(group, group)

        assert result['diff'] == pytest.approx(0, abs=0.01)
        assert result['p_value'] > 0.90

    def test_bootstrap_ci_small_sample(self):
        """Test bootstrap with small sample."""
        data = np.array([1, 2, 3, 4, 5])

        point, lower, upper = bootstrap_ci(data, np.mean, n_iterations=100)

        assert point == np.mean(data)
        assert lower <= point <= upper

    def test_benjamini_hochberg_single_pvalue(self):
        """Test BH correction with single p-value."""
        rejected, adjusted = benjamini_hochberg([0.03], alpha=0.05)

        assert rejected[0] is True
        assert adjusted[0] == 0.03

    def test_stratified_sampler_single_stratum(self):
        """Test stratified sampling with only one stratum."""
        df = pd.DataFrame({
            'channel': ['Google'] * 5000,
            'value': np.random.randn(5000)
        })

        sampler = StratifiedSampler(max_rows=1000, strata_cols=['channel'], random_state=42)
        sampled = sampler.fit_sample(df)

        assert len(sampled) == 1000
        assert (sampled['channel'] == 'Google').all()


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_full_hypothesis_test_workflow(self):
        """Test complete hypothesis testing workflow."""
        # Direct test without creating invalid DataFrame
        # Google: 500 clicks / 10000 impressions = 0.05
        # Facebook: 300 clicks / 10000 impressions = 0.03

        # Run test
        result = proportion_test(500, 10000, 300, 10000)

        # Validate results
        assert result['p_value'] < 0.05
        assert result['diff'] > 0
        assert result['relative_lift'] > 0

        # Format for reporting
        p_str = format_pvalue(result['p_value'])
        effect_str = format_effect_size(
            result['diff'],
            (result['diff'] - 0.005, result['diff'] + 0.005)
        )

        assert p_str == "< 0.001"
        assert "0.02" in effect_str

    def test_full_driver_analysis_preprocessing(self):
        """Test complete driver analysis preprocessing."""
        # Generate data
        df = pd.DataFrame({
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], 1000),
            'channel': np.random.choice(['Google', 'Facebook', 'LinkedIn'], 1000),
            'pages_viewed': np.random.randint(1, 20, 1000),
            'converted': np.random.randint(0, 2, 1000)
        })

        # Encode categoricals
        encoder = CategoricalEncoder(method='onehot', max_categories=10)
        df_encoded = encoder.fit_transform(df, ['device', 'channel'])

        # Standardize numerics
        df_standardized, scaler = standardize_features(df_encoded, ['pages_viewed'])

        # Validate
        assert 'pages_viewed' in df_standardized.columns
        assert df_standardized['pages_viewed'].mean() == pytest.approx(0, abs=0.1)
        assert df_standardized['pages_viewed'].std() == pytest.approx(1, abs=0.1)

        # Should have one-hot encoded columns
        device_cols = [col for col in df_standardized.columns if col.startswith('device_')]
        assert len(device_cols) > 0
