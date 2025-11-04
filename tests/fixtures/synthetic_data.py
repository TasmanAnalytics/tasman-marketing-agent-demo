"""Synthetic data generators for analysis agent tests."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List


def generate_hypothesis_test_data(
    n_groups: int = 2,
    group_names: List[str] = None,
    base_rate: float = 0.05,
    effect_sizes: List[float] = None,
    n_per_group: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic data for hypothesis testing.

    Args:
        n_groups: Number of groups
        group_names: Names for groups
        base_rate: Base conversion rate
        effect_sizes: Relative lift per group (e.g., [0, 0.2] = 20% lift for group 2)
        n_per_group: Samples per group
        random_state: Random seed

    Returns:
        DataFrame with columns: group, trials, successes, rate
    """
    np.random.seed(random_state)

    if group_names is None:
        group_names = [f"Group_{i}" for i in range(n_groups)]

    if effect_sizes is None:
        effect_sizes = [0.0] * n_groups

    data = []
    for i, (group, effect) in enumerate(zip(group_names, effect_sizes)):
        rate = base_rate * (1 + effect)
        successes = np.random.binomial(n_per_group, rate)

        data.append({
            'group': group,
            'trials': n_per_group,
            'successes': successes,
            'rate': successes / n_per_group
        })

    return pd.DataFrame(data)


def generate_driver_analysis_data(
    n_samples: int = 10000,
    n_features: int = 5,
    feature_names: List[str] = None,
    categorical_cols: List[str] = None,
    target_type: str = 'binary',
    true_coefficients: Dict[str, float] = None,
    noise_level: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic data for driver analysis.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        feature_names: Feature names
        categorical_cols: Categorical feature names
        target_type: 'binary' or 'continuous'
        true_coefficients: True feature effects
        noise_level: Noise standard deviation
        random_state: Random seed

    Returns:
        (features_df, target_array)
    """
    np.random.seed(random_state)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    if categorical_cols is None:
        categorical_cols = []

    # Generate features
    data = {}
    for col in feature_names:
        if col in categorical_cols:
            # Categorical: 3-5 categories
            n_cats = np.random.randint(3, 6)
            data[col] = np.random.choice([f"cat_{i}" for i in range(n_cats)], size=n_samples)
        else:
            # Continuous: standard normal
            data[col] = np.random.randn(n_samples)

    df = pd.DataFrame(data)

    # Generate target with known coefficients
    if true_coefficients is None:
        true_coefficients = {col: np.random.randn() for col in feature_names if col not in categorical_cols}

    # Linear combination
    target = np.zeros(n_samples)
    for col, coef in true_coefficients.items():
        if col in df.columns and col not in categorical_cols:
            target += coef * df[col].values

    # Add noise
    target += np.random.randn(n_samples) * noise_level

    # Convert to binary if needed
    if target_type == 'binary':
        # Logistic transform
        prob = 1 / (1 + np.exp(-target))
        target = (np.random.rand(n_samples) < prob).astype(int)

    return df, target


def generate_segmentation_data(
    n_samples: int = 5000,
    n_true_clusters: int = 3,
    n_features: int = 4,
    feature_names: List[str] = None,
    cluster_separation: float = 2.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic data for segmentation with known clusters.

    Args:
        n_samples: Number of samples
        n_true_clusters: True number of clusters
        n_features: Number of features
        feature_names: Feature names
        cluster_separation: Distance between cluster centers
        random_state: Random seed

    Returns:
        (features_df, true_labels)
    """
    np.random.seed(random_state)

    if feature_names is None:
        feature_names = [f"metric_{i}" for i in range(n_features)]

    # Generate cluster centers
    centers = np.random.randn(n_true_clusters, n_features) * cluster_separation

    # Assign samples to clusters
    true_labels = np.random.choice(n_true_clusters, size=n_samples)

    # Generate data around centers
    data = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        cluster = true_labels[i]
        data[i] = centers[cluster] + np.random.randn(n_features) * 0.5

    df = pd.DataFrame(data, columns=feature_names)

    return df, true_labels


def generate_cohort_data(
    n_cohorts: int = 12,
    cohort_period: str = 'month',
    horizon_periods: int = 12,
    avg_cohort_size: int = 1000,
    base_retention: float = 0.5,
    retention_decay: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic cohort retention data.

    Args:
        n_cohorts: Number of cohorts
        cohort_period: 'month' or 'week'
        horizon_periods: Periods to track
        avg_cohort_size: Average cohort size
        base_retention: Initial retention rate
        retention_decay: Decay per period
        random_state: Random seed

    Returns:
        DataFrame with columns: cohort_month, month_offset, cohort_size, active_customers
    """
    np.random.seed(random_state)

    data = []

    for cohort_idx in range(n_cohorts):
        cohort_date = datetime(2024, 1, 1) + timedelta(days=30 * cohort_idx)
        cohort_size = int(avg_cohort_size * (1 + np.random.randn() * 0.2))

        for offset in range(horizon_periods):
            # Exponential decay retention
            retention = base_retention * np.exp(-retention_decay * offset)
            retention = min(1.0, max(0.0, retention))

            active = int(cohort_size * retention * (1 + np.random.randn() * 0.1))
            active = max(0, active)

            data.append({
                'cohort_month': cohort_date.strftime('%Y-%m-01'),
                'month_offset': offset,
                'cohort_size': cohort_size,
                'active_customers': active,
                'retention_rate': active / cohort_size if cohort_size > 0 else 0
            })

    return pd.DataFrame(data)


def generate_attribution_data(
    n_channels: int = 5,
    channel_names: List[str] = None,
    n_days: int = 90,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic attribution data with spend and revenue.

    Args:
        n_channels: Number of channels
        channel_names: Channel names
        n_days: Number of days
        random_state: Random seed

    Returns:
        DataFrame with columns: channel, date, spend, revenue, roas
    """
    np.random.seed(random_state)

    if channel_names is None:
        channel_names = ['Google', 'Facebook', 'LinkedIn', 'TikTok', 'Twitter'][:n_channels]

    # Channel efficiency (true ROAS)
    true_roas = {
        'Google': 3.5,
        'Facebook': 2.8,
        'LinkedIn': 4.2,
        'TikTok': 2.1,
        'Twitter': 1.8
    }

    data = []
    start_date = datetime(2024, 1, 1)

    for day in range(n_days):
        date = start_date + timedelta(days=day)

        for channel in channel_names:
            # Daily spend with some variation
            base_spend = np.random.uniform(500, 2000)
            spend = base_spend * (1 + np.random.randn() * 0.2)
            spend = max(0, spend)

            # Revenue based on ROAS + noise
            roas = true_roas.get(channel, 2.5)
            revenue = spend * roas * (1 + np.random.randn() * 0.3)
            revenue = max(0, revenue)

            data.append({
                'channel': channel,
                'date': date.strftime('%Y-%m-%d'),
                'spend': spend,
                'revenue': revenue,
                'roas': revenue / spend if spend > 0 else 0
            })

    return pd.DataFrame(data)


def generate_timeseries_with_anomalies(
    n_days: int = 90,
    base_value: float = 1000,
    trend: float = 0.0,
    seasonal_period: int = 7,
    seasonal_amplitude: float = 0.2,
    noise_level: float = 0.1,
    anomaly_dates: List[int] = None,
    anomaly_magnitudes: List[float] = None,
    changepoint_dates: List[int] = None,
    changepoint_deltas: List[float] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic time series with anomalies and change-points.

    Args:
        n_days: Number of days
        base_value: Starting value
        trend: Linear trend per day
        seasonal_period: Seasonal period (e.g., 7 for weekly)
        seasonal_amplitude: Seasonal variation amplitude
        noise_level: Gaussian noise std dev
        anomaly_dates: Days with anomalies
        anomaly_magnitudes: Spike magnitudes (in std devs)
        changepoint_dates: Days with structural breaks
        changepoint_deltas: Level shifts at breaks
        random_state: Random seed

    Returns:
        DataFrame with columns: date, value, is_anomaly, is_changepoint
    """
    np.random.seed(random_state)

    if anomaly_dates is None:
        anomaly_dates = []
    if anomaly_magnitudes is None:
        anomaly_magnitudes = []
    if changepoint_dates is None:
        changepoint_dates = []
    if changepoint_deltas is None:
        changepoint_deltas = []

    data = []
    start_date = datetime(2024, 1, 1)
    current_level = base_value

    for day in range(n_days):
        date = start_date + timedelta(days=day)

        # Check for change-point
        if day in changepoint_dates:
            idx = changepoint_dates.index(day)
            current_level += changepoint_deltas[idx]

        # Trend
        trend_component = trend * day

        # Seasonal
        seasonal_component = seasonal_amplitude * base_value * np.sin(2 * np.pi * day / seasonal_period)

        # Noise
        noise = np.random.randn() * noise_level * base_value

        # Base value
        value = current_level + trend_component + seasonal_component + noise

        # Anomaly
        is_anomaly = day in anomaly_dates
        if is_anomaly:
            idx = anomaly_dates.index(day)
            magnitude = anomaly_magnitudes[idx] if idx < len(anomaly_magnitudes) else 3.0
            value += magnitude * noise_level * base_value

        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'value': max(0, value),
            'is_anomaly': is_anomaly,
            'is_changepoint': day in changepoint_dates
        })

    return pd.DataFrame(data)


def generate_full_marketing_dataset(
    n_days: int = 90,
    n_campaigns: int = 20,
    n_customers: int = 5000,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate complete synthetic marketing dataset matching schema.

    Args:
        n_days: Days of data
        n_campaigns: Number of campaigns
        n_customers: Number of customers
        random_state: Random seed

    Returns:
        Dict of DataFrames: {
            'fact_ad_spend': ...,
            'fact_sessions': ...,
            'fact_orders': ...,
            'dim_campaigns': ...,
            'dim_customers': ...,
            'dim_products': ...
        }
    """
    np.random.seed(random_state)

    # Dimensions
    channels = ['Google', 'Facebook', 'LinkedIn', 'TikTok', 'Instagram']
    devices = ['mobile', 'desktop', 'tablet']
    regions = ['US', 'EU', 'APAC', 'LATAM']
    categories = ['Electronics', 'Apparel', 'Home', 'Sports', 'Books']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    # dim_campaigns
    dim_campaigns = pd.DataFrame({
        'campaign_id': range(1, n_campaigns + 1),
        'campaign_name': [f"Campaign_{i}" for i in range(1, n_campaigns + 1)],
        'channel': np.random.choice(channels, n_campaigns)
    })

    # dim_customers
    dim_customers = pd.DataFrame({
        'customer_id': [f"C{i:05d}" for i in range(1, n_customers + 1)],
        'region': np.random.choice(regions, n_customers),
        'segment': np.random.choice(['High', 'Medium', 'Low'], n_customers),
        'first_visit_date': pd.date_range('2023-01-01', periods=n_customers, freq='2H').strftime('%Y-%m-%d')
    })

    # dim_products
    n_products = 50
    dim_products = pd.DataFrame({
        'sku': [f"SKU{i:04d}" for i in range(1, n_products + 1)],
        'product_name': [f"Product_{i}" for i in range(1, n_products + 1)],
        'category': np.random.choice(categories, n_products),
        'brand': np.random.choice(brands, n_products),
        'unit_price': np.random.uniform(10, 500, n_products)
    })

    # fact_ad_spend
    start_date = datetime(2024, 1, 1)
    fact_ad_spend = []

    for day in range(n_days):
        date = (start_date + timedelta(days=day)).strftime('%Y-%m-%d')

        for campaign_id in range(1, n_campaigns + 1):
            spend = np.random.uniform(50, 500)
            impressions = int(spend * np.random.uniform(100, 500))
            clicks = int(impressions * np.random.uniform(0.01, 0.05))

            fact_ad_spend.append({
                'date': date,
                'campaign_id': campaign_id,
                'spend': spend,
                'impressions': impressions,
                'clicks': clicks
            })

    fact_ad_spend = pd.DataFrame(fact_ad_spend)

    # fact_sessions
    n_sessions = n_days * 100  # 100 sessions per day
    fact_sessions = pd.DataFrame({
        'session_id': [f"S{i:07d}" for i in range(1, n_sessions + 1)],
        'date': pd.date_range(start_date, periods=n_sessions, freq='13T').strftime('%Y-%m-%d'),
        'campaign_id': np.random.choice(range(1, n_campaigns + 1), n_sessions),
        'device': np.random.choice(devices, n_sessions),
        'pages_viewed': np.random.randint(1, 20, n_sessions),
        'converted_flag': np.random.rand(n_sessions) < 0.03  # 3% CVR
    })

    # fact_orders
    converted_sessions = fact_sessions[fact_sessions['converted_flag']].copy()
    n_orders = len(converted_sessions)

    fact_orders = pd.DataFrame({
        'order_id': [f"O{i:07d}" for i in range(1, n_orders + 1)],
        'session_id': converted_sessions['session_id'].values,
        'customer_id': np.random.choice(dim_customers['customer_id'], n_orders),
        'order_timestamp': pd.date_range(start_date, periods=n_orders, freq='47T').strftime('%Y-%m-%d %H:%M:%S'),
        'sku': np.random.choice(dim_products['sku'], n_orders),
        'quantity': np.random.randint(1, 5, n_orders),
        'revenue': np.random.uniform(20, 1000, n_orders),
        'margin': np.random.uniform(5, 300, n_orders)
    })

    return {
        'dim_campaigns': dim_campaigns,
        'dim_customers': dim_customers,
        'dim_products': dim_products,
        'fact_ad_spend': fact_ad_spend,
        'fact_sessions': fact_sessions,
        'fact_orders': fact_orders
    }
