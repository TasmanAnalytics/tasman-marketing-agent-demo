# Phase 2 Architecture & Implementation Plan
# Deep Analytics Agents - v0.2.0

**Status:** Planning Document
**Created:** 2025-11-03
**Target Release:** v0.2.0

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Agent Specifications](#agent-specifications)
4. [Testing Strategy](#testing-strategy)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Code Structure](#code-structure)
7. [Data Flow](#data-flow)
8. [Configuration](#configuration)
9. [Acceptance Criteria](#acceptance-criteria)

---

## Executive Summary

### Objective
Extend the v0.1.0 local-first agentic analytics system with **6 deep analysis agents** that perform statistical analysis, modeling, and hypothesis testing - all with minimal LLM usage.

### Core Principles
1. **Local-first**: All statistical computation in Python/DuckDB; LLM only for plan generation
2. **Row policy**: Aggregations unlimited; row-level capped at 250k with stratified sampling
3. **Statistical rigor**: 95% CIs, Benjamini-Hochberg FDR correction for multiple testing
4. **Reproducibility**: JSON plans + versioned seeds + SQL tracking
5. **Testing**: Unit tests + integration tests + golden-file plan validation

### The 6 Analysis Agents

| Agent | Purpose | Primary Methods | Key Outputs |
|-------|---------|-----------------|-------------|
| **HypothesisTestingAgent** | Test assertions (CTR/CVR comparisons) | z-test, t-test, χ², ANOVA | p-values, effect sizes, CIs |
| **DriverAnalysisAgent** | Identify outcome drivers | Logistic/linear regression | Feature importances, ORs |
| **SegmentationAgent** | Customer/campaign segments | K-means, RFM, DBSCAN | Segment profiles, sizes |
| **CohortAndRetentionAgent** | Cohort analysis & retention | Cohort matrices, curves | Retention rates, LTV proxies |
| **AttributionAndROASAgent** | Revenue attribution | Last-touch, variance decomp | ROAS by channel, attribution |
| **AnomalyAndTrendBreakAgent** | Detect spikes/dips | STL, PELT, z-scores | Anomalies, change-points |

### Success Metrics
- **Zero LLM calls** for plan generation (local-first templates)
- **< 3s execution** for 250k row analyses
- **100% test coverage** for all agents
- **Reproducible results** with fixed seeds

---

## System Architecture

### High-Level Flow

```
User Question ("What drives conversion?")
    ↓
[TriageAgent] → mode = "analysis" (Phase 1)
    ↓
[AnalysisPlanBuilder] → selects agent + builds plan (NEW)
    ↓
[Specific Analysis Agent] → executes analysis (NEW)
    ↓
    1. plan() → JSON plan
    2. pull() → SQL + DataFrame
    3. run() → statistical analysis
    4. report() → results + charts + insights
    ↓
Output: JSON + DataFrame + Charts + Markdown summary
```

### Layer Architecture (Extended from Phase 1)

```
┌─────────────────────────────────────────────────────────┐
│                    USER QUESTION                        │
│         "What drives conversion on mobile?"             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│            TriageAgent (Phase 1 - Enhanced)             │
│  Returns: mode="analysis", analysis_type="driver"       │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│          AnalysisPlanBuilder (NEW - Phase 2)            │
│  • Maps question → agent_id                             │
│  • Builds JSON plan (target, features, window)          │
│  • LLM fallback only if ambiguous                       │
└─────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────┴─────────────────┐
        │      Route to Specific Agent      │
        └───────────────┬───────────────────┘
                        ↓
    ┌────────┬────────┬────────┬────────┬────────┬────────┐
    │Hypothe-│Driver  │Segment-│Cohort  │Attribu-│Anomaly │
    │sis     │Analysis│ation   │Retent. │tion    │Trend   │
    └────┬───┴────┬───┴────┬───┴────┬───┴────┬───┴────┬───┘
         │        │        │        │        │        │
         └────────┴────────┴────────┴────────┴────────┘
                          ↓
         ┌────────────────────────────────┐
         │   Common Agent Workflow        │
         │  1. plan() → JSON              │
         │  2. pull() → SQL + DataFrame   │
         │  3. run() → analysis results   │
         │  4. report() → output dict     │
         └────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                Core Analysis Utilities                   │
│  • StratifiedSampler (analysis_utils.py)                │
│  • CategoricalEncoder                                    │
│  • Statistical functions (CIs, FDR, tests)              │
│  • DuckDBConnector (Phase 1)                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Configuration Layer                     │
│  • config/analysis.yaml (row caps, windows, stats)      │
│  • config/business_context.yaml (roles, KPIs)           │
│  • config/schema.json (database schema)                 │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                        OUTPUT                            │
│  • JSON plan (reproducible)                             │
│  • SQL queries executed                                 │
│  • Results DataFrame                                    │
│  • Charts (PNG/PDF)                                     │
│  • Markdown insights + next actions                     │
│  • Diagnostics (sample sizes, assumptions, caveats)     │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Specifications

### Base Agent Interface

All analysis agents inherit from a common base:

```python
class BaseAnalysisAgent:
    """Base class for all analysis agents."""

    def __init__(self, db_connector, config, llm_client=None):
        self.db = db_connector
        self.config = config
        self.llm_client = llm_client

    def plan(self, question: str, role: str, context: dict) -> dict:
        """Generate analysis plan (JSON)."""
        pass

    def pull(self, plan: dict) -> pd.DataFrame:
        """Execute SQL and return DataFrame with sampling."""
        pass

    def run(self, df: pd.DataFrame, plan: dict) -> dict:
        """Perform statistical analysis."""
        pass

    def report(self, results: dict, plan: dict) -> dict:
        """Generate final report with charts and insights."""
        pass

    def execute(self, question: str, role: str, context: dict) -> dict:
        """End-to-end: plan → pull → run → report."""
        plan = self.plan(question, role, context)
        df = self.pull(plan)
        results = self.run(df, plan)
        return self.report(results, plan)
```

### Agent Output Schema

All agents return a standardized output:

```python
{
    "agent_id": str,              # e.g., "hypothesis_testing"
    "plan": {                      # Reproducible plan
        "version": "1.0",
        "target": str,
        "unit": str,               # session | order | customer | channel-day
        "window_days": int,
        "sql_blocks": [str],
        "methods": [str],
        "seed": int
    },
    "data_spec": {                 # Data pulled
        "row_count": int,
        "sample_ratio": float,     # 1.0 if no sampling
        "strata_distribution": dict,
        "columns": [str],
        "dtypes": dict
    },
    "results": {                   # Agent-specific results
        # HypothesisTestingAgent:
        "p_value": float,
        "effect": {"abs": float, "rel": float},
        "ci": [float, float],

        # DriverAnalysisAgent:
        "model_metrics": {"auc": float, "r2": float},
        "drivers": [{"feature": str, "importance": float, "p": float}],

        # etc.
    },
    "diagnostics": {               # Checks and assumptions
        "sample_sizes": dict,
        "assumptions": [str],
        "warnings": [str],
        "confidence_level": float
    },
    "visualizations": {            # Generated charts
        "chart_paths": [str],
        "chart_types": [str]
    },
    "insights": [str],             # Natural language findings
    "caveats": [str],              # Limitations and warnings
    "next_actions": [str],         # Recommended follow-ups
    "execution_time_ms": float
}
```

---

### 1. HypothesisTestingAgent

**Purpose:** Test targeted assertions (e.g., "LinkedIn CTR > Facebook CTR")

**Plan Schema:**
```python
{
    "metric": str,              # ctr | cvr | aov | margin
    "groups": [str],            # [channel1, channel2] or dimension values
    "dimension": str,           # channel | device | region
    "window_days": int,
    "test_type": str,           # auto | z_test | t_test | chi_squared | anova
    "alternative": str,         # two_sided | greater | less
    "alpha": float,             # 0.05
    "corrections": bool         # Apply BH-FDR if > 2 groups
}
```

**SQL Templates:**

```python
# CTR comparison by channel
TEMPLATES = {
    "ctr_by_dimension": """
        SELECT {dimension},
               SUM(clicks) AS clicks,
               SUM(impressions) AS impressions,
               CASE WHEN SUM(impressions) = 0 THEN NULL
                    ELSE SUM(clicks)::DOUBLE / SUM(impressions) END AS ctr
        FROM fact_ad_spend f
        JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
        WHERE f.date >= CURRENT_DATE - INTERVAL {window_days} DAY
        GROUP BY 1
    """,

    "cvr_by_dimension": """
        SELECT {dimension},
               COUNT(*) AS sessions,
               SUM(CASE WHEN converted_flag THEN 1 ELSE 0 END) AS conversions,
               CASE WHEN COUNT(*) = 0 THEN NULL
                    ELSE SUM(CASE WHEN converted_flag THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) END AS cvr
        FROM fact_sessions s
        LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
        WHERE s.date >= CURRENT_DATE - INTERVAL {window_days} DAY
        GROUP BY 1
    """
}
```

**Analysis Logic:**
```python
def run(self, df, plan):
    metric = plan['metric']
    groups = plan['groups']

    # Extract group data
    group_data = {}
    for group in groups:
        group_df = df[df[plan['dimension']] == group]
        group_data[group] = {
            'successes': group_df['conversions'].iloc[0],
            'trials': group_df['sessions'].iloc[0]
        }

    # Choose test
    if metric in ['ctr', 'cvr']:  # Proportions
        if len(groups) == 2:
            result = proportion_test(
                group_data[groups[0]]['successes'],
                group_data[groups[0]]['trials'],
                group_data[groups[1]]['successes'],
                group_data[groups[1]]['trials']
            )
        else:  # > 2 groups: chi-squared + post-hoc
            result = chi_squared_test(group_data)
            if plan['corrections']:
                result['pairwise'] = pairwise_with_fdr(group_data)

    elif metric in ['aov', 'margin']:  # Continuous
        if len(groups) == 2:
            result = welch_ttest(
                group_data[groups[0]]['values'],
                group_data[groups[1]]['values']
            )
        else:  # ANOVA
            result = anova_test(group_data)

    return result
```

**Testing:**
- Unit test: 2-sample z-test with known p-value
- Integration test: CTR comparison Facebook vs Google (synthetic data)
- Golden file: Plan JSON for "LinkedIn CTR > Facebook CTR"

---

### 2. DriverAnalysisAgent

**Purpose:** Identify drivers of an outcome (CVR, orders, revenue)

**Plan Schema:**
```python
{
    "target": str,              # converted_flag | revenue | margin
    "target_type": str,         # binary | continuous
    "unit": str,                # session | order
    "features": [str],          # [device, channel, pages_viewed, ...]
    "window_days": int,
    "model_type": str,          # logistic | linear | lasso | ridge
    "max_rows": int,            # 250000
    "sampling_strata": [str],   # [channel, device]
    "cv_folds": int,            # 5
    "exclude_patterns": [str]   # Leakage prevention
}
```

**SQL Template:**
```python
# Session-level features for CVR prediction
"""
SELECT
  s.session_id,
  s.device,
  c.channel,
  c.campaign_name,
  s.pages_viewed,
  s.converted_flag::INT AS target
FROM fact_sessions s
LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
WHERE s.date >= CURRENT_DATE - INTERVAL {window_days} DAY
"""
```

**Analysis Logic:**
```python
def run(self, df, plan):
    # Prepare features
    target = df[plan['target']]

    # Encode categoricals
    encoder = CategoricalEncoder(
        max_categories=plan.get('max_categories', 20)
    )
    df_encoded = encoder.fit_transform(df, plan['features'])

    X = df_encoded.drop(columns=[plan['target']])
    y = target

    # Choose model
    if plan['target_type'] == 'binary':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty='l2', max_iter=1000)
    else:
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=plan.get('alpha', 1.0))

    # Cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(
        model, X, y,
        cv=plan['cv_folds'],
        scoring='roc_auc' if plan['target_type'] == 'binary' else 'r2'
    )

    # Fit final model
    model.fit(X, y)

    # Feature importances (permutation)
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=10,
        random_state=plan['seed']
    )

    # Build driver table
    drivers = []
    for i, col in enumerate(X.columns):
        drivers.append({
            'feature': col,
            'importance': perm_importance.importances_mean[i],
            'importance_std': perm_importance.importances_std[i]
        })

    drivers = sorted(drivers, key=lambda x: x['importance'], reverse=True)

    return {
        'model_metrics': {
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        },
        'drivers': drivers[:20]  # Top 20
    }
```

**Testing:**
- Unit test: Logistic regression on synthetic binary outcome
- Integration test: CVR drivers with known feature importances
- Golden file: Plan for "What drives conversion?"

---

### 3. SegmentationAgent

**Purpose:** Create actionable customer/campaign segments

**Plan Schema:**
```python
{
    "method": str,              # kmeans | rfm | dbscan
    "unit": str,                # customer | campaign
    "features": [str],          # [orders, revenue, margin, ...]
    "window_days": int,
    "k": int,                   # For k-means (auto if null)
    "k_range": [int, int],      # [2, 10] for silhouette search
    "standardize": bool,        # True
    "outlier_handling": str     # winsorize | clip | remove
}
```

**SQL Template:**
```python
# Customer aggregates for segmentation
"""
WITH cust AS (
  SELECT o.customer_id,
         COUNT(DISTINCT o.order_id) AS orders,
         SUM(o.revenue) AS revenue,
         SUM(o.margin) AS margin,
         MAX(o.order_timestamp) AS last_order_date,
         MIN(o.order_timestamp) AS first_order_date
  FROM fact_orders o
  WHERE o.order_timestamp >= NOW() - INTERVAL {window_days} DAY
  GROUP BY 1
),
acq AS (
  SELECT customer_id, acquisition_channel, region
  FROM dim_customers
)
SELECT *,
       DATE_DIFF('day', NOW(), cust.last_order_date) AS recency_days,
       DATE_DIFF('day', cust.first_order_date, cust.last_order_date) AS customer_age_days
FROM cust
LEFT JOIN acq USING(customer_id)
"""
```

**Analysis Logic:**
```python
def run(self, df, plan):
    features = plan['features']

    # Handle outliers
    if plan['outlier_handling'] == 'winsorize':
        for col in features:
            df[col] = winsorize(df[col])

    # Standardize
    if plan['standardize']:
        df_scaled, scaler = standardize_features(df, features)

    X = df_scaled[features]

    if plan['method'] == 'kmeans':
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Find optimal k
        if plan['k'] is None:
            k_range = range(plan['k_range'][0], plan['k_range'][1] + 1)
            silhouette_scores = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=plan['seed'])
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)

            best_k = k_range[np.argmax(silhouette_scores)]
        else:
            best_k = plan['k']

        # Fit with best k
        kmeans = KMeans(n_clusters=best_k, random_state=plan['seed'])
        labels = kmeans.fit_predict(X)

        # Profile segments
        df['segment'] = labels
        segments = []

        for seg_id in range(best_k):
            seg_df = df[df['segment'] == seg_id]
            profile = {
                'segment_id': seg_id,
                'size': len(seg_df),
                'share': len(seg_df) / len(df),
                'profile': {
                    col: seg_df[col].mean() for col in features
                }
            }
            segments.append(profile)

        return {
            'method': 'kmeans',
            'k': best_k,
            'silhouette_score': silhouette_score(X, labels),
            'segments': segments
        }

    elif plan['method'] == 'rfm':
        # RFM quantiles
        df['R_score'] = pd.qcut(df['recency_days'], 5, labels=False, duplicates='drop')
        df['F_score'] = pd.qcut(df['orders'], 5, labels=False, duplicates='drop')
        df['M_score'] = pd.qcut(df['revenue'], 5, labels=False, duplicates='drop')

        df['RFM_segment'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)

        # Profile top segments
        segment_profiles = df.groupby('RFM_segment').agg({
            'customer_id': 'count',
            'orders': 'mean',
            'revenue': 'mean',
            'margin': 'mean'
        }).reset_index()

        segment_profiles.columns = ['segment', 'size', 'avg_orders', 'avg_revenue', 'avg_margin']
        segment_profiles = segment_profiles.sort_values('avg_revenue', ascending=False)

        return {
            'method': 'rfm',
            'n_segments': len(segment_profiles),
            'segments': segment_profiles.head(10).to_dict('records')
        }
```

**Testing:**
- Unit test: K-means on synthetic 2D clusters
- Integration test: RFM segmentation with known quantiles
- Golden file: Plan for "Segment customers by value"

---

### 4. CohortAndRetentionAgent

**Purpose:** First-visit cohorts; measure retention, repeat orders

**Plan Schema:**
```python
{
    "cohort_period": str,       # month | week | quarter
    "cohort_key": str,          # first_visit_date
    "horizon_periods": int,     # 12 months
    "metrics": [str],           # [retention_rate, orders, revenue]
    "retention_definition": str,# active_orders | any_activity
    "window_months": int,       # Look back window
    "baseline_cohorts": int     # First N cohorts for comparison
}
```

**SQL Template:**
```python
# Monthly cohorts with retention
"""
WITH cohorts AS (
  SELECT customer_id,
         DATE_TRUNC('month', first_visit_date) AS cohort_month
  FROM dim_customers
),
orders AS (
  SELECT customer_id,
         DATE_TRUNC('month', order_timestamp) AS order_month,
         COUNT(DISTINCT order_id) AS orders,
         SUM(revenue) AS revenue
  FROM fact_orders
  GROUP BY 1, 2
)
SELECT c.cohort_month,
       DATE_DIFF('month', c.cohort_month, o.order_month) AS month_offset,
       COUNT(DISTINCT c.customer_id) AS cohort_size,
       COUNT(DISTINCT o.customer_id) AS active_customers,
       SUM(o.orders) AS total_orders,
       SUM(o.revenue) AS total_revenue
FROM cohorts c
LEFT JOIN orders o USING(customer_id)
WHERE c.cohort_month >= DATE_TRUNC('month', NOW() - INTERVAL {window_months} MONTH)
GROUP BY 1, 2
ORDER BY 1, 2
"""
```

**Analysis Logic:**
```python
def run(self, df, plan):
    # Pivot to cohort matrix
    cohort_matrix = df.pivot_table(
        index='cohort_month',
        columns='month_offset',
        values='active_customers',
        fill_value=0
    )

    # Calculate retention rates
    cohort_sizes = cohort_matrix[0]  # Month 0 = cohort size
    retention_matrix = cohort_matrix.div(cohort_sizes, axis=0) * 100

    # Compare latest vs baseline
    baseline = retention_matrix.head(plan['baseline_cohorts']).mean()
    latest = retention_matrix.tail(3).mean()

    delta = latest - baseline

    # Significance test (t-test on retention rates)
    baseline_vals = retention_matrix.head(plan['baseline_cohorts']).values.flatten()
    latest_vals = retention_matrix.tail(3).values.flatten()

    t_result = welch_ttest(latest_vals, baseline_vals)

    return {
        'cohort_matrix': cohort_matrix.to_dict(),
        'retention_matrix': retention_matrix.to_dict(),
        'baseline_retention': baseline.to_dict(),
        'latest_retention': latest.to_dict(),
        'delta': delta.to_dict(),
        'significance': {
            'p_value': t_result['p_value'],
            't_stat': t_result['t_stat']
        }
    }
```

**Testing:**
- Unit test: Cohort matrix pivot calculation
- Integration test: Retention curves with synthetic cohort data
- Golden file: Plan for "Compare cohort retention"

---

### 5. AttributionAndROASAgent

**Purpose:** Revenue attribution & ROAS decomposition

**Plan Schema:**
```python
{
    "attribution_model": str,   # last_touch | first_touch | equal_split
    "window_days": int,
    "unit": str,                # channel | campaign
    "variance_decomposition": bool,
    "sensitivity_windows": [int],  # [60, 90, 120]
    "min_spend_threshold": float
}
```

**SQL Template:**
```python
# Last-touch attribution
"""
WITH rev AS (
  SELECT s.campaign_id,
         SUM(o.revenue) AS revenue,
         COUNT(DISTINCT o.order_id) AS orders
  FROM fact_orders o
  JOIN fact_sessions s ON o.session_id = s.session_id
  WHERE o.order_timestamp >= NOW() - INTERVAL {window_days} DAY
  GROUP BY 1
),
spend AS (
  SELECT campaign_id,
         SUM(spend) AS spend
  FROM fact_ad_spend
  WHERE date >= CURRENT_DATE - INTERVAL {window_days} DAY
  GROUP BY 1
)
SELECT c.channel,
       c.campaign_name,
       COALESCE(spend.spend, 0) AS spend,
       COALESCE(rev.revenue, 0) AS revenue,
       COALESCE(rev.orders, 0) AS orders,
       CASE WHEN spend.spend = 0 OR spend.spend IS NULL THEN NULL
            ELSE rev.revenue::DOUBLE / spend.spend END AS roas
FROM dim_campaigns c
LEFT JOIN spend USING(campaign_id)
LEFT JOIN rev USING(campaign_id)
WHERE COALESCE(spend.spend, 0) >= {min_spend_threshold}
ORDER BY roas DESC NULLS LAST
"""
```

**Analysis Logic:**
```python
def run(self, df, plan):
    # Filter by minimum spend
    df = df[df['spend'] >= plan['min_spend_threshold']]

    # Winsorize ROAS (remove extreme outliers)
    df['roas'] = winsorize(df['roas'], limits=(0.05, 0.95))

    # Variance decomposition (between vs within channel)
    if plan['variance_decomposition']:
        from scipy.stats import f_oneway

        channels = df['channel'].unique()
        channel_roas = [df[df['channel'] == ch]['roas'].dropna() for ch in channels]

        f_stat, p_value = f_oneway(*channel_roas)

        # Calculate variance components
        grand_mean = df['roas'].mean()

        between_var = sum(
            len(df[df['channel'] == ch]) * (df[df['channel'] == ch]['roas'].mean() - grand_mean)**2
            for ch in channels
        ) / (len(channels) - 1)

        within_var = df.groupby('channel')['roas'].var().mean()

        total_var = df['roas'].var()

        variance_explained = between_var / total_var if total_var > 0 else 0

    # Sensitivity to time window
    sensitivity = []
    for window in plan['sensitivity_windows']:
        # Re-run with different window (simplified - would actually re-query)
        sensitivity.append({
            'window_days': window,
            'mean_roas': df['roas'].mean(),  # Placeholder
            'median_roas': df['roas'].median()
        })

    # Bootstrap CIs for ROAS
    roas_ci = bootstrap_ci(df['roas'].dropna(), np.mean)

    return {
        'attribution_model': plan['attribution_model'],
        'summary': {
            'mean_roas': df['roas'].mean(),
            'median_roas': df['roas'].median(),
            'roas_ci': roas_ci
        },
        'variance_decomposition': {
            'between_channel_variance': between_var,
            'within_channel_variance': within_var,
            'variance_explained': variance_explained,
            'f_stat': f_stat,
            'p_value': p_value
        } if plan['variance_decomposition'] else None,
        'sensitivity': sensitivity,
        'top_performers': df.nlargest(10, 'roas').to_dict('records')
    }
```

**Testing:**
- Unit test: ROAS calculation with known spend/revenue
- Integration test: Variance decomposition with synthetic channel data
- Golden file: Plan for "ROAS by channel"

---

### 6. AnomalyAndTrendBreakAgent

**Purpose:** Detect spikes/dips or structural breaks

**Plan Schema:**
```python
{
    "metrics": [str],           # [spend, ctr, cvr, revenue]
    "window_days": int,
    "detection_method": str,    # zscore | iqr | stl_residuals
    "zscore_threshold": float,  # 3.0
    "changepoint_algorithm": str,  # pelt | binary_segmentation
    "confirmation_test": bool,  # Run t-test pre/post break
    "seasonal_period": int      # 7 for weekly
}
```

**SQL Template:**
```python
# Daily metrics time series
"""
SELECT f.date,
       SUM(f.spend) AS spend,
       SUM(f.impressions) AS impressions,
       SUM(f.clicks) AS clicks,
       CASE WHEN SUM(f.impressions) = 0 THEN NULL
            ELSE SUM(f.clicks)::DOUBLE / SUM(f.impressions) END AS ctr
FROM fact_ad_spend f
WHERE f.date >= CURRENT_DATE - INTERVAL {window_days} DAY
GROUP BY 1
ORDER BY 1
"""
```

**Analysis Logic:**
```python
def run(self, df, plan):
    import ruptures as rpt
    from statsmodels.tsa.seasonal import STL

    results = {}

    for metric in plan['metrics']:
        series = df[metric].fillna(method='ffill').values

        # Anomaly detection
        if plan['detection_method'] == 'stl_residuals':
            # STL decomposition
            stl = STL(series, period=plan['seasonal_period'], robust=True)
            res = stl.fit()

            residuals = res.resid
            threshold = plan['zscore_threshold'] * np.std(residuals)

            anomalies_idx = np.where(np.abs(residuals) > threshold)[0]
            anomalies = [
                {
                    'date': df.iloc[i]['date'],
                    'value': series[i],
                    'residual': residuals[i],
                    'z_score': residuals[i] / np.std(residuals)
                }
                for i in anomalies_idx
            ]

        elif plan['detection_method'] == 'zscore':
            mean = np.mean(series)
            std = np.std(series)
            z_scores = (series - mean) / std

            anomalies_idx = np.where(np.abs(z_scores) > plan['zscore_threshold'])[0]
            anomalies = [
                {
                    'date': df.iloc[i]['date'],
                    'value': series[i],
                    'z_score': z_scores[i]
                }
                for i in anomalies_idx
            ]

        # Change-point detection
        if plan['changepoint_algorithm'] == 'pelt':
            algo = rpt.Pelt(model='rbf').fit(series)
            breakpoints = algo.predict(pen=plan.get('changepoint_penalty', 10))
        else:
            algo = rpt.Binseg(model='l2').fit(series)
            breakpoints = algo.predict(n_bkps=5)

        # Remove last point (end of series)
        breakpoints = [bp for bp in breakpoints if bp < len(series)]

        # Confirm breaks with t-test
        confirmed_breaks = []
        for bp in breakpoints:
            if bp < 7 or bp > len(series) - 7:  # Need min segment length
                continue

            pre = series[max(0, bp-30):bp]
            post = series[bp:min(len(series), bp+30)]

            if len(pre) >= 7 and len(post) >= 7:
                t_result = welch_ttest(pre, post)

                if t_result['p_value'] < 0.05:
                    confirmed_breaks.append({
                        'date': df.iloc[bp]['date'],
                        'pre_mean': t_result['mean1'],
                        'post_mean': t_result['mean2'],
                        'change': t_result['diff'],
                        'p_value': t_result['p_value']
                    })

        results[metric] = {
            'anomalies': sorted(anomalies, key=lambda x: abs(x.get('z_score', 0)), reverse=True)[:10],
            'breakpoints': confirmed_breaks if plan['confirmation_test'] else [
                {'date': df.iloc[bp]['date'], 'index': bp} for bp in breakpoints
            ]
        }

    return results
```

**Testing:**
- Unit test: Z-score anomaly detection on synthetic spike
- Integration test: PELT change-point on step function
- Golden file: Plan for "Detect CTR anomalies"

---

## Testing Strategy

### Test Infrastructure

```
tests/
├── test_analysis_utils.py       # Core utilities (sampling, encoding, stats)
├── test_analysis_plan_builder.py
├── test_hypothesis_agent.py
├── test_driver_agent.py
├── test_segmentation_agent.py
├── test_cohort_agent.py
├── test_attribution_agent.py
├── test_anomaly_agent.py
├── fixtures/
│   ├── synthetic_data.py         # Generate test datasets
│   ├── golden_plans/             # Expected plan JSONs
│   │   ├── hypothesis_ctr.json
│   │   ├── driver_cvr.json
│   │   └── ...
│   └── expected_results/         # Known statistical results
│       ├── hypothesis_ctr.json
│       └── ...
└── integration/
    └── test_analysis_end_to_end.py
```

### Test Levels

#### 1. Unit Tests (Core Utilities)

**test_analysis_utils.py:**
```python
import pytest
import numpy as np
from core.analysis_utils import (
    StratifiedSampler,
    proportion_test,
    welch_ttest,
    benjamini_hochberg,
    bootstrap_ci
)

def test_proportion_test_known_result():
    """Test z-test with known p-value."""
    # Group 1: 50/100 = 0.50
    # Group 2: 30/100 = 0.30
    # Expected z ≈ 3.27, p ≈ 0.001

    result = proportion_test(50, 100, 30, 100)

    assert result['p1'] == 0.50
    assert result['p2'] == 0.30
    assert result['diff'] == pytest.approx(0.20, abs=0.01)
    assert result['p_value'] < 0.01
    assert result['z_stat'] > 3.0

def test_benjamini_hochberg_correction():
    """Test BH-FDR correction."""
    pvalues = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060]

    rejected, adjusted = benjamini_hochberg(pvalues, alpha=0.05)

    # With alpha=0.05, expect first 4-5 to be rejected
    assert sum(rejected) >= 4
    assert adjusted[0] < pvalues[0]  # Adjusted should be different

def test_stratified_sampler():
    """Test stratified sampling preserves distribution."""
    import pandas as pd

    # Create imbalanced dataset
    df = pd.DataFrame({
        'channel': ['Google']*7000 + ['Facebook']*2000 + ['LinkedIn']*1000,
        'value': np.random.randn(10000)
    })

    sampler = StratifiedSampler(max_rows=1000, strata_cols=['channel'])
    sampled = sampler.fit_sample(df)

    assert len(sampled) == 1000
    assert sampler.sample_ratio_ == 0.1

    # Check distribution preserved (approximately)
    original_dist = df['channel'].value_counts(normalize=True)
    sampled_dist = sampled['channel'].value_counts(normalize=True)

    for channel in original_dist.index:
        assert abs(original_dist[channel] - sampled_dist[channel]) < 0.1

def test_bootstrap_ci_coverage():
    """Test bootstrap CI has correct coverage."""
    np.random.seed(42)
    data = np.random.normal(10, 2, size=1000)

    point, lower, upper = bootstrap_ci(data, np.mean, n_iterations=1000)

    # Point estimate should be close to true mean (10)
    assert point == pytest.approx(10, abs=0.2)

    # CI should contain true mean
    assert lower < 10 < upper

    # CI width should be reasonable
    assert (upper - lower) < 1.0
```

#### 2. Unit Tests (Agent Logic)

**test_hypothesis_agent.py:**
```python
import pytest
import pandas as pd
from agents.agent_hypothesis import HypothesisTestingAgent

@pytest.fixture
def hypothesis_agent(db_connector, config):
    return HypothesisTestingAgent(db_connector, config)

def test_plan_generation_ctr_comparison(hypothesis_agent):
    """Test plan generation for CTR comparison."""
    question = "Is LinkedIn CTR better than Facebook CTR?"
    role = "marketer"

    plan = hypothesis_agent.plan(question, role, {})

    assert plan['metric'] == 'ctr'
    assert 'LinkedIn' in plan['groups']
    assert 'Facebook' in plan['groups']
    assert plan['dimension'] == 'channel'
    assert plan['test_type'] in ['z_test', 'auto']
    assert plan['alternative'] == 'two_sided'

def test_run_proportion_test_synthetic(hypothesis_agent):
    """Test proportion test execution with synthetic data."""
    df = pd.DataFrame({
        'channel': ['LinkedIn', 'Facebook'],
        'clicks': [500, 300],
        'impressions': [10000, 10000],
        'ctr': [0.05, 0.03]
    })

    plan = {
        'metric': 'ctr',
        'groups': ['LinkedIn', 'Facebook'],
        'dimension': 'channel',
        'test_type': 'z_test'
    }

    result = hypothesis_agent.run(df, plan)

    assert 'p_value' in result
    assert 'effect' in result
    assert result['effect']['abs'] == pytest.approx(0.02, abs=0.001)
    assert result['p_value'] < 0.05  # Significant difference

def test_multiple_comparison_correction(hypothesis_agent):
    """Test BH-FDR correction for >2 groups."""
    df = pd.DataFrame({
        'channel': ['Google', 'Facebook', 'LinkedIn', 'TikTok'],
        'clicks': [1000, 800, 500, 300],
        'impressions': [20000, 20000, 10000, 10000],
        'ctr': [0.05, 0.04, 0.05, 0.03]
    })

    plan = {
        'metric': 'ctr',
        'groups': ['Google', 'Facebook', 'LinkedIn', 'TikTok'],
        'dimension': 'channel',
        'test_type': 'chi_squared',
        'corrections': True
    }

    result = hypothesis_agent.run(df, plan)

    assert 'pairwise' in result
    assert 'adjusted_pvalues' in result['pairwise']
```

#### 3. Golden File Tests

**test_golden_plans.py:**
```python
import pytest
import json
from pathlib import Path

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'golden_plans'

def load_golden_plan(agent_id, test_case):
    """Load golden plan JSON."""
    path = GOLDEN_DIR / f"{agent_id}_{test_case}.json"
    with open(path) as f:
        return json.load(f)

def test_hypothesis_ctr_plan_matches_golden(hypothesis_agent):
    """Test that plan matches golden file."""
    question = "Is LinkedIn CTR better than Facebook CTR in the last 90 days?"
    role = "marketer"

    plan = hypothesis_agent.plan(question, role, {})
    golden = load_golden_plan('hypothesis', 'ctr_comparison')

    # Check key fields match
    assert plan['metric'] == golden['metric']
    assert set(plan['groups']) == set(golden['groups'])
    assert plan['dimension'] == golden['dimension']
    assert plan['window_days'] == golden['window_days']

def test_driver_cvr_plan_matches_golden(driver_agent):
    """Test driver analysis plan matches golden."""
    question = "What drives conversion on mobile?"
    role = "marketer"

    plan = driver_agent.plan(question, role, {})
    golden = load_golden_plan('driver', 'cvr_mobile')

    assert plan['target'] == golden['target']
    assert plan['target_type'] == golden['target_type']
    assert set(plan['features']) == set(golden['features'])
```

#### 4. Integration Tests

**test_analysis_end_to_end.py:**
```python
import pytest
from agents.agent_analysis_plan import AnalysisPlanBuilder

@pytest.fixture
def full_stack(db_connector, config, all_agents):
    """Full analysis stack with all agents."""
    return AnalysisPlanBuilder(
        db_connector=db_connector,
        config=config,
        agents=all_agents
    )

def test_end_to_end_hypothesis_ctr(full_stack):
    """Test full hypothesis testing workflow."""
    question = "Is LinkedIn CTR better than Facebook CTR?"
    role = "marketer"

    result = full_stack.execute(question, role)

    assert result['agent_id'] == 'hypothesis_testing'
    assert result['plan'] is not None
    assert result['data_spec']['row_count'] > 0
    assert 'p_value' in result['results']
    assert len(result['insights']) > 0
    assert len(result['next_actions']) > 0

def test_end_to_end_driver_cvr(full_stack):
    """Test full driver analysis workflow."""
    question = "What drives conversion?"
    role = "marketer"

    result = full_stack.execute(question, role)

    assert result['agent_id'] == 'driver_analysis'
    assert result['data_spec']['sample_ratio'] <= 1.0
    assert 'drivers' in result['results']
    assert len(result['results']['drivers']) > 0
    assert 'model_metrics' in result['results']

@pytest.mark.slow
def test_performance_250k_rows(full_stack):
    """Test that 250k row analysis completes in < 3s."""
    import time

    question = "What drives conversion?"
    role = "marketer"

    start = time.time()
    result = full_stack.execute(question, role)
    elapsed = time.time() - start

    assert elapsed < 3.0
    assert result['data_spec']['row_count'] <= 250000
```

#### 5. Reproducibility Tests

**test_reproducibility.py:**
```python
def test_analysis_reproducible_with_seed(driver_agent):
    """Test that analysis is reproducible with fixed seed."""
    question = "What drives conversion?"
    role = "marketer"

    # Run twice with same seed
    result1 = driver_agent.execute(question, role, seed=42)
    result2 = driver_agent.execute(question, role, seed=42)

    # Results should be identical
    assert result1['results']['drivers'] == result2['results']['drivers']
    assert result1['results']['model_metrics'] == result2['results']['model_metrics']

def test_plan_version_tracked(hypothesis_agent):
    """Test that plan includes version for reproducibility."""
    plan = hypothesis_agent.plan("CTR comparison", "marketer", {})

    assert 'version' in plan
    assert 'seed' in plan
    assert plan['version'] == '1.0'
```

### Test Coverage Goals

- **Core utilities**: 100% line coverage
- **Agent logic**: 95% line coverage
- **Integration**: All 6 agents × 2-3 test cases = 12-18 tests
- **Golden files**: 6 agents × 1 plan = 6 golden files minimum
- **Performance**: All agents < 3s for 250k rows

---

## Implementation Roadmap

### Phase 2.1: Foundation (Week 1)
- ✅ Add dependencies (scikit-learn, statsmodels, ruptures, scipy)
- ✅ Create config/analysis.yaml
- ✅ Create core/analysis_utils.py with full test coverage
- ⏳ Write test_analysis_utils.py (100% coverage)
- ⏳ Create test fixtures and synthetic data generators

### Phase 2.2: Analysis Plan Builder (Week 1-2)
- Create agents/agent_analysis_plan.py
- Implement intent mapping (question → agent_id)
- LLM fallback for ambiguous cases
- Write tests + golden files

### Phase 2.3: Agent Implementation (Week 2-4)

**Priority 1 (Week 2):**
- HypothesisTestingAgent (full implementation + tests)
- DriverAnalysisAgent (full implementation + tests)

**Priority 2 (Week 3):**
- SegmentationAgent (full implementation + tests)
- CohortAndRetentionAgent (full implementation + tests)

**Priority 3 (Week 4):**
- AttributionAndROASAgent (full implementation + tests)
- AnomalyAndTrendBreakAgent (full implementation + tests)

### Phase 2.4: Integration (Week 4-5)
- Update TriageAgent to return analysis_type
- Connect AnalysisPlanBuilder to SearchAgent
- Create notebook section "Analysis Mode"
- Write integration tests
- Performance benchmarking

### Phase 2.5: Documentation & Release (Week 5-6)
- Update README to v0.2.0
- Create TECHNICAL_SPEC_v0.2.0.md
- Update architecture diagrams
- Write acceptance test suite
- Version bump and release

---

## Code Structure

### File Organization

```
tasman-marketing-agent/
├── agents/
│   ├── agent_analysis_plan.py       # NEW: Analysis plan builder
│   ├── agent_hypothesis.py          # NEW: Hypothesis testing
│   ├── agent_driver.py               # NEW: Driver analysis
│   ├── agent_segmentation.py        # NEW: Segmentation
│   ├── agent_cohort.py              # NEW: Cohort & retention
│   ├── agent_attribution.py         # NEW: Attribution & ROAS
│   ├── agent_anomaly.py             # NEW: Anomaly detection
│   └── base_analysis_agent.py       # NEW: Base class
├── core/
│   ├── analysis_utils.py             # DONE: Statistical utilities
│   └── (Phase 1 files)
├── config/
│   ├── analysis.yaml                 # DONE: Analysis config
│   └── (Phase 1 configs)
├── tests/
│   ├── test_analysis_utils.py        # NEW
│   ├── test_analysis_plan_builder.py # NEW
│   ├── test_hypothesis_agent.py      # NEW
│   ├── test_driver_agent.py          # NEW
│   ├── test_segmentation_agent.py    # NEW
│   ├── test_cohort_agent.py          # NEW
│   ├── test_attribution_agent.py     # NEW
│   ├── test_anomaly_agent.py         # NEW
│   ├── fixtures/
│   │   ├── synthetic_data.py         # NEW
│   │   ├── golden_plans/             # NEW
│   │   └── expected_results/         # NEW
│   └── integration/
│       └── test_analysis_end_to_end.py # NEW
└── notebooks/
    └── Agentic_Analytics_Demo.ipynb  # UPDATED: Add Analysis Mode section
```

### Estimated Lines of Code

| Component | Estimated LOC | Status |
|-----------|---------------|--------|
| analysis_utils.py | 600 | ✅ Done |
| base_analysis_agent.py | 150 | Pending |
| agent_analysis_plan.py | 300 | Pending |
| agent_hypothesis.py | 400 | Pending |
| agent_driver.py | 500 | Pending |
| agent_segmentation.py | 450 | Pending |
| agent_cohort.py | 400 | Pending |
| agent_attribution.py | 350 | Pending |
| agent_anomaly.py | 450 | Pending |
| Test files (all) | 2000 | Pending |
| Notebook updates | 500 | Pending |
| Documentation | 1000 | Pending |
| **Total** | **~7,100** | **~8% done** |

---

## Data Flow

### Analysis Request Flow

```
1. User Question
   "What drives conversion on mobile?"

2. TriageAgent (Phase 1)
   → mode = "analysis"
   → analysis_type = "driver_analysis"

3. AnalysisPlanBuilder
   → Detect agent: driver_analysis
   → Extract entities:
      - target: converted_flag
      - filters: device=mobile
      - window: 90 days (default)
   → Build plan JSON

4. DriverAnalysisAgent
   a. plan() → Validate & enrich plan
   b. pull() → Execute SQL with sampling
      SQL: SELECT session_id, device, channel, pages_viewed, converted_flag
           FROM fact_sessions s
           LEFT JOIN dim_campaigns c USING(campaign_id)
           WHERE s.device = 'mobile'
             AND s.date >= CURRENT_DATE - INTERVAL 90 DAY

      Result: 180,000 rows → sample to 250,000 (no sampling needed)

   c. run() → Fit logistic regression
      - Encode categoricals (channel → one-hot)
      - Standardize numerics (pages_viewed)
      - 5-fold CV
      - Permutation importance

   d. report() → Format results
      - Top 10 drivers
      - Model AUC = 0.73
      - Charts: feature importance bar chart
      - Insights: "Channel has strongest effect on conversion (OR=2.3)"
      - Next actions: "Test removing low-performing channels"

5. Return to user
   {
     "agent_id": "driver_analysis",
     "plan": {...},
     "results": {
       "model_metrics": {"auc": 0.73, "cv_std": 0.02},
       "drivers": [
         {"feature": "channel_Google", "importance": 0.42, "p": 0.001},
         {"feature": "pages_viewed", "importance": 0.31, "p": 0.003},
         ...
       ]
     },
     "insights": [...],
     "chart_paths": ["driver_importance.png"]
   }
```

### SQL Query Patterns

**Aggregation (no row limit):**
```sql
-- Hypothesis testing: CTR by channel
SELECT channel, SUM(clicks), SUM(impressions)
FROM fact_ad_spend f
JOIN dim_campaigns c USING(campaign_id)
WHERE date >= CURRENT_DATE - INTERVAL 90 DAY
GROUP BY 1
-- No LIMIT needed (aggregated to ~5 rows)
```

**Row-level (with sampling):**
```sql
-- Driver analysis: session features
SELECT session_id, device, channel, pages_viewed, converted_flag
FROM fact_sessions s
LEFT JOIN dim_campaigns c USING(campaign_id)
WHERE date >= CURRENT_DATE - INTERVAL 90 DAY
-- Returns 500k rows → sample to 250k via StratifiedSampler
```

---

## Configuration

### analysis.yaml Key Sections

**Row Caps:**
```yaml
row_caps:
  modelling_max_rows: 250000
  hypothesis_max_rows: 100000
  segmentation_max_rows: 200000
```

**Statistical Parameters:**
```yaml
stats:
  confidence_level: 0.95
  alpha: 0.05
  fdr_rate: 0.1
  min_sample_size: 30
  bootstrap_iterations: 1000
  cv_folds: 5
```

**Agent Defaults:**
```yaml
hypothesis:
  default_test_type: "auto"
  multiple_comparison_correction: true

driver:
  default_model_type: "auto"
  max_features: 50
  include_interactions: false

segmentation:
  default_method: "kmeans"
  k_range: [2, 10]
  standardize: true
```

---

## Acceptance Criteria

### Functional Requirements

1. **Zero LLM calls for templated questions**
   - "Is CTR higher for LinkedIn vs Facebook?" → local plan
   - "What drives conversion?" → local plan
   - "Segment customers by value" → local plan

2. **Correct statistical outputs**
   - Hypothesis tests: p-values, CIs, effect sizes
   - Driver analysis: AUC > 0.65 on test set
   - Segmentation: Silhouette score > 0.3

3. **Sampling preserves distributions**
   - Stratified by channel/device/region
   - Sample ratio reported
   - Strata distribution tracked

4. **Reproducibility**
   - Same seed → same results
   - Plan JSON includes version + seed
   - SQL queries logged

### Performance Requirements

1. **Execution time < 3s for 250k rows**
   - Driver analysis with 50 features
   - K-means with k=5
   - Hypothesis test with 10 groups

2. **Memory usage < 4GB**
   - With 250k rows × 50 features

3. **Test suite < 30s**
   - All unit + integration tests

### Quality Requirements

1. **Test coverage**
   - Core utilities: 100%
   - Agent logic: 95%
   - Integration: 18+ tests

2. **Documentation**
   - Technical spec v0.2.0 (1000+ lines)
   - API reference for all agents
   - Notebook with 6 working examples

3. **Code quality**
   - Type hints on all public methods
   - Docstrings with examples
   - No pylint errors

---

## Next Steps

### Immediate (This Session)
1. ✅ Review Phase 2 spec
2. ✅ Add dependencies
3. ✅ Create analysis.yaml
4. ✅ Create analysis_utils.py
5. ⏳ Create this architecture doc

### Week 1
1. Write test_analysis_utils.py (100% coverage)
2. Create test fixtures (synthetic data generators)
3. Create base_analysis_agent.py
4. Implement AnalysisPlanBuilder
5. Write tests for AnalysisPlanBuilder

### Week 2-4
1. Implement 6 analysis agents (in priority order)
2. Write tests for each agent (unit + integration)
3. Create golden plan files
4. Performance benchmarking

### Week 5-6
1. Update notebook with Analysis Mode section
2. Write TECHNICAL_SPEC_v0.2.0.md
3. Update architecture diagrams
4. Run acceptance tests
5. Version bump and release

---

**End of Phase 2 Architecture & Implementation Plan**
