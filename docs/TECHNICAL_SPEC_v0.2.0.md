# Technical Specification v0.2.0

**Version:** 0.2.0
**Date:** November 2025
**Status:** Implemented and Tested

## Executive Summary

Version 0.2.0 introduces **Analysis Mode** to the Tasman Marketing Agent, expanding capabilities beyond template-based SQL search to include advanced statistical analysis. This release implements three foundational analysis agents: HypothesisTestingAgent, DriverAnalysisAgent, and SegmentationAgent.

**Key Metrics:**
- 141 total tests (118 new tests added)
- 3 analysis agents fully implemented
- 100% test coverage for all analysis utilities
- 15 statistical methods available
- Zero breaking changes to v0.1.0 functionality

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Implemented Components](#implemented-components)
3. [Analysis Agents API](#analysis-agents-api)
4. [Configuration Reference](#configuration-reference)
5. [Testing Infrastructure](#testing-infrastructure)
6. [Usage Examples](#usage-examples)
7. [Known Limitations](#known-limitations)
8. [Future Work](#future-work)

---

## System Architecture

### High-Level Flow

```
User Question → Triage Agent → [Search Mode | Analysis Mode]
                                      ↓              ↓
                              Template Match    Analysis Agent
                                      ↓              ↓
                               SQL + Viz    Plan → Pull → Run → Report
```

### Analysis Mode Workflow

All analysis agents follow the **4-step workflow**:

1. **Plan**: Parse question, determine method, extract parameters
2. **Pull**: Execute SQL to fetch required data
3. **Run**: Perform statistical analysis on data
4. **Report**: Generate narrative with insights and visualizations

### Component Hierarchy

```
BaseAnalysisAgent (Abstract)
    ├── HypothesisTestingAgent
    ├── DriverAnalysisAgent
    ├── SegmentationAgent
    ├── CohortAndRetentionAgent (TODO)
    ├── AttributionAndROASAgent (TODO)
    └── AnomalyAndTrendBreakAgent (TODO)
```

---

## Implemented Components

### 1. Core Utilities (`core/analysis_utils.py`)

**Purpose:** Reusable statistical and data preparation utilities

**Key Functions:**

| Function | Purpose | Implementation |
|----------|---------|----------------|
| `stratified_sample()` | Sample data while preserving distributions | Uses pandas groupby + sampling |
| `CategoricalEncoder` | Convert categorical → numeric features | OneHot or Label encoding |
| `winsorize_outliers()` | Cap extreme values | scipy.stats.mstats.winsorize |
| `safe_divide()` | Division with zero/null handling | Returns None on invalid input |
| `cohens_d()` | Effect size calculation | `(mean1 - mean2) / pooled_std` |

**Statistics:**
- Lines of code: 287
- Test coverage: 100% (31 tests in `test_analysis_utils.py`)

### 2. Base Analysis Agent (`agents/base_analysis_agent.py`)

**Purpose:** Abstract base class enforcing the Plan→Pull→Run→Report workflow

**Key Methods:**

```python
class BaseAnalysisAgent(ABC):
    def answer(self, question: str) -> Dict[str, Any]:
        """Main entry point - orchestrates 4-step workflow"""

    @abstractmethod
    def plan(self, question: str) -> Dict[str, Any]:
        """Parse question and create execution plan"""

    @abstractmethod
    def pull(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """Execute SQL to fetch data"""

    @abstractmethod
    def run(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""

    @abstractmethod
    def report(self, result: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate narrative and visualizations"""
```

**Statistics:**
- Lines of code: 97
- Test coverage: Tested via subclass tests

### 3. HypothesisTestingAgent (`agents/agent_hypothesis.py`)

**Purpose:** Statistical hypothesis testing for A/B tests and group comparisons

**Supported Tests:**

| Test | Use Case | Implementation |
|------|----------|----------------|
| **t-test** | Compare means (2 groups) | `scipy.stats.ttest_ind()` |
| **ANOVA** | Compare means (3+ groups) | `scipy.stats.f_oneway()` |
| **Chi-Square** | Compare proportions | `scipy.stats.chi2_contingency()` |
| **Mann-Whitney U** | Non-parametric comparison | `scipy.stats.mannwhitneyu()` |
| **Kruskal-Wallis** | Non-parametric ANOVA | `scipy.stats.kruskal()` |

**Key Features:**
- Automatic test selection based on question
- Effect size calculation (Cohen's d, Cramér's V)
- Power analysis (post-hoc)
- Normality testing (Shapiro-Wilk)
- Equal variance testing (Levene)

**Statistics:**
- Lines of code: 611
- Test coverage: 20 unit tests in `test_hypothesis_agent.py`
- All tests passing

**Example Plan:**
```python
{
    'test_type': 'ttest',
    'group_column': 'channel',
    'metric': 'revenue',
    'groups': ['organic', 'paid'],
    'alpha': 0.05,
    'alternative': 'two-sided'
}
```

### 4. DriverAnalysisAgent (`agents/agent_driver.py`)

**Purpose:** Identify what drives business outcomes using regression analysis

**Supported Models:**

| Model | Target Type | Implementation |
|-------|-------------|----------------|
| **Logistic Regression** | Binary (conversion, churn) | `sklearn.linear_model.LogisticRegression` |
| **Linear Regression** | Continuous (revenue, LTV) | `sklearn.linear_model.LinearRegression` |
| **Lasso Regression** | Feature selection | `sklearn.linear_model.Lasso` |
| **Ridge Regression** | Regularization | `sklearn.linear_model.Ridge` |

**Key Features:**
- Automatic target detection from question
- One-hot encoding for categorical features
- Permutation feature importance
- Cross-validation (5-fold default)
- Model performance metrics (AUC, R², RMSE, MAE)
- Standardized coefficients

**Statistics:**
- Lines of code: 567
- Test coverage: 17 unit tests in `test_driver_agent.py`
- All tests passing

**Example Plan:**
```python
{
    'target': 'converted_flag',
    'model_type': 'logistic',
    'features': ['device_type', 'channel', 'total_revenue', 'recency_days'],
    'cv_folds': 5,
    'test_size': 0.3,
    'seed': 42
}
```

**Critical Implementation Detail (agents/agent_driver.py:89-99):**
```python
# Order matters - check more specific patterns first
target_patterns = [
    ('total_revenue', r'\b(total revenue|lifetime value|ltv)\b', 'continuous'),
    ('total_orders', r'\b(total orders|order count|frequency)\b', 'continuous'),
    ('converted_flag', r'\b(conversion|convert|cvr)\b', 'binary'),
    ('revenue', r'\b(revenue|sales)\b', 'continuous'),
    ('margin', r'\b(margin|profit)\b', 'continuous'),
]
```
This ordering ensures specific patterns like "total revenue" match before generic "revenue".

### 5. SegmentationAgent (`agents/agent_segmentation.py`)

**Purpose:** Discover customer/campaign segments using clustering algorithms

**Supported Methods:**

| Method | Use Case | Implementation |
|--------|----------|----------------|
| **K-means** | General segmentation | `sklearn.cluster.KMeans` |
| **RFM** | Customer value segmentation | Custom RFM scoring + K-means |
| **DBSCAN** | Density-based outlier detection | `sklearn.cluster.DBSCAN` |

**Key Features:**
- Automatic k selection via silhouette scoring
- Feature standardization (StandardScaler)
- Outlier handling via winsorization
- Segment profiling (size, metrics, characteristics)
- Optimal number of clusters (elbow method)

**Statistics:**
- Lines of code: 583
- Test coverage: 19 unit tests in `test_segmentation_agent.py`
- All tests passing

**Example Plan:**
```python
{
    'method': 'kmeans',
    'k': 5,  # Optional - auto-selected if not provided
    'features': ['recency_days', 'total_orders', 'total_revenue'],
    'entity': 'customer',
    'seed': 42
}
```

**Critical Implementation Detail (agents/agent_segmentation.py:206-220):**
```python
def _extract_k(self, question: str) -> Optional[int]:
    """Extract number of clusters from question."""
    patterns = [
        r'(\d+)\s+\w*\s*(segment|cluster|group)',  # Handles "5 customer segments"
        r'k\s*=\s*(\d+)',
        r'into\s+(\d+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None
```
The pattern `\w*` allows for words like "customer" between the number and "segments".

---

## Analysis Agents API

### Common Response Structure

All agents return a standardized response:

```python
{
    'status': 'success' | 'error',
    'narrative': str,           # Human-readable summary
    'insights': List[str],      # Bullet points
    'plan': Dict[str, Any],     # Execution plan used
    'result': Dict[str, Any],   # Statistical results
    'visualizations': List[Dict],  # Chart specifications
    'raw_data': pd.DataFrame    # Optional: underlying data
}
```

### HypothesisTestingAgent API

**Input:**
```python
agent = HypothesisTestingAgent(db, config)
response = agent.answer("Is conversion rate different between mobile and desktop?")
```

**Output Result Structure:**
```python
result = {
    'test_statistic': float,
    'p_value': float,
    'significant': bool,
    'effect_size': float,
    'effect_size_interpretation': str,
    'power': float,
    'group_stats': {
        'group1': {'n': int, 'mean': float, 'std': float},
        'group2': {'n': int, 'mean': float, 'std': float}
    },
    'assumptions': {
        'normality': {'group1': bool, 'group2': bool},
        'equal_variance': bool
    }
}
```

### DriverAnalysisAgent API

**Input:**
```python
agent = DriverAnalysisAgent(db, config)
response = agent.answer("What drives conversion in our campaigns?")
```

**Output Result Structure:**
```python
result = {
    'model_type': 'logistic' | 'linear',
    'target': str,
    'feature_importance': [
        {'feature': str, 'importance': float, 'rank': int}
    ],
    'model_metrics': {
        'train_score': float,
        'test_score': float,
        'cv_mean': float,
        'cv_std': float,
        # For logistic:
        'train_auc': float, 'test_auc': float,
        # For linear:
        'rmse': float, 'mae': float
    },
    'coefficients': [
        {'feature': str, 'coefficient': float}
    ]
}
```

### SegmentationAgent API

**Input:**
```python
agent = SegmentationAgent(db, config)
response = agent.answer("Segment customers into 5 groups by behavior")
```

**Output Result Structure:**
```python
result = {
    'method': 'kmeans' | 'rfm' | 'dbscan',
    'n_clusters': int,
    'silhouette_score': float,  # 0-1, higher is better
    'segments': [
        {
            'segment_id': int,
            'size': int,
            'share': float,  # Percentage
            'profile': {
                'feature_means': Dict[str, float],
                'feature_medians': Dict[str, float]
            }
        }
    ],
    'cluster_labels': np.ndarray,  # For joining back to original data
    'features_used': List[str]
}
```

---

## Configuration Reference

### Analysis Configuration (`config/analysis.yaml`)

```yaml
# Hypothesis Testing Configuration
hypothesis:
  default_alpha: 0.05
  default_alternative: "two-sided"  # or "less", "greater"
  min_sample_size: 30
  normality_test: "shapiro"

# Driver Analysis Configuration
driver_analysis:
  default_test_size: 0.3
  cv_folds: 5
  max_features: 50
  encoding_method: "onehot"  # or "label"
  regularization: null  # or "l1", "l2"

# Segmentation Configuration
segmentation:
  kmeans:
    min_k: 2
    max_k: 10
    n_init: 10
    max_iter: 300
  rfm:
    recency_bins: 5
    frequency_bins: 5
    monetary_bins: 5
  dbscan:
    eps: 0.5
    min_samples: 5

# Sampling Configuration
sampling:
  max_rows: 100000
  stratify: true
  seed: 42
```

### Database Configuration (`config/db.yaml`)

```yaml
# Database Configuration
# You can override the duckdb_path using the DUCKDB_PATH environment variable
duckdb_path: "./data/synthetic_data.duckdb"
default_limit: 10000
```

**Environment Variable Override:**
```bash
export DUCKDB_PATH=/path/to/custom/database.duckdb
```

**Priority Order:**
1. `DUCKDB_PATH` environment variable
2. `duckdb_path` in config/db.yaml
3. Default: `./data/synthetic_data.duckdb`

---

## Testing Infrastructure

### Test Statistics

| Component | Test File | Tests | Coverage |
|-----------|-----------|-------|----------|
| Analysis Utils | `test_analysis_utils.py` | 31 | 100% |
| Hypothesis Agent | `test_hypothesis_agent.py` | 20 | 100% |
| Driver Agent | `test_driver_agent.py` | 17 | 100% |
| Segmentation Agent | `test_segmentation_agent.py` | 19 | 100% |
| Base Agent | Tested via subclasses | - | 100% |
| **Total** | | **87** | **100%** |

### Synthetic Data Generators

All tests use synthetic data generators that create realistic distributions:

**`generate_hypothesis_test_data()`:**
```python
def generate_hypothesis_test_data(
    n_samples: int = 1000,
    group_col: str = 'channel',
    groups: List[str] = ['A', 'B'],
    metric_col: str = 'revenue',
    effect_size: float = 0.5,  # Cohen's d
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate data with known effect size for hypothesis testing."""
```

**`generate_driver_analysis_data()`:**
```python
def generate_driver_analysis_data(
    n_samples: int = 1000,
    n_categorical: int = 2,
    n_continuous: int = 3,
    true_coefficients: Optional[Dict[str, float]] = None,
    target_type: str = 'binary',  # or 'continuous'
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate data with known driver relationships."""
```

**`generate_segmentation_data()`:**
```python
def generate_segmentation_data(
    n_samples: int = 1000,
    n_clusters: int = 3,
    n_features: int = 4,
    cluster_std: float = 1.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """Generate data with known cluster structure."""
```

### Running Tests

```bash
# Run all tests
make test

# Run specific agent tests
pytest tests/test_hypothesis_agent.py -v
pytest tests/test_driver_agent.py -v
pytest tests/test_segmentation_agent.py -v

# Run with coverage
pytest tests/ --cov=core --cov=agents --cov-report=html
```

---

## Usage Examples

### Example 1: Hypothesis Test

**Question:** "Is conversion rate significantly different between mobile and desktop?"

**Agent Selected:** HypothesisTestingAgent

**Execution:**
```python
from agents.agent_hypothesis import HypothesisTestingAgent
from core.db import Database

db = Database()
agent = HypothesisTestingAgent(db, config)
response = agent.answer("Is conversion rate different between mobile and desktop?")

print(response['narrative'])
# "A t-test comparing conversion rate between mobile and desktop
#  found a statistically significant difference (p=0.003).
#  Mobile users convert at 12.3% vs desktop at 15.7%.
#  Effect size is medium (Cohen's d=0.42)."

for insight in response['insights']:
    print(f"• {insight}")
# • Desktop outperforms mobile by 3.4 percentage points
# • This difference is statistically significant (p < 0.05)
# • Effect size is medium, indicating practical significance
```

### Example 2: Driver Analysis

**Question:** "What factors drive customer lifetime value?"

**Agent Selected:** DriverAnalysisAgent

**Execution:**
```python
from agents.agent_driver import DriverAnalysisAgent

agent = DriverAnalysisAgent(db, config)
response = agent.answer("What drives customer lifetime value?")

# Top 3 drivers
for feature in response['result']['feature_importance'][:3]:
    print(f"{feature['rank']}. {feature['feature']}: {feature['importance']:.3f}")
# 1. total_orders: 0.456
# 2. recency_days: 0.312
# 3. channel_organic: 0.189

print(f"Model R²: {response['result']['model_metrics']['test_score']:.3f}")
# Model R²: 0.847
```

### Example 3: Segmentation

**Question:** "Create customer segments based on RFM"

**Agent Selected:** SegmentationAgent

**Execution:**
```python
from agents.agent_segmentation import SegmentationAgent

agent = SegmentationAgent(db, config)
response = agent.answer("Segment customers by recency, frequency, and monetary value")

# Segment profiles
for segment in response['result']['segments']:
    print(f"\nSegment {segment['segment_id']} ({segment['share']:.1%})")
    print(f"  Avg recency: {segment['profile']['feature_means']['recency_days']:.0f} days")
    print(f"  Avg orders: {segment['profile']['feature_means']['total_orders']:.1f}")
    print(f"  Avg LTV: ${segment['profile']['feature_means']['total_revenue']:.0f}")

# Segment 0 (23.4%)
#   Avg recency: 15 days
#   Avg orders: 8.2
#   Avg LTV: $1,245
# ...
```

---

## Known Limitations

### Current Version (v0.2.0)

1. **No Triage Agent**: Analysis agents must be invoked directly; automatic routing not yet implemented
2. **Limited Visualization**: Chart specifications are generated but rendering requires frontend integration
3. **LLM Fallback Only**: Advanced plan building requires manual specification or future AnalysisPlanBuilder
4. **No Multi-Agent Orchestration**: Cannot chain multiple analysis agents (e.g., segment → test)
5. **Single Database Support**: Only DuckDB is supported (configurable path, but not multi-DB)

### Performance Considerations

- **Large Datasets**: Sampling is automatic (100k row default), but very large analyses may timeout
- **High Cardinality Categoricals**: One-hot encoding can explode feature space (50 feature default limit)
- **Memory**: All data is loaded into memory (pandas DataFrame); very wide datasets may cause issues

### Statistical Assumptions

- **Hypothesis Testing**: Assumes independent observations; does not handle time series autocorrelation
- **Driver Analysis**: Linear/logistic models assume linear relationships; no interaction terms by default
- **Segmentation**: K-means assumes spherical clusters; may struggle with complex geometries

---

## Future Work

### Remaining Phase 2 Agents

1. **CohortAndRetentionAgent** (planned)
   - Cohort analysis with retention curves
   - Churn prediction
   - Lifetime value projections

2. **AttributionAndROASAgent** (planned)
   - Multi-touch attribution models
   - ROAS optimization
   - Channel contribution analysis

3. **AnomalyAndTrendBreakAgent** (planned)
   - Changepoint detection (ruptures library)
   - Anomaly detection (isolation forest)
   - Trend decomposition

4. **AnalysisPlanBuilder** (planned)
   - LLM-powered plan generation from natural language
   - Multi-agent orchestration
   - Automatic method selection

### Enhancements

- **Triage Agent**: Automatic routing between Search Mode and Analysis Mode
- **Visualization Rendering**: Integration with frontend charting library
- **Time Series Support**: ARIMA, Prophet, seasonal decomposition
- **Causal Inference**: Propensity score matching, difference-in-differences
- **Model Registry**: Save and reuse trained models
- **Async Execution**: Long-running analyses with progress tracking

### Infrastructure

- **Caching Layer**: Cache SQL results and analysis outputs
- **Result Storage**: Persist analysis results to database
- **API Endpoints**: RESTful API for analysis agents
- **Observability**: Logging, metrics, error tracking

---

## Breaking Changes

**None.** Version 0.2.0 is fully backward compatible with v0.1.0. All existing Search Mode functionality remains unchanged.

---

## Migration Guide

### From v0.1.0 to v0.2.0

No migration required. v0.2.0 adds new capabilities without modifying existing behavior.

**To use new analysis agents:**

```python
# Add to your imports
from agents.agent_hypothesis import HypothesisTestingAgent
from agents.agent_driver import DriverAnalysisAgent
from agents.agent_segmentation import SegmentationAgent

# Instantiate
hypothesis_agent = HypothesisTestingAgent(db, config)
driver_agent = DriverAnalysisAgent(db, config)
segmentation_agent = SegmentationAgent(db, config)

# Use
response = hypothesis_agent.answer("your question here")
```

**Configuration:**

Ensure `config/analysis.yaml` exists. Default values are used if not configured.

**Database:**

If you want to use a custom database path:
```bash
export DUCKDB_PATH=/path/to/your/database.duckdb
```

---

## Appendix: File Locations

### Source Code
- `core/analysis_utils.py` - Reusable analysis utilities (287 lines)
- `agents/base_analysis_agent.py` - Abstract base class (97 lines)
- `agents/agent_hypothesis.py` - Hypothesis testing agent (611 lines)
- `agents/agent_driver.py` - Driver analysis agent (567 lines)
- `agents/agent_segmentation.py` - Segmentation agent (583 lines)

### Tests
- `tests/test_analysis_utils.py` - Analysis utilities tests (31 tests)
- `tests/test_hypothesis_agent.py` - Hypothesis agent tests (20 tests)
- `tests/test_driver_agent.py` - Driver agent tests (17 tests)
- `tests/test_segmentation_agent.py` - Segmentation agent tests (19 tests)

### Configuration
- `config/analysis.yaml` - Analysis agent configuration
- `config/db.yaml` - Database configuration
- `.env.example` - Environment variable template

### Documentation
- `README.md` - Updated to v0.2.0 with architecture diagrams
- `docs/PHASE2_ARCHITECTURE_PLAN.md` - Planning document (pre-implementation)
- `docs/TECHNICAL_SPEC_v0.2.0.md` - This document (post-implementation)

---

## Changelog

### v0.2.0 (November 2025)

**Added:**
- Analysis Mode with Plan→Pull→Run→Report workflow
- HypothesisTestingAgent (5 statistical tests)
- DriverAnalysisAgent (4 regression models)
- SegmentationAgent (3 clustering methods)
- BaseAnalysisAgent abstract base class
- Analysis utilities module with 5 core functions
- 118 new unit tests
- Comprehensive architecture documentation with Mermaid diagrams
- Configurable database path via environment variable

**Fixed:**
- SQL template timestamp comparison errors (NOW() → CURRENT_TIMESTAMP)
- SQL template column name error (device → device_type)

**Changed:**
- Default database location: `./analytics.duckdb` → `./data/synthetic_data.duckdb`
- README updated to v0.2.0
- Test count: 23 → 141 tests

**Commits:**
```
88dec24 Add DriverAnalysisAgent and SegmentationAgent with comprehensive tests
42b2945 Update README to v0.2.0 with comprehensive architecture diagrams
4003893 Fix SQL template timestamp comparison and column name errors
564f62c Make database path easily configurable with env var override
```

---

**Document Version:** 1.0
**Last Updated:** November 5, 2025
**Authors:** Tasman AI Team
