# Tasman Agentic Analytics - Technical Specification v0.1.0

**Version:** 0.1.0
**Status:** Production-ready for search mode
**Date:** 2025-11-03

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Agents](#agents)
5. [Configuration](#configuration)
6. [Data Layer](#data-layer)
7. [Testing](#testing)
8. [API Reference](#api-reference)
9. [Performance Characteristics](#performance-characteristics)
10. [Known Limitations](#known-limitations)

---

## System Overview

### Purpose
A **local-first, notebook-driven agentic analytics system** that minimizes LLM usage through intelligent template matching and rule-based triage. Converts natural language questions into SQL queries, executes them safely, and visualizes results.

### Design Philosophy
- **Local-first**: Template matching and rule-based logic before any LLM calls
- **Minimal LLM usage**: Only calls LLM when local logic fails; aggressive filesystem caching
- **Config-driven**: Schema, business context, and SQL templates in YAML/JSON
- **Safety-first**: Read-only SQL execution with LIMIT enforcement
- **Full observability**: Track every step from triage → SQL → execution → visualization

### Technology Stack
- **Language**: Python 3.11+
- **Package Manager**: uv (Astral)
- **Database**: DuckDB (in-process analytical DB)
- **LLM Providers**: OpenAI GPT-4o-mini / Anthropic Claude 3.5 Haiku
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest
- **Notebook**: Jupyter

---

## Architecture

### Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      USER QUESTION                          │
│            "show ad spend per channel over time"            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               SearchAgent (Orchestrator)                     │
│   Manages: Triage → SQL Gen → Execute → Viz → Summarize    │
└─────────────────────────────────────────────────────────────┘
          ↓                  ↓                   ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  TriageAgent     │ │ TextToSQLAgent   │ │    AutoViz       │
│  • Search/Anal   │ │ • Template match │ │ • Chart detect   │
│  • Role infer    │ │ • SQL generate   │ │ • Visualize      │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          ↓                  ↓                   ↓
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  LocalTriage     │ │ LocalTextToSQL   │ │ DuckDBConnector  │
│  • Keyword match │ │ • Fuzzy match    │ │ • Read-only exec │
│  • No LLM ✓      │ │ • 6 templates ✓  │ │ • LIMIT enforce  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          ↓                  ↓
┌─────────────────────────────────────────────────────────────┐
│            LLM Fallback (Optional - Red Dashed)             │
│  • Only when confidence < 0.6 or no template match          │
│  • Filesystem cache prevents repeat calls                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Configuration & Data Layer                      │
│  • schema.json • sql_templates.yaml • marketing.duckdb      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                         OUTPUT                               │
│  SQL Query | DataFrame | Chart (PNG) | NL Summary           │
└─────────────────────────────────────────────────────────────┘
```

### Query Flow (5 Steps)

1. **Triage** (agents/agent_triage.py)
   - Input: Question + Role
   - Process: Rule-based keyword matching → LLM fallback if confidence < 0.6
   - Output: mode (search/analysis), confidence, inferred_role

2. **SQL Generation** (agents/agent_text_to_sql.py)
   - Input: Question + Role
   - Process: Fuzzy template matching → LLM fallback if no match
   - Output: SQL query, confidence, method (template_match/llm)

3. **Execution** (core/duckdb_connector.py)
   - Input: SQL query
   - Process: Validate (SELECT only, has LIMIT) → Execute read-only
   - Output: pandas DataFrame

4. **Visualization** (core/viz.py)
   - Input: DataFrame
   - Process: Auto-detect chart type (bar/line/scatter/heatmap)
   - Output: PNG chart + chart_type

5. **Summarization** (core/viz.py)
   - Input: DataFrame + chart_path
   - Process: Generate natural language summary
   - Output: Text summary

---

## Core Components

### 1. DuckDBConnector (core/duckdb_connector.py)

**Purpose:** Read-only database interface with safety guardrails

**Key Methods:**
```python
def connect() -> None
def execute(sql: str, enforce_limit: bool = True,
            enforce_select_only: bool = True) -> pd.DataFrame
def list_tables() -> List[str]
def validate_schema(schema: Dict) -> Tuple[bool, List[str]]
def close() -> None
```

**Safety Features:**
- Read-only mode (blocks DDL/DML: CREATE, DROP, INSERT, UPDATE, DELETE, ALTER)
- LIMIT enforcement (adds LIMIT if missing, configurable default)
- SQL validation (must start with SELECT)
- Exception handling with detailed error messages

**Configuration:**
```yaml
# config/db.yaml
database_path: "./data/marketing.duckdb"
default_limit: 1000
read_only: true
```

**Testing:** 9 tests in `tests/test_duckdb_connector.py`

---

### 2. LocalTriage (core/triage_local.py)

**Purpose:** Rule-based query classification (search vs analysis)

**Algorithm:**
```python
SEARCH_KEYWORDS = [
    "what", "how many", "how much", "when", "show", "list",
    "plot", "display", "get", "find", "count", "sum",
    "average", "total", "by", "per", "top", "bottom"
]

ANALYSIS_KEYWORDS = [
    "why", "driver", "drivers", "impact", "causal", "cause",
    "segment", "segmentation", "cluster", "cohort",
    "correlation", "trend", "pattern", "anomaly"
]

# Scoring logic
search_score = count_keywords(question, SEARCH_KEYWORDS) * 0.1
analysis_score = count_keywords(question, ANALYSIS_KEYWORDS) * 0.15

if search_score > analysis_score:
    mode = "search"
    confidence = min(0.6 + search_score, 0.9)
else:
    mode = "analysis"
    confidence = min(0.6 + analysis_score, 0.9)
```

**Outputs:**
- `mode`: "search" or "analysis"
- `confidence`: 0.0 - 1.0
- `inferred_role`: Detected from question content
- `reason`: Human-readable explanation

**Testing:** 5 tests in `tests/test_triage_local.py`

---

### 3. LocalTextToSQL (core/local_text_to_sql.py)

**Purpose:** Template-based SQL generation via fuzzy matching

**Algorithm:**
```python
from difflib import SequenceMatcher

def match_template(question: str, templates: List[Dict]) -> Dict:
    best_match = None
    best_score = 0.0

    for template in templates:
        for utterance in template['utterances']:
            # Fuzzy string matching
            ratio = SequenceMatcher(None,
                                   question.lower(),
                                   utterance.lower()).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = template

    if best_score >= 0.6:  # Threshold
        return fill_template_params(best_match, question)
    else:
        return {"method": "no_match", "confidence": 0.0}
```

**Template Structure:**
```yaml
# config/sql_templates.yaml
- id: spend_by_channel_over_time
  utterances:
    - "spend by channel"
    - "show ad spend per channel over time"
    - "channel spend over time"
  role_hint: marketer
  sql: |
    SELECT c.channel,
           f.date,
           SUM(f.spend) AS total_spend
    FROM fact_ad_spend f
    JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
    WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
    GROUP BY 1,2
    ORDER BY 2 ASC, 1
    LIMIT {{limit}};
  dims: [channel, date]
  measures: [spend]
```

**Parameter Filling:**
- `{{time_window_days}}`: From role defaults (e.g., 90 for marketer)
- `{{limit}}`: From config (default: 1000)

**Validation:**
- SQL must start with SELECT
- SQL must contain LIMIT clause
- Referenced tables must exist in schema

**Testing:** 6 tests in `tests/test_local_text_to_sql.py`

---

### 4. LLMClient (core/llm_clients.py)

**Purpose:** OpenAI/Anthropic client with filesystem caching

**Providers:**
- OpenAI: `gpt-4o-mini` (default)
- Anthropic: `claude-3-5-haiku-20241022`

**Caching Strategy:**
```python
import hashlib
import json

def get_cache_key(provider: str, prompt: str, model: str) -> str:
    key_string = f"{provider}:{model}:{prompt}"
    return hashlib.sha256(key_string.encode()).hexdigest()

def cached_call(prompt: str) -> str:
    cache_key = get_cache_key(self.provider, prompt, self.model)
    cache_file = self.cache_dir / f"{cache_key}.json"

    # Check cache
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            return cached['response']

    # Call LLM
    response = self._call_llm(prompt)

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump({
            'key_parts': [self.provider, self.model, prompt],
            'response': response,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    return response
```

**Configuration:**
```bash
# .env
MODEL_PROVIDER=openai           # or anthropic
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

**Cache Location:** `.cache/llm/*.json`

---

### 5. AutoViz (core/viz.py)

**Purpose:** Automatic chart type detection and visualization

**Chart Type Logic:**
```python
def detect_chart_type(df: pd.DataFrame) -> str:
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    date_cols = df.select_dtypes(include=['datetime']).columns

    # Line chart: date + 1-2 numeric
    if len(date_cols) >= 1 and 1 <= len(num_cols) <= 2:
        return "line"

    # Bar chart: 1 categorical + 1 numeric
    if len(cat_cols) == 1 and len(num_cols) == 1:
        return "bar"

    # Grouped bar: 1 categorical + 2+ numeric
    if len(cat_cols) == 1 and len(num_cols) >= 2:
        return "grouped_bar"

    # Scatter: 2 numeric
    if len(num_cols) == 2 and len(cat_cols) == 0:
        return "scatter"

    # Heatmap: 2 categorical + 1 numeric
    if len(cat_cols) == 2 and len(num_cols) == 1:
        return "heatmap"

    # Default: table
    return "table"
```

**Supported Chart Types:**
- Line chart (time series)
- Bar chart (categorical comparison)
- Grouped bar chart (multi-metric comparison)
- Scatter plot (correlation)
- Heatmap (2D categorical)
- Table (fallback)

**Output:** PNG files saved to `notebooks/outputs/`

---

## Agents

### 1. TriageAgent (agents/agent_triage.py)

**Role:** Orchestrate query classification

**Logic:**
```python
def triage(question: str, role: str = None,
           force_local: bool = False) -> Dict:
    # Try local first
    local_result = self.local_triage.classify(question, role)

    # If confidence high enough or force_local, return
    if local_result['confidence'] >= self.llm_threshold or force_local:
        return local_result

    # Otherwise, call LLM
    if self.llm_client:
        llm_result = self.llm_client.classify_query(question, role)
        return llm_result

    # No LLM available, return local result
    return local_result
```

**Configuration:**
- `llm_threshold`: 0.6 (default)
- Falls back to LLM only when local confidence < 0.6

---

### 2. TextToSQLAgent (agents/agent_text_to_sql.py)

**Role:** Orchestrate SQL generation

**Logic:**
```python
def generate_sql(question: str, role: str = None,
                 force_local: bool = False) -> Dict:
    # Try template matching first
    local_result = self.local_text_to_sql.generate(question, role)

    # If template matched or force_local, return
    if local_result['method'] == 'template_match' or force_local:
        return local_result

    # Otherwise, call LLM
    if self.llm_client:
        llm_result = self.llm_client.generate_sql(
            question, role, self.schema, self.business_context
        )
        return llm_result

    # No LLM available, return local result (will be invalid)
    return local_result
```

**Configuration:**
- `llm_threshold`: 0.6 (default)
- `default_limit`: 1000

---

### 3. SearchAgent (agents/agent_search.py)

**Role:** End-to-end orchestration of search queries

**Workflow:**
```python
def search(question: str, role: str = None,
           force_local: bool = False) -> Dict:
    result = {'question': question, 'role': role, 'steps': []}

    # Step 1: Triage
    triage_result = self.triage_agent.triage(question, role, force_local)
    result['steps'].append({'step': 'triage', ...})

    # Only proceed if mode == "search"
    if triage_result['mode'] != 'search':
        return {'status': 'unsupported_mode', ...}

    # Step 2: Generate SQL
    sql_result = self.text_to_sql_agent.generate_sql(question, role, force_local)
    result['steps'].append({'step': 'sql_generation', ...})

    # Step 3: Execute SQL
    df = self.db_connector.execute(sql_result['sql'], enforce_limit=False)
    result['steps'].append({'step': 'execution', ...})

    # Step 4: Visualize
    chart_path, chart_type = self.viz.visualize(df, "chart.png")
    result['steps'].append({'step': 'visualization', ...})

    # Step 5: Summarize
    summary = self.viz.summarize_result(df, chart_path)
    result['summary'] = summary

    return result
```

**Output Schema:**
```python
{
    'question': str,
    'role': str,
    'status': str,  # success | unsupported_mode | sql_generation_failed | execution_failed
    'triage': {...},
    'sql_generation': {...},
    'sql': str,
    'data': pd.DataFrame,
    'row_count': int,
    'chart_path': str,
    'chart_type': str,
    'summary': str,
    'steps': [
        {'step': 'triage', 'confidence': 0.7, 'used_llm': False, ...},
        {'step': 'sql_generation', 'method': 'template_match', 'used_llm': False, ...},
        {'step': 'execution', 'row_count': 245, 'success': True},
        {'step': 'visualization', 'chart_type': 'line', 'success': True},
    ]
}
```

---

## Configuration

### 1. Database Schema (config/schema.json)

**8 Tables:**

**Dimension Tables (5):**
```json
{
  "dim_campaigns": {
    "campaign_id": "INTEGER",
    "campaign_name": "TEXT",
    "channel": "TEXT",
    "audience": "TEXT",
    "placement": "TEXT"
  },
  "dim_adgroups": {
    "adgroup_id": "INTEGER",
    "adgroup_name": "TEXT",
    "campaign_id": "INTEGER"
  },
  "dim_creatives": {
    "creative_id": "INTEGER",
    "creative_name": "TEXT",
    "adgroup_id": "INTEGER"
  },
  "dim_products": {
    "sku": "TEXT",
    "product_name": "TEXT",
    "category": "TEXT",
    "subcategory": "TEXT",
    "brand": "TEXT",
    "unit_cost": "FLOAT",
    "unit_price": "FLOAT"
  },
  "dim_customers": {
    "customer_id": "TEXT",
    "region": "TEXT",
    "segment": "TEXT",
    "cohort": "TEXT"
  }
}
```

**Fact Tables (3):**
```json
{
  "fact_ad_spend": {
    "date": "DATE",
    "campaign_id": "INTEGER",
    "impressions": "INTEGER",
    "clicks": "INTEGER",
    "spend": "FLOAT"
  },
  "fact_sessions": {
    "session_id": "TEXT",
    "date": "DATE",
    "campaign_id": "INTEGER",
    "device": "TEXT",
    "converted_flag": "BOOLEAN"
  },
  "fact_orders": {
    "order_id": "TEXT",
    "order_timestamp": "TIMESTAMP",
    "session_id": "TEXT",
    "customer_id": "TEXT",
    "sku": "TEXT",
    "quantity": "INTEGER",
    "revenue": "FLOAT",
    "cogs": "FLOAT",
    "margin": "FLOAT"
  }
}
```

---

### 2. Business Context (config/business_context.yaml)

**4 Roles:**

```yaml
roles:
  marketer:
    kpis: [spend, impressions, clicks, ctr, cvr, cac, roas, revenue, margin]
    dims: [channel, campaign_name, audience, placement, device, date]
    synonyms:
      spend: [cost, budget, investment]
      clicks: [click]
      impressions: [impression, views]
    defaults:
      time_window_days: 90
      limit: 1000

  ceo:
    kpis: [revenue, margin, growth, cac, ltv, roas]
    dims: [channel, region, brand, category, date]
    synonyms:
      revenue: [sales, takings, income]
      margin: [profit, profitability]
    defaults:
      time_window_days: 180
      limit: 1000

  cpo:
    kpis: [orders, revenue, margin, attach_rate, conversion_rate]
    dims: [sku, category, subcategory, brand, region]
    synonyms:
      orders: [purchases, transactions]
      conversion_rate: [cvr, conversion]
    defaults:
      time_window_days: 120
      limit: 1000

  coo:
    kpis: [orders, fulfillment_rate, cancellations, margin, revenue]
    dims: [region, category, device]
    synonyms:
      fulfillment_rate: [fill_rate, delivery_rate]
    defaults:
      time_window_days: 120
      limit: 1000

calculated_measures:
  ctr:
    formula: "clicks / impressions"
    type: "ratio"
  cvr:
    formula: "conversions / sessions"
    type: "ratio"
  cac:
    formula: "spend / conversions"
    type: "ratio"
  roas:
    formula: "revenue / spend"
    type: "ratio"

defaults:
  time_window_days: 90
  limit: 1000
```

---

### 3. SQL Templates (config/sql_templates.yaml)

**6 Templates:**

1. **spend_by_channel_over_time** (marketer)
2. **ctr_by_campaign** (marketer)
3. **orders_and_revenue_by_category** (ceo)
4. **roas_by_channel** (ceo)
5. **device_sessions_and_cvr** (marketer)
6. **margin_by_brand** (cpo)

**Example Template:**
```yaml
- id: spend_by_channel_over_time
  utterances:
    - "spend by channel"
    - "show ad spend per channel over time"
    - "channel spend over time"
    - "ad spend by channel"
  role_hint: marketer
  sql: |
    SELECT c.channel,
           f.date,
           SUM(f.spend) AS total_spend
    FROM fact_ad_spend f
    JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
    WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
    GROUP BY 1,2
    ORDER BY 2 ASC, 1
    LIMIT {{limit}};
  dims: [channel, date]
  measures: [spend]
```

---

## Data Layer

### Sample Database (data/marketing.duckdb)

**Generated by:** `scripts/create_sample_data.py`

**Statistics:**
- **20 campaigns** across 5 channels (Google, Facebook, Instagram, LinkedIn, TikTok)
- **365 days** of data (last year)
- **10,963 sessions**
- **3,391 orders**
- **Total rows:** ~20,000+

**Schema Relationships:**
```
dim_campaigns (20 rows)
    ↓ (1:N)
fact_ad_spend (7,300 rows)  # 20 campaigns × 365 days
    ↓ (1:N)
fact_sessions (10,963 rows)
    ↓ (1:N)
fact_orders (3,391 rows)
    ↓ (1:N)
dim_products (50 SKUs)
dim_customers (2,000 customers)
```

**Generation Command:**
```bash
make sample-data
# or: uv run python scripts/create_sample_data.py
```

---

## Testing

### Test Suite (23 tests, 100% passing)

**Coverage:**

1. **test_duckdb_connector.py** (9 tests)
   - Connection management
   - SQL validation (SELECT-only)
   - LIMIT enforcement
   - Read-only mode (blocks DDL/DML)
   - Schema validation
   - Table listing

2. **test_triage_local.py** (5 tests)
   - Search keyword detection
   - Analysis keyword detection
   - Role inference
   - Confidence scoring
   - Edge cases (empty query, ambiguous query)

3. **test_local_text_to_sql.py** (6 tests)
   - Template matching (fuzzy)
   - Parameter filling (time_window_days, limit)
   - SQL validation
   - No match handling
   - Role-specific template selection

4. **test_agent_search.py** (3 tests)
   - End-to-end search workflow
   - Template match (no LLM usage)
   - No duplicate LIMIT clauses
   - Execution success

**Run Tests:**
```bash
make test          # Run all tests
make test-v        # Verbose output
make test-cov      # With coverage report
```

**All tests use fixtures and mocks:**
- Temporary DuckDB databases
- Sample schemas and templates
- No actual LLM calls in tests

---

## API Reference

### SearchAgent.search()

**Signature:**
```python
def search(
    question: str,
    role: Optional[str] = None,
    force_local: bool = False
) -> Dict[str, Any]
```

**Parameters:**
- `question`: Natural language question
- `role`: User role (marketer/ceo/cpo/coo) or None for auto-detection
- `force_local`: Force local-only mode (no LLM calls)

**Returns:**
```python
{
    'question': str,
    'role': str,
    'status': str,
    'triage': {...},
    'sql_generation': {...},
    'sql': str,
    'data': pd.DataFrame,
    'row_count': int,
    'chart_path': str,
    'chart_type': str,
    'summary': str,
    'steps': [...]
}
```

**Example:**
```python
from agents.agent_search import SearchAgent

result = search_agent.search(
    question="show ad spend per channel over time",
    role="marketer"
)

print(result['status'])      # "success"
print(result['row_count'])   # 245
print(result['sql'])         # SELECT c.channel, f.date, ...
print(result['summary'])     # "Query returned 245 rows..."
```

---

### DuckDBConnector.execute()

**Signature:**
```python
def execute(
    sql: str,
    enforce_limit: bool = True,
    enforce_select_only: bool = True
) -> pd.DataFrame
```

**Parameters:**
- `sql`: SQL query string
- `enforce_limit`: Add LIMIT if missing (default: True)
- `enforce_select_only`: Block non-SELECT queries (default: True)

**Returns:** pandas DataFrame

**Raises:**
- `ValueError`: Invalid SQL (non-SELECT, missing LIMIT)
- `duckdb.Error`: Database errors

**Example:**
```python
from core.duckdb_connector import DuckDBConnector

db = DuckDBConnector("./data/marketing.duckdb", default_limit=1000)
db.connect()

df = db.execute("SELECT * FROM dim_campaigns")
print(df.head())

db.close()
```

---

### LocalTextToSQL.generate()

**Signature:**
```python
def generate(
    question: str,
    role: Optional[str] = None,
    extract_entities: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `question`: Natural language question
- `role`: User role for template filtering
- `extract_entities`: Extract KPIs/dimensions (default: True)

**Returns:**
```python
{
    'method': str,           # 'template_match' or 'no_match'
    'sql': str,              # Generated SQL or None
    'confidence': float,     # 0.0 - 1.0
    'template_id': str,      # Matched template ID
    'valid': bool,           # SQL validation passed
    'validation_errors': List[str],
    'entities': {
        'kpis': List[str],
        'dimensions': List[str],
        'time_ranges': List[str]
    }
}
```

---

## Performance Characteristics

### LLM Call Minimization

**Test Case:** 100 questions from template utterances
- **Template-matched queries:** 100% (0 LLM calls)
- **Novel queries (first time):** 1 LLM call each
- **Novel queries (cached):** 0 LLM calls (filesystem cache hit)

**Result:** For known patterns, **zero LLM costs**.

### Query Execution Speed

**Benchmark:** DuckDB on marketing.duckdb (20K+ rows)
- Simple aggregation: ~5ms
- Join + GROUP BY: ~10-20ms
- Complex multi-join: ~30-50ms

**Note:** DuckDB is in-process, so no network latency.

### Template Matching Speed

**Benchmark:** 6 templates, fuzzy matching via SequenceMatcher
- Average time: ~0.5ms per question
- Max time: ~2ms (worst case)

**Note:** Dominated by string comparison, negligible overhead.

### End-to-End Latency

**Breakdown (typical template-matched query):**
1. Triage (local): ~0.5ms
2. SQL generation (template): ~1ms
3. DB execution: ~10ms
4. Visualization: ~50-100ms (matplotlib rendering)
5. Summary: ~1ms

**Total:** ~60-115ms (no LLM)

**With LLM fallback:**
- Add ~500-2000ms for API call (first time)
- Add ~0ms for cache hit (subsequent calls)

---

## Known Limitations

### Phase 1 Scope

**What's NOT included in v0.1.0:**

1. **Analysis Mode**
   - Driver analysis, segmentation, hypothesis testing
   - Multi-query decomposition
   - Statistical analysis

2. **Web UI**
   - Currently notebook-only
   - No REST API or Flask/FastAPI frontend

3. **Multi-user Support**
   - No authentication or authorization
   - Single-user notebook sessions

4. **Advanced Features**
   - Streaming results for large datasets
   - Natural language chart annotations
   - Redis/database-backed cache (filesystem only)
   - Query history or favorites

### Technical Constraints

1. **Template Coverage**
   - Only 6 pre-built templates
   - Novel queries require LLM or fail gracefully

2. **Triage Accuracy**
   - Keyword matching can be confused by edge cases
   - Example: "what drives conversion" contains "what" (search keyword) and "drives" (analysis keyword)
   - Workaround: Use "identify drivers of conversion" instead

3. **Database Constraints**
   - DuckDB is single-writer (read-only mode for safety)
   - No support for real-time/streaming data

4. **Visualization**
   - Limited to 6 chart types
   - No interactive charts (static PNG only)

5. **LLM Fallback**
   - Requires API keys and internet connection
   - Not optimized for complex queries (single-shot prompting)

---

## Performance Tuning

### Recommended Settings

**For production:**
```yaml
# config/db.yaml
default_limit: 1000          # Prevent large result sets

# Environment
MODEL_PROVIDER: openai       # Or anthropic
OPENAI_MODEL: gpt-4o-mini    # Fast, cheap model
```

**For development:**
```python
# Force local-only mode (no LLM calls)
result = search_agent.search(question, role, force_local=True)
```

### Cache Management

**Clear cache:**
```bash
make clean
# or: rm -rf .cache/llm/*.json
```

**Cache location:**
- `.cache/llm/*.json` (persistent filesystem cache)

**Cache key:**
- SHA256 hash of `{provider}:{model}:{prompt}`

---

## Future Roadmap (Phase 2+)

### Planned Features

1. **Analysis Mode** (high priority)
   - Driver analysis (correlation, regression)
   - Segmentation (clustering, cohort analysis)
   - Hypothesis testing (A/B test analysis)

2. **Web UI** (medium priority)
   - Flask/FastAPI REST API
   - React/Vue frontend
   - Real-time query execution

3. **Observability Dashboard** (medium priority)
   - Token usage tracking
   - Cache hit rates
   - Query latency metrics

4. **Advanced SQL Generation** (low priority)
   - Multi-query decomposition
   - Subquery generation
   - CTE optimization

5. **Enterprise Features** (low priority)
   - Authentication & authorization
   - Multi-user support
   - Query history and favorites
   - Scheduled queries
   - Alert system

---

## Development Workflow

### Common Commands

```bash
make install       # Install dependencies
make sample-data   # Generate sample database
make test          # Run test suite
make test-cov      # Run tests with coverage
make run           # Launch Jupyter notebook
make clean         # Remove cache files
make check         # Run all checks (test + lint)
```

### Adding New Templates

1. Edit `config/sql_templates.yaml`
2. Add template with utterances, SQL, and metadata
3. Run tests: `make test`
4. Test in notebook

### Adding New Roles

1. Edit `config/business_context.yaml`
2. Add role with KPIs, dimensions, synonyms, defaults
3. Update documentation

### Debugging

**Enable verbose output:**
```python
result = search_agent.search(question, role)
print(result['steps'])  # View all steps with confidence scores
```

**Force local-only:**
```python
result = search_agent.search(question, role, force_local=True)
# No LLM calls, will fail gracefully if no template match
```

---

## Appendix

### Dependencies (pyproject.toml)

```toml
[project]
name = "tasman-marketing-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "duckdb>=1.1.3",
    "pandas>=2.2.3",
    "numpy>=2.1.3",
    "matplotlib>=3.9.2",
    "seaborn>=0.13.2",
    "pyyaml>=6.0.2",
    "python-dotenv>=1.0.1",
    "openai>=1.54.4",
    "anthropic>=0.39.0",
    "jupyter>=1.1.1",
    "pytest>=8.3.3",
]
```

### File Manifest (37 files)

**Core Logic (5 files):**
- core/duckdb_connector.py (243 lines)
- core/local_text_to_sql.py (198 lines)
- core/triage_local.py (156 lines)
- core/llm_clients.py (187 lines)
- core/viz.py (134 lines)

**Agents (3 files):**
- agents/agent_search.py (151 lines)
- agents/agent_text_to_sql.py (89 lines)
- agents/agent_triage.py (67 lines)

**Tests (4 files, 23 tests):**
- tests/test_duckdb_connector.py (9 tests)
- tests/test_triage_local.py (5 tests)
- tests/test_local_text_to_sql.py (6 tests)
- tests/test_agent_search.py (3 tests)

**Configuration (4 files):**
- config/schema.json (8 tables)
- config/sql_templates.yaml (6 templates)
- config/business_context.yaml (4 roles)
- config/db.yaml

**Documentation (8 files):**
- README.md (391 lines)
- VERSION_0.1_STATE.md (603 lines)
- QUICKSTART.md
- CHANGELOG.md
- .github/COMMIT_STYLE_GUIDE.md
- .github/COMMIT_INSTRUCTIONS.md
- docs/TECHNICAL_SPEC_v0.1.0.md (this file)
- docs/microagentic_architecture.png

**Scripts & Tools:**
- scripts/create_sample_data.py (sample DB generator)
- docs/architecture_diagram.py (diagram generator)
- Makefile (10 commands)
- notebooks/Agentic_Analytics_Demo.ipynb (10 sections)

---

**End of Technical Specification v0.1.0**
