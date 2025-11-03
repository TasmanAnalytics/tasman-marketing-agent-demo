Awesome — here’s the adjusted Phase-1 brief tuned for a notebook-first demo, minimal API usage, config-driven context, and no synthetic data generation.

⸻

🧠 Tasman Agentic Analytics — Phase 1 (Notebook Demo) Brief

🎯 Objective

Deliver a Jupyter notebook demo that shows the end-to-end “Search” path with local-first logic and minimal LLM calls:
	1.	Take a natural-language question + user role.
	2.	Map to SQL against a local DuckDB (existing data; no synthetic generation).
	3.	Execute + visualise.
	4.	Triage: rule-based first; only call LLM if ambiguous.
	5.	Return a clean, reproducible result (SQL + chart + short summary).

This phase implements:
	•	Notebook as the main execution surface.
	•	Config directory for schema, DB location, and business context.
	•	Local SQL template library + schema understanding.
	•	Tiny LLM usage (only when needed) with caching & easy model switch (ChatGPT ↔︎ Claude).

⸻

🗂 Project Layout (repo)

agentic_framework/
  ├── notebooks/
  │   └── Agentic_Analytics_Demo.ipynb
  ├── config/
  │   ├── schema.json                 # authoritative schema for DuckDB
  │   ├── business_context.yaml       # roles, KPIs, vocab, join hints
  │   ├── db.yaml                     # path to DuckDB and table names
  │   └── sql_templates.yaml          # local query patterns (search use-cases)
  ├── core/
  │   ├── duckdb_connector.py
  │   ├── local_text_to_sql.py        # rule/template engine; optional LLM fallback hook
  │   ├── triage_local.py             # rule-based triage
  │   ├── llm_clients.py              # ChatGPT + Claude thin clients + cache
  │   └── viz.py                      # simple auto-plot
  ├── agents/
  │   ├── agent_text_to_sql.py        # uses local engine first, LLM only if needed
  │   ├── agent_search.py
  │   ├── agent_triage.py             # wraps triage_local + (optional) LLM
  │   └── __init__.py
  ├── .env.example                    # OPENAI_API_KEY= / ANTHROPIC_API_KEY=
  ├── pyproject.toml
  ├── uv.lock
  ├── tests/                          # optional; runnable outside the notebook
  │   ├── test_duckdb_connector.py
  │   ├── test_local_text_to_sql.py
  │   ├── test_agent_search.py
  │   └── test_agent_triage.py
  └── README.md


⸻

🧰 Tooling & Environment
	•	Main execution: notebooks/Agentic_Analytics_Demo.ipynb
	•	Package manager: uv
	•	Python 3.11+
	•	Packages: duckdb, pandas, matplotlib, pyyaml, python-dotenv, pytest, tqdm, openai, anthropic, jsonschema
	•	Env: .env (use .env.example to create)
	•	OPENAI_API_KEY=...
	•	ANTHROPIC_API_KEY=...
	•	MODEL_PROVIDER=openai|anthropic
	•	OPENAI_MODEL=gpt-4.1-mini (default)
	•	ANTHROPIC_MODEL=claude-3-5-haiku (default)
	•	Config:
	•	config/db.yaml

duckdb_path: "./data/marketing.duckdb"   # change to your local path
default_limit: 1000


	•	config/schema.json — full schema (exact columns & types you gave)
	•	config/business_context.yaml
	•	roles → relevant KPIs, preferred dims/measures, vocabulary synonyms
	•	join hints (e.g., fact_ad_spend.campaign_id -> dim_campaigns.campaign_id)
	•	config/sql_templates.yaml
	•	canonical search questions → parameterised SQL
	•	e.g., “spend by channel over time”, “CTR by campaign”, “orders by category”

⸻

🧩 Data & Schema (No Synthetic Generation)
	•	We assume you already have a DuckDB with these tables and columns:

Dimensions
	•	dim_campaigns (campaign_id INT, channel TEXT, campaign_name TEXT, start_date DATE, end_date DATE, objective TEXT)
	•	dim_adgroups (adgroup_id INT, campaign_id INT, audience TEXT, placement TEXT)
	•	dim_creatives (creative_id INT, adgroup_id INT, format TEXT, asset_url TEXT)
	•	dim_products (sku TEXT, category TEXT, subcategory TEXT, price FLOAT, margin FLOAT, brand TEXT)
	•	dim_customers (customer_id INT, acquisition_channel TEXT, first_visit_date DATE, region TEXT)

Facts
	•	fact_ad_spend (date DATE, campaign_id INT, adgroup_id INT, creative_id INT, spend FLOAT, impressions INT, clicks INT)
	•	fact_sessions (session_id TEXT, customer_id INT, date DATE, campaign_id INT, adgroup_id INT, creative_id INT, utm_source TEXT, utm_medium TEXT, utm_campaign TEXT, device TEXT, pages_viewed INT, converted_flag BOOLEAN)
	•	fact_orders (order_id TEXT, session_id TEXT, customer_id INT, order_timestamp TIMESTAMP, sku TEXT, quantity INT, revenue FLOAT, margin FLOAT)

The notebook should validate schema at startup and fail fast if any table/column is missing.

⸻

🧠 Local-First Agentic Design (minimise LLM)

0) Rule-based triage (local)
	•	triage_local.py: keyword & pattern rules:
	•	Search if query is “what/how many/how much/when/show/list/plot…”
	•	Candidate analysis if “why/driver/impact/causal/segment/cluster”
	•	Returns {mode: "search"|"analysis", confidence: 0–1, reason}.
	•	Only call LLM if confidence < 0.6 and the question is ambiguous.

1) Local Text-to-SQL
	•	local_text_to_sql.py:
	•	Template matcher: map question + role + business_context → a parametric template (from sql_templates.yaml).
	•	Heuristics for dims/measures inference (e.g., “spend”, “CTR”, “orders”, “revenue”, “margin”).
	•	Schema guardrail: ensure referenced columns/tables exist.
	•	Default time window: past 90 days if missing.
	•	Always LIMIT (from db.yaml).
	•	If template/heuristic mapping fails: only then call LLM (with strict prompt + schema) to produce SQL. Cache result.

2) Query execution & visualisation (local)
	•	duckdb_connector.py executes SQL safely (read-only session, LIMIT injection if missing).
	•	viz.py:
	•	Simple auto-chart: time series → line; categorical → bar; two numeric → scatter.
	•	Save PNG in notebooks/outputs/.
	•	Return path + a 1-2 sentence textual summary.

3) LLM clients & caching
	•	llm_clients.py:
	•	Tiny wrappers for OpenAI/Anthropic chosen by MODEL_PROVIDER.
	•	Filesystem cache (.cache/llm/) keyed by: hash(user_question + role + schema_hash + template_version).
	•	Temperature ≤ 0.2; max tokens small (e.g., 512).
	•	Expose two calls only:
	•	triage_llm(question, role, business_context) — used only if local triage is low-confidence.
	•	text_to_sql_llm(question, role, schema_json, business_context) — used only if local SQL fails.

⸻

🔒 Testing & Quality (even in a Notebook demo)
	•	The notebook includes “Test cells”:
	1.	Schema check: confirm all required tables/columns exist.
	2.	Template coverage: for N canonical questions, local engine emits valid SQL.
	3.	LLM fallback: simulate an unknown question → valid SQL generated & cached.
	4.	Viz output: chart files created; DataFrame non-empty for common queries.
	•	Optional pytest tests (runnable outside the notebook) mirror the above.

⸻

📝 Notebook Walkthrough (sections)
	1.	Setup
	•	Load .env, choose provider/model, load configs.
	•	Validate DB connectivity & schema.
	2.	Business Context & Templates
	•	Show role KPIs (from business_context.yaml).
	•	Print available templates & what they cover.
	3.	Ask a Question
	•	Inputs: user_question, user_role.
	•	Triage (local → optional LLM only if needed).
	•	If mode == "search":
	•	Local Text-to-SQL (template/heuristics → optional LLM).
	•	Execute SQL; show head() and chart.
	•	Summarise result (1–2 sentences).
	4.	Observability
	•	Display JSON logs of each step (triage, template chosen, SQL, execution time, cache hits).
	5.	Test Cells
	•	Run quick smoke tests.

⸻

🔧 Config Examples

business_context.yaml (excerpt)

roles:
  marketer:
    kpis: [spend, impressions, clicks, ctr, cvr, cac, roas]
    dims: [channel, campaign_name, audience, device, region]
    synonyms:
      revenue: [sales, takings]
      impressions: [views]
      clicks: [taps]
    defaults:
      time_window_days: 90

joins:
  fact_ad_spend:
    campaign_id: dim_campaigns.campaign_id
    adgroup_id: dim_adgroups.adgroup_id
    creative_id: dim_creatives.creative_id
  fact_sessions:
    campaign_id: dim_campaigns.campaign_id
    adgroup_id: dim_adgroups.adgroup_id
    creative_id: dim_creatives.creative_id
  fact_orders:
    sku: dim_products.sku
    customer_id: dim_customers.customer_id

sql_templates.yaml (excerpt)

- id: spend_by_channel_over_time
  utterances: ["spend by channel", "how much did we spend per channel", "plot channel spend"]
  role_hint: "marketer"
  sql: |
    SELECT c.channel,
           f.date,
           SUM(f.spend) AS total_spend
    FROM fact_ad_spend f
    JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
    WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
    GROUP BY 1,2
    ORDER BY 2 DESC
    LIMIT {{limit}};
  dims: [channel, date]
  measures: [spend]


⸻

🧩 LLM Prompts (lean + token-tight)

Text-to-SQL (fallback only)

System

You convert a business question into a VALID DuckDB SQL query using ONLY the provided schema.
Rules:
- Use only known tables/columns.
- Past 90 days if no date filter is given.
- LIMIT {{limit}} at the end.
- Return EXACTLY one fenced SQL block and NOTHING else.

User

Role: {role}
Question: {question}
Schema JSON:
{schema_json}
Business context (dims/measures/synonyms/hints):
{business_context}

Triage (fallback only)

System

Classify if a question is simple descriptive "search" or requires "analysis".
Return ONLY this JSON:
{"mode": "...", "analysis_type": null|"hypothesis_testing"|"driver_analysis"|"segmentation", "confidence": 0.0-1.0, "reason":"..."}

User

Role: {role}
Question: {question}

----

Example files to be used

⸻

config/business_context.yaml (starter)

roles:
  marketer:
    kpis: [spend, impressions, clicks, ctr, cvr, cac, roas, revenue, margin]
    dims: [channel, campaign_name, audience, placement, device, region, category, subcategory]
    synonyms:
      spend: [ad_spend, cost]
      impressions: [views]
      clicks: [taps]
      ctr: [click_through_rate]
      cvr: [conversion_rate]
      revenue: [sales, takings]
      margin: [profit]
      channel: [utm_source, source]
    defaults:
      time_window_days: 90
  ceo:
    kpis: [revenue, margin, growth, cac, ltv, roas]
    dims: [channel, region, brand, category]
    synonyms:
      growth: [trend, run_rate]
      ltv: [lifetime_value]
    defaults:
      time_window_days: 180
  cpo:
    kpis: [orders, revenue, margin, attach_rate, conversion_rate]
    dims: [sku, category, subcategory, brand]
    synonyms:
      attach_rate: [attach, bundle_rate]
      conversion_rate: [cvr]
    defaults:
      time_window_days: 120
  coo:
    kpis: [orders, fulfillment_rate, cancellations, margin, revenue]
    dims: [region, category, device]
    synonyms:
      cancellations: [refunds, returns]
    defaults:
      time_window_days: 120

joins:
  fact_ad_spend:
    campaign_id: dim_campaigns.campaign_id
    adgroup_id: dim_adgroups.adgroup_id
    creative_id: dim_creatives.creative_id
  fact_sessions:
    campaign_id: dim_campaigns.campaign_id
    adgroup_id: dim_adgroups.adgroup_id
    creative_id: dim_creatives.creative_id
    customer_id: dim_customers.customer_id
  fact_orders:
    sku: dim_products.sku
    customer_id: dim_customers.customer_id

calculated_measures:
  ctr: "CAST(SUM(clicks) AS DOUBLE) / NULLIF(SUM(impressions), 0)"
  cvr: "CAST(SUM(CASE WHEN converted_flag THEN 1 ELSE 0 END) AS DOUBLE) / NULLIF(COUNT(*), 0)"
  roas: "CAST(SUM(revenue) AS DOUBLE) / NULLIF(SUM(spend), 0)"

defaults:
  time_window_days: 90
  limit: 1000


⸻

config/sql_templates.yaml (starter)

- id: spend_by_channel_over_time
  utterances:
    - "spend by channel"
    - "show ad spend per channel over time"
    - "plot channel spend trend"
  role_hint: "marketer"
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

- id: ctr_by_campaign
  utterances:
    - "ctr by campaign"
    - "which campaigns have the best click-through rate"
  role_hint: "marketer"
  sql: |
    SELECT c.campaign_name,
           SUM(f.impressions) AS impressions,
           SUM(f.clicks) AS clicks,
           CASE WHEN SUM(f.impressions)=0 THEN NULL
                ELSE SUM(f.clicks)::DOUBLE / SUM(f.impressions) END AS ctr
    FROM fact_ad_spend f
    JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
    WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
    GROUP BY 1
    ORDER BY ctr DESC NULLS LAST
    LIMIT {{limit}};
  dims: [campaign_name]
  measures: [impressions, clicks, ctr]

- id: orders_and_revenue_by_category
  utterances:
    - "orders by category"
    - "revenue by product category"
  role_hint: "ceo"
  sql: |
    SELECT p.category,
           COUNT(DISTINCT o.order_id) AS orders,
           SUM(o.revenue) AS revenue,
           SUM(o.margin)  AS margin
    FROM fact_orders o
    JOIN dim_products p ON o.sku = p.sku
    WHERE o.order_timestamp >= NOW() - INTERVAL {{time_window_days}} DAY
    GROUP BY 1
    ORDER BY revenue DESC
    LIMIT {{limit}};
  dims: [category]
  measures: [orders, revenue, margin]

- id: roas_by_channel
  utterances:
    - "roas by channel"
    - "which channel is most efficient"
  role_hint: "ceo"
  sql: |
    WITH spend AS (
      SELECT c.channel, SUM(f.spend) AS spend
      FROM fact_ad_spend f
      JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
      WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
      GROUP BY 1
    ),
    rev AS (
      SELECT c.channel, SUM(o.revenue) AS revenue
      FROM fact_orders o
      JOIN fact_sessions s ON o.session_id = s.session_id
      JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
      WHERE o.order_timestamp >= NOW() - INTERVAL {{time_window_days}} DAY
      GROUP BY 1
    )
    SELECT COALESCE(spend.channel, rev.channel) AS channel,
           spend.spend,
           rev.revenue,
           CASE WHEN spend.spend=0 OR spend.spend IS NULL THEN NULL
                ELSE rev.revenue::DOUBLE / spend.spend END AS roas
    FROM spend FULL OUTER JOIN rev USING(channel)
    ORDER BY roas DESC NULLS LAST
    LIMIT {{limit}};
  dims: [channel]
  measures: [spend, revenue, roas]

- id: device_sessions_and_cvr
  utterances:
    - "conversion rate by device"
    - "sessions and cvr by device"
  role_hint: "marketer"
  sql: |
    SELECT s.device,
           COUNT(*) AS sessions,
           SUM(CASE WHEN s.converted_flag THEN 1 ELSE 0 END) AS conversions,
           CASE WHEN COUNT(*)=0 THEN NULL
                ELSE SUM(CASE WHEN s.converted_flag THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) END AS cvr
    FROM fact_sessions s
    WHERE s.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
    GROUP BY 1
    ORDER BY cvr DESC NULLS LAST
    LIMIT {{limit}};
  dims: [device]
  measures: [sessions, conversions, cvr]

- id: margin_by_brand
  utterances:
    - "margin by brand"
    - "which brands contribute most margin"
  role_hint: "cpo"
  sql: |
    SELECT p.brand,
           SUM(o.margin) AS margin,
           SUM(o.revenue) AS revenue,
           COUNT(DISTINCT o.order_id) AS orders
    FROM fact_orders o
    JOIN dim_products p ON o.sku = p.sku
    WHERE o.order_timestamp >= NOW() - INTERVAL {{time_window_days}} DAY
    GROUP BY 1
    ORDER BY margin DESC
    LIMIT {{limit}};
  dims: [brand]
  measures: [margin, revenue, orders]


⸻

First notebook cell — bootstrap & validation

File: notebooks/Agentic_Analytics_Demo.ipynb (Cell 1)

# --- Bootstrap: env, config, DB connectivity, schema validation ---

import os, json, hashlib, textwrap
from pathlib import Path
from datetime import datetime

import duckdb
import pandas as pd
import yaml

# 1) Load environment (.env is optional here; you can also rely on shell env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROJECT_ROOT = Path.cwd().resolve().parents[0] if (Path.cwd().name == "notebooks") else Path.cwd()
CONFIG_DIR   = PROJECT_ROOT / "config"
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = PROJECT_ROOT / ".cache" / "llm"
OUTPUT_DIR   = Path.cwd() / "outputs"

for d in (CACHE_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 2) Load configs
def _read_yaml(p: Path):
    with open(p, "r") as f:
        return yaml.safe_load(f)

db_cfg_path       = CONFIG_DIR / "db.yaml"
schema_json_path  = CONFIG_DIR / "schema.json"
biz_ctx_path      = CONFIG_DIR / "business_context.yaml"
templates_path    = CONFIG_DIR / "sql_templates.yaml"

assert db_cfg_path.exists(), f"Missing {db_cfg_path}"
assert schema_json_path.exists(), f"Missing {schema_json_path}"
assert biz_ctx_path.exists(), f"Missing {biz_ctx_path}"
assert templates_path.exists(), f"Missing {templates_path}"

db_cfg       = _read_yaml(db_cfg_path)
biz_ctx      = _read_yaml(biz_ctx_path)
templates    = _read_yaml(templates_path)
with open(schema_json_path, "r") as f:
    schema_spec = json.load(f)

DUCKDB_PATH  = Path(db_cfg.get("duckdb_path", "./data/marketing.duckdb")).expanduser().resolve()
DEFAULT_LIMIT = int(db_cfg.get("default_limit", 1000))

print(f"✅ Config loaded. DB: {DUCKDB_PATH} | LIMIT default: {DEFAULT_LIMIT}")

# 3) Select model provider (but we’ll avoid calls unless needed)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").lower()  # "openai" or "anthropic"
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ANTHROPIC_MODEL= os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if MODEL_PROVIDER == "openai":
    if not OPENAI_API_KEY:
        print("ℹ️ OPENAI_API_KEY not set — LLM fallback will be disabled.")
elif MODEL_PROVIDER == "anthropic":
    if not ANTHROPIC_API_KEY:
        print("ℹ️ ANTHROPIC_API_KEY not set — LLM fallback will be disabled.")
else:
    print(f"⚠️ Unknown MODEL_PROVIDER '{MODEL_PROVIDER}'. Falling back to local-only.")

# 4) Connect to DuckDB
assert DUCKDB_PATH.exists(), f"DuckDB file not found: {DUCKDB_PATH}"
con = duckdb.connect(database=str(DUCKDB_PATH), read_only=True)
print("✅ DuckDB connected (read-only).")

# 5) Introspect actual schema from DB
def list_tables(conn):
    return [r[0] for r in conn.execute("SHOW TABLES").fetchall()]

def list_columns(conn, table):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # rows: [ (cid, name, type, notnull, dflt_value, pk) ... ]
    return {r[1]: r[2] for r in rows}

actual_tables = list_tables(con)
print(f"📦 Tables present: {actual_tables}")

# 6) Validate schema — ensure all required tables & columns exist
missing = []
for tbl, cols in schema_spec.items():
    if tbl not in actual_tables:
        missing.append(f"Missing table: {tbl}")
        continue
    actual_cols = list_columns(con, tbl)
    # accept type differences loosely; enforce column presence
    for col in cols.keys():
        if col not in actual_cols:
            missing.append(f"Missing column: {tbl}.{col}")

if missing:
    print("❌ Schema validation failed:")
    for m in missing:
        print("  -", m)
    raise SystemExit("Fix your DuckDB schema and re-run.")
else:
    print("✅ Schema validation passed.")

# 7) Helper: inject LIMIT if missing (basic safeguard)
def ensure_limit(sql: str, limit: int = DEFAULT_LIMIT) -> str:
    lowered = sql.lower()
    if " limit " in lowered:
        return sql
    # naive check to avoid adding after semicolon
    sql_ = sql.strip().rstrip(";")
    return f"{sql_}\nLIMIT {limit};"

# 8) Helper: simple read-only guard (reject DDL/DML)
_PROHIBITED = ("insert ", "update ", "delete ", "create ", "alter ", "drop ", "truncate ", "merge ")
def is_select_only(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select ") and not any(k in s for k in _PROHIBITED)

print("🔧 Bootstrap complete. You can now run local triage & template SQL in the next cells.")

Note: config/schema.json should be a simple object mapping table → columns (types optional). Example:

{
  "dim_campaigns": {
    "campaign_id": "INT",
    "channel": "TEXT",
    "campaign_name": "TEXT",
    "start_date": "DATE",
    "end_date": "DATE",
    "objective": "TEXT"
  },
  "dim_adgroups": {
    "adgroup_id": "INT",
    "campaign_id": "INT",
    "audience": "TEXT",
    "placement": "TEXT"
  },
  "dim_creatives": {
    "creative_id": "INT",
    "adgroup_id": "INT",
    "format": "TEXT",
    "asset_url": "TEXT"
  },
  "dim_products": {
    "sku": "TEXT",
    "category": "TEXT",
    "subcategory": "TEXT",
    "price": "FLOAT",
    "margin": "FLOAT",
    "brand": "TEXT"
  },
  "dim_customers": {
    "customer_id": "INT",
    "acquisition_channel": "TEXT",
    "first_visit_date": "DATE",
    "region": "TEXT"
  },
  "fact_ad_spend": {
    "date": "DATE",
    "campaign_id": "INT",
    "adgroup_id": "INT",
    "creative_id": "INT",
    "spend": "FLOAT",
    "impressions": "INT",
    "clicks": "INT"
  },
  "fact_sessions": {
    "session_id": "TEXT",
    "customer_id": "INT",
    "date": "DATE",
    "campaign_id": "INT",
    "adgroup_id": "INT",
    "creative_id": "INT",
    "utm_source": "TEXT",
    "utm_medium": "TEXT",
    "utm_campaign": "TEXT",
    "device": "TEXT",
    "pages_viewed": "INT",
    "converted_flag": "BOOLEAN"
  },
  "fact_orders": {
    "order_id": "TEXT",
    "session_id": "TEXT",
    "customer_id": "INT",
    "order_timestamp": "TIMESTAMP",
    "sku": "TEXT",
    "quantity": "INT",
    "revenue": "FLOAT",
    "margin": "FLOAT"
  }
}



⸻

✅ Acceptance Criteria
	•	Notebook runs start-to-finish with a local DuckDB path from config/db.yaml.
	•	For common marketer queries, no LLM calls (template hit).
	•	If a novel query appears, one LLM call (triage or SQL) at most, and then cached.
	•	All SQLs validated against schema; results limited; chart saved.
	•	Test cells pass.

⸻

📋 Documentation Standards

**Technical Specification Requirement:**

For EVERY version and subversion release, you MUST create a comprehensive technical specification document:

**Location:** `docs/TECHNICAL_SPEC_v{VERSION}.md`

**Required Sections:**
1. **System Overview** - Purpose, design philosophy, technology stack
2. **Architecture** - Layer structure, query flow, component diagram
3. **Core Components** - Detailed documentation of each module with:
   - Purpose and responsibilities
   - Key methods and signatures
   - Configuration options
   - Safety features and guardrails
   - Testing coverage
4. **Agents** - Agent orchestration logic and workflow
5. **Configuration** - Schema, business context, templates with examples
6. **Data Layer** - Database structure, relationships, sample data stats
7. **Testing** - Test suite coverage, commands, fixture details
8. **API Reference** - Function signatures, parameters, returns, examples
9. **Performance Characteristics** - Benchmarks, latency, LLM usage stats
10. **Known Limitations** - Scope boundaries, technical constraints, workarounds
11. **Future Roadmap** - Planned features for next phases
12. **Development Workflow** - Common commands, adding templates/roles, debugging
13. **Appendix** - Dependencies, file manifest, version info

**Version Naming Convention:**
- Major releases: `TECHNICAL_SPEC_v1.0.0.md`
- Minor releases: `TECHNICAL_SPEC_v1.1.0.md`
- Patch releases: `TECHNICAL_SPEC_v1.0.1.md`

**When to Create:**
- Before marking a version as production-ready
- After all features for that version are implemented
- Before creating git tags/releases
- When major architectural changes occur

**Quality Standards:**
- Minimum 500 lines of detailed technical content
- Include code examples for all major components
- Document all configuration options
- Provide performance benchmarks
- List known limitations and workarounds
- Include complete API reference with examples

⸻

🪜 Suggested Plan of Steps for Claude
	1.	Scaffold with uv
	•	uv init + add deps.
	•	Create configs (schema.json, db.yaml, business_context.yaml, sql_templates.yaml).
	2.	Core utilities
	•	duckdb_connector.py (read-only, LIMIT enforcement).
	•	viz.py (auto-chart; save PNG).
	3.	Local logic
	•	triage_local.py (keyword rules, confidence scoring).
	•	local_text_to_sql.py (template matcher + heuristics + schema guard).
	4.	LLM wrappers (optional path)
	•	llm_clients.py (OpenAI/Anthropic switch + filesystem cache).
	•	agent_text_to_sql.py and agent_triage.py call local logic first, LLM only if needed.
	5.	Notebook
	•	Build Agentic_Analytics_Demo.ipynb with the sections listed above.
	•	Add “Test cells”.
	6.	Smoke test with real DB
	•	Point config/db.yaml at your DuckDB.
	•	Try 3–5 canonical marketer questions (should be local-only).
	•	Try 1 novel question (should trigger LLM once, then cache).
	7.	(Optional) pytest
	•	Add basic tests mirroring notebook test-cells.

