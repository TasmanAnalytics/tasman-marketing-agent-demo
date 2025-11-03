# Tasman Agentic Analytics

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](./VERSION)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](./tests)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

A **local-first, notebook-driven** agentic analytics system that minimizes LLM usage through intelligent template matching and rule-based triage.

> **Version 0.1.0** - Initial release with search mode, template library, and multi-role support.

## 🎯 Key Features

- **Local-first architecture**: Template matching and rule-based logic before any LLM calls
- **Minimal LLM usage**: Only calls LLM when local logic fails; aggressive filesystem caching
- **Config-driven**: Schema, business context, and SQL templates in YAML/JSON
- **Multi-role support**: Marketer, CEO, CPO, COO roles with tailored KPIs and dimensions
- **Automatic visualization**: Smart chart generation based on data characteristics
- **Read-only safety**: SQL execution with guardrails and LIMIT enforcement
- **Full observability**: Track every step from triage → SQL → execution → visualization

## 📂 Project Structure

```
tasman-marketing-agent/
├── notebooks/
│   └── Agentic_Analytics_Demo.ipynb   # Main demo notebook
├── config/
│   ├── schema.json                     # Database schema definition
│   ├── db.yaml                         # Database connection config
│   ├── business_context.yaml           # Roles, KPIs, synonyms
│   └── sql_templates.yaml              # Canonical query templates
├── core/
│   ├── duckdb_connector.py             # Read-only DB connector
│   ├── local_text_to_sql.py            # Template-based SQL generation
│   ├── triage_local.py                 # Rule-based query triage
│   ├── llm_clients.py                  # OpenAI/Anthropic clients + caching
│   └── viz.py                          # Auto-visualization
├── agents/
│   ├── agent_triage.py                 # Triage orchestration
│   ├── agent_text_to_sql.py            # SQL generation orchestration
│   └── agent_search.py                 # End-to-end search agent
├── tests/                              # pytest test suite
├── data/                               # DuckDB database location
└── .env                                # Environment variables (create from .env.example)
```

## 🚀 Quick Start

This project uses [uv](https://github.com/astral-sh/uv) for fast package management and includes a Makefile for convenience.

### 1. Install Dependencies

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
make install
# or: uv sync
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys (optional - system works without them!)
# nano .env
```

### 3. Generate Sample Database

Create a sample database with synthetic marketing data:

```bash
make sample-data
# or: uv run python scripts/create_sample_data.py
```

This creates `./data/marketing.duckdb` with:
- 20 campaigns, 5 channels, 365 days of data
- 10,963 sessions, 3,391 orders
- Full schema with 8 tables

**Using your own database?** Update `config/db.yaml` and ensure your schema matches `config/schema.json`.

### 4. Launch Jupyter Notebook

```bash
make run
# or: uv run jupyter notebook notebooks/Agentic_Analytics_Demo.ipynb
```

### 5. Verify Installation

```bash
make test
# or: uv run pytest tests/ -v
```

All 20 tests should pass ✅

## 🧠 How It Works

### Query Flow

```
User Question
    ↓
[1] Triage (Rule-based → LLM fallback)
    ↓
[2] Text-to-SQL (Template match → LLM fallback)
    ↓
[3] Execute SQL (Read-only, LIMIT enforced)
    ↓
[4] Visualize (Auto chart type detection)
    ↓
[5] Summarize (Natural language result)
```

### Local-First Design

The system **minimizes LLM calls** through:

1. **Rule-based triage**: Keyword matching to classify queries as "search" vs "analysis"
2. **Template library**: Pre-built SQL templates for common queries (6+ templates included)
3. **Synonym mapping**: Role-specific vocabulary (e.g., "revenue" = "sales" = "takings")
4. **Entity extraction**: Automatically identify KPIs, dimensions, and time ranges
5. **Filesystem cache**: Hash-based caching of LLM responses

**Result**: For template-matched queries, zero LLM calls. For novel queries, one LLM call then cached.

## 📊 Example Usage

### From the Notebook

```python
# Template-matched query (no LLM)
result = search_agent.search(
    question="show ad spend per channel over time",
    role="marketer"
)

# Novel query (LLM fallback, then cached)
result = search_agent.search(
    question="what are the top 10 products by revenue in the last 30 days?",
    role="cpo"
)

# Auto role detection
result = search_agent.search(
    question="conversion rate by device"
)
```

### Sample Output

```
QUESTION: show ad spend per channel over time
ROLE: marketer
================================================================================

📋 STEPS:
  Triage: ⚡ (Local)
    Confidence: 0.90
  SQL Generation: ⚡ (Local)
    Method: template_match
    Confidence: 0.95

📊 STATUS: success

💾 SQL QUERY:
SELECT c.channel, f.date, SUM(f.spend) AS total_spend
FROM fact_ad_spend f
JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
WHERE f.date >= CURRENT_DATE - INTERVAL 90 DAY
GROUP BY 1,2
ORDER BY 2 ASC, 1
LIMIT 1000;

📊 RESULTS (245 rows)
[DataFrame preview with chart]
```

## 🧪 Running Tests

```bash
# Run all tests
make test

# Run with verbose output
make test-v

# Run with coverage report
make test-cov

# Run specific test file
uv run pytest tests/test_local_text_to_sql.py -v
```

## 🛠️ Makefile Commands

The project includes a Makefile for common tasks:

```bash
make help         # Show all available commands
make install      # Install dependencies
make sample-data  # Generate sample database
make test         # Run test suite
make test-cov     # Run tests with coverage
make run          # Launch Jupyter notebook
make clean        # Remove cache files
make check        # Run all checks (test + lint)
```

## 🎨 Configuration

### Adding New SQL Templates

Edit `config/sql_templates.yaml`:

```yaml
- id: my_new_template
  utterances:
    - "show me X by Y"
    - "plot X over time"
  role_hint: "marketer"
  sql: |
    SELECT ...
    WHERE date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
    LIMIT {{limit}};
  dims: [channel, date]
  measures: [spend]
```

### Adding New Roles

Edit `config/business_context.yaml`:

```yaml
roles:
  my_role:
    kpis: [revenue, orders, margin]
    dims: [region, category]
    synonyms:
      revenue: [sales, takings]
    defaults:
      time_window_days: 90
```

### Switching LLM Providers

In `.env`:

```bash
# Use OpenAI
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Or use Anthropic
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-haiku-20241022
```

## 🔧 Architecture Decisions

### Why Local-First?

- **Cost**: Minimize expensive LLM API calls
- **Speed**: Local matching is instant
- **Reliability**: Works offline for known queries
- **Transparency**: Full observability of decision logic

### Why DuckDB?

- **Fast**: In-process analytical queries
- **Portable**: Single-file database
- **SQL-complete**: Full analytical SQL support
- **Read-only mode**: Safety for analytics workloads

### Why Notebooks?

- **Exploratory**: Natural fit for analytics
- **Iterative**: Easy to experiment and refine
- **Sharable**: Mix code, results, and narrative
- **Production-ready**: Core logic can be extracted to scripts

## 📦 Version 0.1.0 - Current State

### ✅ What's Included

**Core Features:**
- ✅ Local-first architecture with template matching
- ✅ Rule-based triage (search vs analysis)
- ✅ 6 pre-built SQL templates
- ✅ Multi-role support (4 roles)
- ✅ Automatic visualization
- ✅ LLM fallback (OpenAI + Anthropic)
- ✅ Filesystem caching
- ✅ Read-only database safety

**Testing & Quality:**
- ✅ 20 unit tests (100% passing)
- ✅ pytest fixtures and mocks
- ✅ Sample data generator
- ✅ Comprehensive documentation

**Developer Experience:**
- ✅ Makefile for common tasks
- ✅ Interactive Jupyter notebook
- ✅ Environment configuration
- ✅ Quick start guide

### ⏳ What's Not Included (Yet)

**Planned for Phase 2+:**
- ⏳ Analysis mode (driver analysis, segmentation, hypothesis testing)
- ⏳ Multi-query decomposition
- ⏳ Web UI (Flask/FastAPI)
- ⏳ Observability dashboard
- ⏳ Streaming results
- ⏳ Natural language chart annotations
- ⏳ Redis/database-backed cache

### 🎯 Production Readiness

**Ready for production:**
- Search mode queries
- Template-matched analytics
- Role-based KPI access
- SQL execution safety

**Use with caution:**
- Novel queries (requires LLM configuration)
- Large result sets (manual LIMIT recommended)
- Concurrent users (notebook is single-user)

**Not production-ready:**
- Analysis mode (not implemented)
- Web API (use notebook for now)
- Multi-user access (no authentication)

## 📈 Roadmap

### Phase 2 (Planned)

- [ ] Analysis mode (hypothesis testing, driver analysis, segmentation)
- [ ] Multi-query workflows (decompose complex questions)
- [ ] Observability dashboard (token usage, cache hits, latency)
- [ ] Web UI (Flask/FastAPI frontend)
- [ ] Streaming results (for large datasets)
- [ ] Natural language chart annotations

### Future Considerations

- [ ] Authentication & authorization
- [ ] Multi-user support
- [ ] Query history & favorites
- [ ] Scheduled queries
- [ ] Alert system
- [ ] Export to various formats

## 🤝 Contributing

This is Phase 1 (proof-of-concept). To extend:

1. Add more templates to `config/sql_templates.yaml`
2. Add more roles to `config/business_context.yaml`
3. Extend `LocalTriage` with more sophisticated rules
4. Add more chart types to `core/viz.py`

## 📄 License

MIT

## 🙏 Acknowledgments

Built with:
- [DuckDB](https://duckdb.org/) - Fast analytical database
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- [OpenAI](https://openai.com/) / [Anthropic](https://anthropic.com/) - LLM providers
- [Jupyter](https://jupyter.org/) - Interactive notebook environment

---

**Questions?** Open an issue or check the notebook for detailed examples
