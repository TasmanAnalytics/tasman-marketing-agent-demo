# Quick Start Guide

Get up and running with Tasman Agentic Analytics in 5 minutes!

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Clone the repository
cd tasman-marketing-agent

# Install dependencies (uv handles everything)
make install
# or: uv sync
```

## Setup

### 1. Create Sample Database

Generate a sample marketing database with synthetic data:

```bash
make sample-data
# or: uv run python scripts/create_sample_data.py
```

This creates `./data/marketing.duckdb` with:
- 20 campaigns across 5 channels
- 50 ad groups
- 100 creatives
- 200 products
- 5,000 customers
- ~10,000 sessions
- ~3,000 orders
- 365 days of data

### 2. Configure LLM (Optional)

The system works **without API keys** for template-matched queries!

To enable LLM fallback for novel queries:

```bash
# Copy environment template
cp .env.example .env

# Edit and add your API key
nano .env
```

Available providers:
- **OpenAI**: `gpt-4o-mini` (recommended, fast & cheap)
- **Anthropic**: `claude-3-5-haiku-20241022`

## Usage

### Launch Jupyter Notebook

```bash
make run
# or: uv run jupyter notebook notebooks/Agentic_Analytics_Demo.ipynb
```

### Try Sample Queries

The notebook includes 5+ example queries. Here are some to try:

**Template-matched (no LLM):**
- "show ad spend per channel over time"
- "which campaigns have the best click-through rate"
- "conversion rate by device"

**Role-specific:**
- "revenue by product category" (CEO role)
- "margin by brand" (CPO role)

**Auto role detection:**
- "show me spend and impressions by channel"

### Run Tests

```bash
# All tests
make test

# With verbose output
make test-v

# With coverage
make test-cov

# Specific module
uv run pytest tests/test_local_text_to_sql.py -v
```

## Makefile Commands

All common tasks have Makefile targets:

```bash
make help         # Show all commands
make install      # Install dependencies
make sample-data  # Generate database
make test         # Run tests
make run          # Launch notebook
make clean        # Remove cache files
```

## What to Expect

For **template-matched queries** (most common questions):
- ⚡ **Instant response** (no LLM call)
- 💾 **Direct SQL generation** from templates
- 📊 **Automatic visualization**

For **novel queries** (if LLM is configured):
- 🤖 **One LLM call** to generate SQL
- 💰 **~$0.0001 per query** (GPT-4o-mini)
- 💾 **Cached forever** (subsequent calls free)

## Example Output

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
[DataFrame with chart showing spend trends by channel]
```

## Next Steps

1. **Explore the notebook** - Run all cells to see the system in action
2. **Try your own queries** - Use the interactive query cell
3. **Add templates** - Edit `config/sql_templates.yaml` for your common queries
4. **Customize roles** - Edit `config/business_context.yaml` for your org
5. **Check cache stats** - See how many LLM calls you've saved!

## Troubleshooting

### "DuckDB file not found"

Run the sample data generator:
```bash
uv run python scripts/create_sample_data.py
```

### "LLM call failed"

The system will fall back gracefully. For novel queries without LLM:
- ℹ️ System will return "SQL generation failed"
- ✅ Template-matched queries still work perfectly
- 💡 Add API key to `.env` to enable LLM fallback

### Tests failing

Make sure dependencies are installed:
```bash
uv sync
```

## Learn More

- **Full documentation**: See [README.md](README.md)
- **Architecture details**: Check `claude-prompts/instructions-phase1.md`
- **Add more templates**: Edit `config/sql_templates.yaml`
- **Configure roles**: Edit `config/business_context.yaml`

---

**Have fun exploring! 🎉**
