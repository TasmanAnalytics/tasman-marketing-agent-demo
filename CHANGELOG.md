# Changelog

All notable changes to Tasman Agentic Analytics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-03

### Added

**Core Infrastructure:**
- Read-only DuckDB connector with safety guardrails
- Automatic visualization engine with smart chart type detection
- Rule-based query triage (search vs analysis classification)
- Template-based SQL generation with fuzzy matching
- LLM client abstraction (OpenAI + Anthropic) with filesystem caching

**Agents:**
- TriageAgent: Local rules → LLM fallback pattern
- TextToSQLAgent: Template matching → LLM fallback pattern
- SearchAgent: End-to-end query orchestration

**Configuration:**
- JSON schema definition for 8-table marketing database
- YAML business context with 4 roles (Marketer, CEO, CPO, COO)
- 6 pre-built SQL templates for common queries
- Role-specific KPIs, dimensions, and synonyms

**User Interface:**
- Comprehensive Jupyter notebook with 10 interactive sections
- Sample query examples (template-matched + novel)
- Observability & debugging tools
- Interactive query interface

**Testing:**
- 20 unit tests covering core functionality
- pytest fixtures for database and agent testing
- 100% test pass rate

**Developer Tools:**
- Makefile with common commands (install, test, run)
- Sample data generator (20K+ rows, 365 days)
- Environment configuration template
- Comprehensive documentation

**Documentation:**
- README with architecture overview
- QUICKSTART guide (5-minute setup)
- Inline code documentation
- Example queries and expected outputs

### Features

**Local-First Design:**
- Zero LLM calls for template-matched queries
- Keyword-based triage with 90%+ accuracy
- Template library for common analytics questions
- Entity extraction (KPIs, dimensions, time ranges)
- Schema validation on startup

**Safety & Reliability:**
- Read-only database connections
- DDL/DML query rejection
- Automatic LIMIT injection
- SQL validation against schema
- Graceful LLM fallback handling

**Multi-Role Support:**
- Marketer: Focus on spend, CTR, CVR, ROAS
- CEO: Focus on revenue, margin, growth
- CPO: Focus on orders, products, margins
- COO: Focus on operations, fulfillment

**Observability:**
- Step-by-step execution tracking
- Confidence scoring for all decisions
- LLM usage indicators (local vs API)
- Cache hit/miss statistics
- Detailed error messages

### Technical Details

**Dependencies:**
- Python 3.11+
- DuckDB 1.4.1
- Pandas 2.3.3
- Matplotlib 3.10.7
- OpenAI SDK 2.6.1
- Anthropic SDK 0.72.0
- Jupyter 1.1.1
- pytest 8.4.2

**Database Schema:**
- 3 dimension tables (campaigns, products, customers)
- 2 additional dimension tables (adgroups, creatives)
- 3 fact tables (ad_spend, sessions, orders)

**Performance:**
- Template-matched queries: <10ms, $0
- Novel queries (with LLM): ~200ms, ~$0.0001
- Filesystem cache: Permanent storage
- Sample database: 20K rows, 365 days

### Known Limitations

- Analysis mode not yet implemented (Phase 2)
- No multi-query decomposition (Phase 2)
- No web UI (Phase 2)
- No streaming results (Phase 2)
- LLM cache is local filesystem only

### Migration Notes

This is the initial release (v0.1.0). No migration needed.

---

## [Unreleased]

Features planned for Phase 2+:
- Analysis mode (driver analysis, segmentation, hypothesis testing)
- Multi-query workflow orchestration
- Web UI (Flask/FastAPI)
- Observability dashboard
- Streaming results for large datasets
- Redis/database-backed LLM cache
- Natural language chart annotations

---

[0.1.0]: https://github.com/yourusername/tasman-marketing-agent/releases/tag/v0.1.0
