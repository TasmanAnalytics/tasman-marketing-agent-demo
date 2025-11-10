# Compass Demo: Bad vs Good Agent Architecture

This demo showcases the **correct** way to build LLM-powered analytics agents by contrasting two approaches:

## 📁 Contents

### Notebooks

- **`01_bad_oneshot_raw.ipynb`** - Anti-pattern demonstration
  - Shows what goes wrong with one-shot LLM on raw data
  - Intentional failures: wrong joins, time window drift, metric errors
  - Demonstrates overconfident but incorrect results

- **`02_good_modular_dspy.ipynb`** - Production-grade implementation
  - Modular DSPy agent architecture
  - Semantic layer integration
  - Reproducible, testable, observable
  - Quantified uncertainty with confidence intervals

### Utilities (`utils/`)

Lightweight support modules (clean-room implementation):

- `env_loader.py` - Environment variable management
- `db_connector.py` - DuckDB connection with schema validation
- `semantic_parser.py` - Semantic layer YAML parser and SQL compiler
- `plotting.py` - Matplotlib helpers for channel metrics
- `observability.py` - Run record for reproducibility

## 🎯 Business Question

Both notebooks address the same question:

> "Which channel mix change is most likely to improve CAC next month, given a recent anomaly in referral traffic?"

## 🏛️ Architecture Comparison

```mermaid
flowchart TB
    subgraph bad["❌ BAD: One-Shot LLM on Raw Data"]
        direction TB
        Q1[Business Question]
        L1[LLM]
        S1[Raw SQL Generation]
        D1[(DuckDB<br/>8 Tables)]
        R1[Results]
        N1[Narrative]

        Q1 -->|Schema Dump| L1
        L1 -->|Unvalidated SQL| S1
        S1 -->|Direct Query| D1
        D1 -->|Wrong Data| R1
        R1 -->|Overconfident| N1

        style L1 fill:#ff6b6b,stroke:#c92a2a,color:#fff
        style S1 fill:#ff6b6b,stroke:#c92a2a,color:#fff
        style R1 fill:#ff6b6b,stroke:#c92a2a,color:#fff
        style N1 fill:#ff6b6b,stroke:#c92a2a,color:#fff

        F1["⚠️ FAILURES:<br/>• Wrong joins<br/>• Time drift<br/>• Metric errors<br/>• No validation<br/>• No CI"]
        style F1 fill:#fff,stroke:#c92a2a,stroke-width:2px,stroke-dasharray: 5 5
    end

    subgraph good["✅ GOOD: Modular DSPy Agent Architecture"]
        direction TB
        Q2[Business Question]
        T[TriageAgent<br/>Local Rules]
        TS[TextToSemantic<br/>Template Match]
        SL[(Semantic Layer<br/>semantic.yml)]
        MR[MetricRunner<br/>SQL Compiler]
        DB[(DuckDB<br/>8 Tables)]
        H[HypothesisAgent<br/>Bootstrap CI]
        NR[NarratorAgent<br/>Constrained Output]
        O[Observability<br/>Run Record]

        Q2 -->|Classify| T
        T -->|analysis| TS
        TS -->|Semantic Request| SL
        SL -->|Compile Query| MR
        MR -->|Safe SQL| DB
        DB -->|Validated Data| H
        H -->|Projection + CI| NR
        NR -->|Decision Memo| O

        style T fill:#51cf66,stroke:#2f9e44,color:#000
        style TS fill:#51cf66,stroke:#2f9e44,color:#000
        style SL fill:#339af0,stroke:#1971c2,color:#fff
        style MR fill:#51cf66,stroke:#2f9e44,color:#000
        style H fill:#51cf66,stroke:#2f9e44,color:#000
        style NR fill:#51cf66,stroke:#2f9e44,color:#000
        style O fill:#ffd43b,stroke:#f59f00,color:#000

        V["✓ GUARANTEES:<br/>• Canonical metrics<br/>• Validated joins<br/>• Consistent windows<br/>• Quantified uncertainty<br/>• Full provenance"]
        style V fill:#fff,stroke:#2f9e44,stroke-width:2px,stroke-dasharray: 5 5
    end

    style bad fill:#fff5f5,stroke:#c92a2a,stroke-width:3px
    style good fill:#f3faf3,stroke:#2f9e44,stroke-width:3px
```

### Key Architectural Differences

| Aspect | Bad Approach | Good Approach |
|--------|-------------|---------------|
| **Data Access** | Direct LLM → SQL → Database | Semantic Layer → Compiled SQL → Database |
| **Validation** | None | Schema validation, sanity checks, inline tests |
| **Metric Definitions** | Ad-hoc in SQL | Canonical in semantic.yml |
| **Join Logic** | Uncontrolled | Enforced join rules |
| **Time Windows** | Inconsistent | Centralized defaults |
| **Uncertainty** | None | Bootstrap confidence intervals |
| **Observability** | None | Full run record with provenance |
| **Reproducibility** | Impossible | Run ID + versioned specs |
| **LLM Usage** | Heavy (SQL generation) | Minimal (ambiguity resolution only) |
| **Testing** | None | Comprehensive inline tests |

## 🧠 Decision Flow: Where LLMs Are (and Aren't) Used

```mermaid
flowchart TD
    Start([Business Question]) --> Triage{TriageAgent<br/>Decision Point}

    Triage -->|1. Try Local Rules| Keywords[Keyword Matching<br/>analysis_keywords<br/>search_keywords]
    Keywords -->|Score ≥ 2<br/>Confidence: 0.9| TriageResult[mode: analysis]
    Keywords -->|Score < 2<br/>Uncertain| LLM1{{LLM Fallback<br/>DSPy Signature}}
    LLM1 -->|Classified| TriageResult

    TriageResult --> Semantic{TextToSemantic<br/>Decision Point}

    Semantic -->|1. Try Templates| Template[Template Matching<br/>cac by channel<br/>roas by channel]
    Template -->|Exact Match<br/>Confidence: 1.0| SemanticReq[Semantic Request:<br/>metric, dims, window]
    Template -->|No Match| Pattern[Pattern Matching<br/>cac + improve/optimize]
    Pattern -->|Matched| SemanticReq
    Pattern -->|No Match| LLM2{{LLM Fallback<br/>Constrained DSPy}}
    LLM2 -->|Validate| Validate{Metric & Dims<br/>in Catalog?}
    Validate -->|✓ Valid| SemanticReq
    Validate -->|✗ Invalid| Error1[Error: Unknown<br/>metric/dimension]

    SemanticReq --> Compile[MetricRunner<br/>SQL Compilation]
    Compile -->|Deterministic| LoadSemantic[Load semantic.yml<br/>Get query template]
    LoadSemantic --> Substitute[Substitute params:<br/>window_days, limit]
    Substitute --> CompiledSQL[Compiled SQL<br/>+ Query ID]

    CompiledSQL --> Execute[Execute Query<br/>DuckDB Read-Only]
    Execute --> Sanity{Sanity Checks}
    Sanity -->|Row count OK| ResultDF[DataFrame]
    Sanity -->|Cartesian explosion| Error2[Error: Too many rows]

    ResultDF --> Hypothesis[HypothesisAgent<br/>Budget Simulation]
    Hypothesis -->|Deterministic| CalcCAC[Calculate CAC<br/>by channel]
    CalcCAC --> Identify[Identify best/worst<br/>channels]
    Identify --> Simulate[Simulate 5pp shift]
    Simulate --> Bootstrap[Bootstrap CI<br/>n=1000 samples]
    Bootstrap --> HypResult[Projected CAC<br/>+ 95% CI]

    HypResult --> Narrator{NarratorAgent<br/>Decision Point}
    Narrator -->|If Offline Mode| Template2[Template Memo<br/>with placeholders]
    Narrator -->|Else| LLM3{{LLM Generation<br/>Constrained DSPy}}
    Template2 --> Validate2[Validate memo:<br/>≤150 words<br/>References metrics]
    LLM3 --> Validate2
    Validate2 --> Memo[Decision Memo]

    Memo --> Observability[Record Everything<br/>Run ID, timings, SQL IDs]
    Observability --> Tests[Inline Tests<br/>5 test suites]
    Tests --> End([Complete])

    %% Styling
    style Keywords fill:#e7f5ff,stroke:#339af0
    style Template fill:#e7f5ff,stroke:#339af0
    style Pattern fill:#e7f5ff,stroke:#339af0
    style LoadSemantic fill:#e7f5ff,stroke:#339af0
    style Substitute fill:#e7f5ff,stroke:#339af0
    style CalcCAC fill:#e7f5ff,stroke:#339af0
    style Identify fill:#e7f5ff,stroke:#339af0
    style Simulate fill:#e7f5ff,stroke:#339af0
    style Bootstrap fill:#e7f5ff,stroke:#339af0
    style Template2 fill:#e7f5ff,stroke:#339af0
    style Observability fill:#e7f5ff,stroke:#339af0
    style Tests fill:#e7f5ff,stroke:#339af0

    style LLM1 fill:#fff3bf,stroke:#f59f00,stroke-width:3px
    style LLM2 fill:#fff3bf,stroke:#f59f00,stroke-width:3px
    style LLM3 fill:#fff3bf,stroke:#f59f00,stroke-width:3px

    style Validate fill:#fff5f5,stroke:#ff6b6b
    style Sanity fill:#fff5f5,stroke:#ff6b6b
    style Validate2 fill:#fff5f5,stroke:#ff6b6b

    style Error1 fill:#ffe3e3,stroke:#c92a2a
    style Error2 fill:#ffe3e3,stroke:#c92a2a

    style Start fill:#d3f9d8,stroke:#2f9e44
    style End fill:#d3f9d8,stroke:#2f9e44
```

### 🎯 LLM Usage Strategy: Local-First

The diagram shows **only 3 LLM decision points** (yellow) in the entire pipeline:

#### 1. **TriageAgent** (Rarely Used)
- **Primary**: Keyword matching (deterministic)
- **Fallback**: LLM only if keyword score < 2
- **Reality**: For typical questions, local rules have 0.9 confidence

#### 2. **TextToSemanticAgent** (Sometimes Used)
- **Primary**: Template matching (exact string match)
- **Secondary**: Pattern matching (e.g., "cac" + "improve")
- **Fallback**: LLM only if no template/pattern match
- **Constraint**: LLM output validated against semantic catalog
- **Reality**: Most questions hit template or pattern match

#### 3. **NarratorAgent** (Optional)
- **Primary**: LLM for natural language generation
- **Constraint**: Output validated (≤150 words, references metrics)
- **Alternative**: Offline mode uses template with placeholders
- **Reality**: Only component where LLM adds value (not logic)

### 🔷 Deterministic Components (Blue)

These components use **zero LLMs**:

- **MetricRunner**: SQL compilation from YAML templates
- **HypothesisAgent**: Statistical calculations (numpy)
- **Observability**: Logging and provenance tracking
- **Inline Tests**: Validation checks

### 🔶 Validation Gates (Red)

Every potentially risky operation has validation:

1. **Semantic validation**: Reject unknown metrics/dimensions
2. **Sanity checks**: Detect cartesian explosions
3. **Narration validation**: Check length and metric references

### 📊 LLM Usage Comparison

| Component | Bad Demo | Good Demo |
|-----------|----------|-----------|
| **SQL Generation** | 🟡 LLM (unvalidated) | 🔵 Deterministic (semantic.yml) |
| **Triage** | 🟡 LLM | 🔵 Local rules (LLM fallback) |
| **Semantic Mapping** | 🟡 LLM | 🔵 Templates (LLM fallback) |
| **Query Execution** | 🟡 LLM decides | 🔵 Deterministic |
| **Calculations** | 🟡 LLM (if any) | 🔵 Deterministic (numpy) |
| **Narration** | 🟡 LLM (unconstrained) | 🟡 LLM (constrained + validated) |

**Result**: Good demo uses LLMs in ~3% of decisions vs. 100% in bad demo.

## 🏗️ Architecture Principles (Good Demo)

1. **Semantic Layer First** - All metrics defined in `config/semantic.yml`
2. **Modular Agents** - Small, focused components with clear contracts
3. **Local-First Logic** - Deterministic rules; LLM only for ambiguity
4. **DSPy Signatures** - Structured, declarative prompts
5. **Observability** - Every decision logged with provenance

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install duckdb pandas numpy matplotlib pyyaml dspy-ai openai python-dotenv

# Ensure .env file exists with OPENAI_API_KEY
cp ../.env.example ../.env  # if needed
```

### Run the Demos

```bash
# From repo root
cd demo

# Bad demo (60-90 seconds)
jupyter notebook 01_bad_oneshot_raw.ipynb

# Good demo (3-4 minutes)
jupyter notebook 02_good_modular_dspy.ipynb
```

## 📊 Data Source

- Database: `../data/synthetic_data.duckdb` (read-only)
- Schema: 8 tables (dimensions + facts)
- Semantic layer: `../config/semantic.yml`

## 🎭 Agent Pipeline (Good Demo)

```
Question → Triage → Text-to-Semantic → Metric Compilation → Execution →
          Hypothesis Simulation → Narration → Observability
```

### Agent Roles

1. **TriageAgent** - Classify question type (search vs analysis)
2. **TextToSemanticAgent** - Map NL to semantic request
3. **MetricRunner** - Compile and execute SQL from semantic layer
4. **HypothesisAgent** - Simulate budget scenarios with bootstrap CI
5. **NarratorAgent** - Generate decision memo with constraints

## ✅ Acceptance Criteria

### Bad Demo

- [x] SQL runs but produces wrong results
- [x] Confidently wrong narrative
- [x] 3+ documented failure modes

### Good Demo

- [x] No raw LLM SQL generation
- [x] Reproducible with run_id and versioning
- [x] Quantified CAC projection with 95% CI
- [x] Concise narration with risks and next steps
- [x] All inline tests pass

## 📈 Outputs

The good demo produces:

- **Charts**: CAC by channel, ROAS by channel, hypothesis comparison
- **Run Record**: JSON with full provenance (`outputs/run_*.json`)
- **Metrics**: Query IDs, timings, row counts
- **Tests**: Schema validation, SQL compilation, sanity checks

## 🔑 Key Differences

| Bad Demo | Good Demo |
|----------|-----------|
| One-shot LLM on raw data | Modular agents with semantic layer |
| No validation | Schema validation + sanity checks + tests |
| Wrong joins, time windows | Enforced in semantic.yml |
| No confidence intervals | Bootstrap CI for projections |
| Overconfident narratives | Constrained output with risks |
| No reproducibility | Full observability with run_id |

## 📝 Notes

- **Clean-room implementation**: No imports from existing agent code
- **Local-first**: LLM used only in narrow fallback paths
- **DSPy**: Signatures define agent contracts; enables future optimization
- **Offline mode**: Set `OFFLINE_MODE=True` in good demo to disable LLM calls

## 🎤 Demo Choreography

### Bad Demo (60-90s)
1. Show raw table schemas (no guidance)
2. Run one-shot LLM prompt
3. Display overconfident but wrong results
4. Highlight failure modes

### Good Demo (3-4 mins)
1. Load semantic catalogue
2. Show triage → semantic JSON
3. Display compiled SQL IDs → charts
4. Present hypothesis result with CI
5. Show narrator memo
6. Display observability JSON

## 📚 Further Reading

- DSPy: https://github.com/stanfordnlp/dspy
- Semantic layers: https://www.getdbt.com/blog/what-is-a-semantic-layer/
- Bootstrap confidence intervals: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)

---

**Version**: 1.0.0
**Date**: November 2025
**Status**: Production-ready demonstration
