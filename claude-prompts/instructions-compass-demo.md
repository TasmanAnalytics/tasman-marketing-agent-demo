Thomas, give Claude these instructions verbatim. No reuse of any pre-existing agentic code. Build from scratch. Two notebooks under demo/. Semantic layer in config/semantic.yml. Data at data/synthetic_data.duckdb. LLM via .env (OpenAI key). Use DSPy for signatures, structured prompting, and optional optimisation. No code now; only exact tasks, contracts, and acceptance criteria.

⸻

1) Ground rules for Claude
	•	Work clean-room. Do not import, read, or adapt any pre-existing “agentic” modules in this repo.
	•	Keep everything local-first. LLM calls are permitted only in narrow fallback paths.
	•	Use DuckDB in read-only mode against data/synthetic_data.duckdb.
	•	Read .env for OPENAI_API_KEY and model. Don’t hardcode secrets.
	•	Strict separation of concerns:
	•	Bad demo: one-shot LLM on raw data, no semantics, no tests, intentional failure.
	•	Good demo: micro-agents, semantic layer, tests, observability, reproducibility, explainability.
	•	Use DSPy to define agent interfaces (signatures), keep prompts declarative, and enable later optimisation.
	•	Everything must be runnable end-to-end in the two notebooks without external scripts.

⸻

2) Deliverables overview
	•	Two Jupyter notebooks in a demo/ folder:
	•	01_bad_oneshot_raw.ipynb – intentionally brittle, “AI analyst” style.
	•	02_good_modular_dspy.ipynb – modular, semantic, testable, observable.
	•	Semantic layer YAML in config/semantic.yml:
	•	Dimensions, entities, join rules, base queries, derived metrics, defaults.
	•	Lightweight support modules (only if needed to keep notebooks readable):
	•	Minimal utilities for environment, DB connection, semantic YAML parsing, metric compilation, plotting, observability logging.
	•	Keep them tiny and obvious; the primary UX is inside the notebooks.

⸻

3) Shared scenario, data, and metrics
	•	Shared business question for both demos:
“Which channel mix change is most likely to improve CAC next month, given a recent anomaly in referral traffic?”
	•	Warehouse: data/synthetic_data.duckdb.
	•	Schema: exactly as provided (dim_campaigns, dim_adgroups, dim_creatives, dim_products, dim_customers, fact_ad_spend, fact_sessions, fact_orders).
	•	Canonical metric definitions (authoritative for the good demo; the bad demo will get these wrong by design):
	•	spend = SUM(fact_ad_spend.spend)
	•	impressions = SUM(fact_ad_spend.impressions)
	•	clicks = SUM(fact_ad_spend.clicks)
	•	ctr = clicks / NULLIF(impressions, 0)
	•	sessions = COUNT(*) FROM fact_sessions
	•	conversions = SUM(CASE WHEN converted_flag THEN 1 ELSE 0)
	•	cvr = conversions / NULLIF(sessions, 0)
	•	orders = COUNT(DISTINCT fact_orders.order_id)
	•	revenue = SUM(fact_orders.revenue)
	•	roas = revenue / NULLIF(spend, 0)
	•	cac = spend / NULLIF(conversions, 0)
	•	Time windows:
	•	Default analysis window: last 90 days.
	•	“Next month” forecast: simple proportional reweighting using the last 60–90 days, not a model.

⸻

4) Notebook 1 — “Bad one-shot on raw data” (demonstrate failure convincingly)

Purpose: Expose how a single LLM call on raw tables produces confident nonsense: wrong joins, double counting, stale windows, inconsistent dimensions, overclaiming.

Design the notebook with clear sections:
	1.	Intro / Warning
	•	One markdown cell: state this is the anti-pattern demo.
	•	List the expected failure modes you will provoke.
	2.	Raw inventory
	•	Show table list and a few columns per table, but don’t offer any semantic guidance.
	•	Emphasise that the LLM will see only table/column names.
	3.	One-shot prompt execution
	•	Single LLM call that:
	•	Reads the business question.
	•	Writes one SQL query directly against raw tables.
	•	Explains the decision in prose.
	•	Do not constrain joins, windows, or metrics.
	•	Execute the SQL and display a small preview.
	4.	Failure exhibit
	•	A markdown cell enumerating what went wrong. Make at least three concrete failures happen:
	•	Revenue attribution error: attributes revenue to dim_campaigns without the fact_orders → fact_sessions → dim_campaigns chain.
	•	Many-to-many inflation: joins fact_sessions with fact_ad_spend on campaign_id only, multiplying rows.
	•	Time window drift: runs across entire history or mismatched time windows for spend vs revenue.
	•	Metric misuse: uses orders instead of conversions in CAC, or mixes day and month grains.
	•	Dimension ambiguity: sometimes uses utm_source, sometimes dim_campaigns.channel, collapsing or duplicating channels.
	•	Show the “insight” text that is overconfident.
	5.	Post-mortem
	•	Short table mapping each failure → what the correct contract should have been (join path, metric definition, time window).
	•	End with: “This is exactly why modular agents + semantic layer are mandatory.”

Acceptance criteria for the bad demo:
	•	The generated SQL runs and returns a table, but the logic is provably flawed.
	•	The narrative confidently recommends a channel mix change based on the wrong numbers.
	•	The notebook clearly highlights at least three discrete, understandable failure modes.

⸻

5) Notebook 2 — “Good modular DSPy” (small agents, semantics, tests, logs)

Purpose: Solve the same question with a compact, production-grade architecture:
	•	Triage → Semantic mapping → Metric compilation & execution → Hypothesis simulation → Narration → Observability.
	•	Minimal LLM usage (only in fallback classification/mapping), everything else deterministic.

Organise the notebook into clean sections with headings and short explanatory text:
	1.	Bootstrap
	•	Load .env and verify presence of OPENAI_API_KEY.
	•	Connect to DuckDB in read-only mode.
	•	Validate schema (required tables and columns exist). Fail early if not.
	•	Print defaults: time window, result limit.
	2.	Semantic catalogue (from config/semantic.yml)
	•	Show dimensions, base joins, and derived metrics.
	•	Display the exact contracts for roas_by_channel and cac_by_channel.
	•	State the only accepted channel dimension is dim_campaigns.channel.
	•	State revenue attribution rule: last-touch via fact_orders → fact_sessions → dim_campaigns.
	3.	Agent architecture (DSPy style)
	•	Define the agent roles and their strict inputs/outputs (describe in text; implement later in cells):
	•	TriageAgent: question → {mode: search|analysis, confidence, reason}.
	•	Local keyword rules first; LLM fallback only if confidence < threshold.
	•	TextToSemanticAgent: NL question + role + catalogue → {metric, dimensions, filters, window_days}.
	•	Template mapping first (utterances). LLM fallback only within catalogue constraints.
	•	MetricRunner: semantic request → df + compiled_sql + elapsed_ms + rowcount.
	•	Compile from YAML; no freehand SQL from LLM.
	•	HypothesisAgent: channel metrics → candidate budget re-weight (e.g., +5pp from worst CAC to best), projected blended CAC, 95% bootstrap CI, and short method notes.
	•	NarratorAgent: inputs + result → brief decision memo and risks (≤ 150 words), references the actual metric names used.
	•	State explicitly that these are DSPy signatures with clear contracts and can be optimised later, but optimisation is not required for the demo.
	4.	Triage
	•	Run triage on the shared business question.
	•	Show the JSON result and a one-line explanation.
	•	Assert it chooses analysis (or give reason if search then branch).
	5.	Semantic mapping
	•	Attempt template mapping from a small templates.yml (include utterances like “roas by channel”, “cac by channel”).
	•	If not matched, use DSPy to constrain the mapping strictly to known metrics/dimensions; refuse unknowns.
	•	Show the semantic request JSON (metric, dims, filters, window).
	6.	Metric compilation and execution
	•	Compile two queries: cac_by_channel and roas_by_channel for the requested window.
	•	Execute safely: read-only, window enforced, limits respected.
	•	Display compact tables and clean charts (one chart per question; don’t over-style).
	•	Assert basic sanity: non-negative spend, roas only when spend > 0, channel cardinality looks reasonable.
	7.	Hypothesis simulation
	•	Compute current CAC by channel. Identify top two channels by efficiency and the “anomalous referral” if present.
	•	Simulate a small budget shift (+5 percentage points) from a weaker channel into a stronger one, compute projected blended CAC.
	•	Use a simple bootstrap to provide a 95% CI around the projected delta.
	•	Present a clear result: candidate change, estimated delta CAC, CI, caveats.
	8.	Narration
	•	Generate a succinct decision memo that:
	•	Names the metrics actually queried (“cac_by_channel_90d”, “roas_by_channel_90d”).
	•	States the proposed mix change and the confidence interval.
	•	Lists two risks (data quality, attribution choice, seasonality) and two next steps (small controlled budget shift; monitor CAC over 14 days).
	9.	Observability
	•	Emit a single run JSON with:
	•	run_id (timestamp + short hash),
	•	triage decision,
	•	semantic request,
	•	IDs of the compiled SQL blocks,
	•	execution timings, row counts,
	•	hypothesis parameters and CI result,
	•	paths to saved artefacts (tables/charts),
	•	versions (prompt config version, templates version, semantic spec hash).
	•	Print it, and save it next to the notebook outputs.
	10.	Inline tests / smoke checks

	•	Schema presence and column checks.
	•	SQL compiles for both derived metrics.
	•	No cartesian blow-up: check that channel counts are sane and row counts align pre/post join.
	•	Triage accuracy: run on 4–6 canned queries; assert ≥ a set threshold.
	•	Narration lint: length bound, references at least one metric ID.

Acceptance criteria for the good demo:
	•	No raw LLM SQL generation; all data access goes through semantic compilation.
	•	Reproducible outputs with logged run_id, SQL IDs, and YAML semantic hash.
	•	The hypothesis step produces a quantified projected CAC change with a CI.
	•	The narration is concise, references real metric IDs, and includes risks + next steps.
	•	All inline tests pass in a clean state.

⸻

6) Semantic layer YAML (content requirements)
	•	Defaults: window_days, limit.
	•	Dimensions: expose channel, campaign_name, device, region. Each maps to a fully qualified source.
	•	Entities: list keys for campaign, adgroup, creative, customer, session, order (helps future extensions).
	•	Base joins/queries (named blocks) that are safe on their own:
	•	Spend by channel (fact_ad_spend → dim_campaigns).
	•	Conversions by channel (fact_sessions → dim_campaigns).
	•	Revenue by channel last-touch (fact_orders → fact_sessions → dim_campaigns).
	•	Impressions/clicks by channel (fact_ad_spend → dim_campaigns).
	•	Derived metrics:
	•	roas_by_channel built from spend+revenue blocks.
	•	cac_by_channel built from spend+conversions blocks.
	•	Compilation rules: allow only {{metric:...}}, {{window_days}}, {{limit}} substitutions. No arbitrary templating.

⸻

7) DSPy usage (practical minimum)
	•	Define DSPy signatures for Triage, Text-to-Semantic, Narrator.
	•	For Triage and Text-to-Semantic, prefer rule/template paths and only fallback to DSPy when ambiguous.
	•	Keep temperature low. Responses must conform to the signature; reject if not.
	•	Do not use DSPy to write SQL. Use it to map or narrate.
	•	Expose a simple switch to disable LLM (offline mode) for rehearsals.

⸻

8) Observability expectations
	•	One JSON “run record” per execution in the good demo, printed and saved.
	•	Include environment info (model name), timing, rows, semantic spec hash, compiled SQL IDs, and file artefacts.
	•	Keep the structure stable so it can be inspected or parsed later.

⸻

9) Visual output baseline
	•	Use matplotlib defaults (single chart per result).
	•	“ROAS by channel” → sorted bar chart.
	•	“CAC by channel” → sorted bar chart; highlight the two channels used in the mix simulation.
	•	Hypothesis chart: tiny annotation showing the proposed +5pp reweight and projected blended CAC with CI band (text annotation is enough).

⸻

10) Guardrails and refusals
	•	If a requested metric is not defined in the semantic YAML, refuse cleanly with a short error pointing to the missing definition.
	•	If SQL compilation would create an ambiguous join path, refuse and point to the needed rule in semantics.
	•	If LLM mapping proposes an unknown metric/dimension, refuse and show the allowed list.

⸻

11) Demo choreography on stage
	•	Bad demo (60–90s): run the one-shot cell, show table head and the overconfident narrative, then flash the failure bullets.
	•	Good demo (3–4 mins):
	•	print catalogue,
	•	triage → semantic JSON,
	•	compiled SQL ids → charts,
	•	hypothesis result with CI,
	•	narrator memo,
	•	observability JSON.
	•	Always tie each step back to the “software rules” slide: small, testable, semantic, observable.

⸻

12) Final acceptance checklist
	•	Both notebooks run start-to-finish with the provided DuckDB file and .env.
	•	The bad demo produces plausible but wrong output and explains why it is wrong.
	•	The good demo produces consistent CAC/ROAS by channel aligned with semantic definitions, a quantified mix recommendation with CI, and a single, well-formed observability JSON.
	•	No references to any pre-existing agentic code.
	•	DSPy is present and used for signatures and constrained fallbacks.
	•	All inline tests pass in the good demo.

Deliver exactly this.