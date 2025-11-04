
⸻

Phase 2 — Deep Analytics Agents

Cross-cutting rules (apply to all agents)
	•	Local-first: All transforms/statistics in Python/DuckDB. LLM only to: (a) choose agent/playbook, (b) explain results.
	•	Result-set policy:
	•	Aggregations push down in SQL (no LIMIT).
	•	Row-level fetches: max_rows default 250k (configurable), with stratified sampling (by channel/device/region) if above cap.
	•	Always return row_count, sample_ratio, and data_spec (columns, types).
	•	Reproducibility: Each agent outputs a JSON plan + versioned seed + SQL used.
	•	Stats hygiene: 95% CI by default, multiple-test control via Benjamini–Hochberg (FDR 0.1) where applicable.
	•	Notebook sections: Inputs → Data pulls → Checks → Model/Stats → Diagnostics → Findings → Caveats → Next actions.

⸻

1) HypothesisTestingAgent

Purpose: Test targeted assertions (e.g., “LinkedIn CTR > Facebook CTR in last 90 days”, “CPO wants to know if margin differs by brand”).

Typical queries on this dataset
	•	CTR by campaign/channel/placement (from fact_ad_spend + dim_campaigns)
	•	CVR by device/region (from fact_sessions)
	•	Margin/Revenue by brand/category (from fact_orders + dim_products)

SQL inputs (examples)
	•	CTR comparison (channel):

SELECT c.channel,
       SUM(f.clicks) AS clicks,
       SUM(f.impressions) AS impressions
FROM fact_ad_spend f
JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
WHERE f.date >= CURRENT_DATE - INTERVAL {{days}} DAY
GROUP BY 1;

	•	CVR comparison (device):

SELECT s.device,
       COUNT(*) AS sessions,
       SUM(CASE WHEN s.converted_flag THEN 1 ELSE 0 END) AS conversions
FROM fact_sessions s
WHERE s.date >= CURRENT_DATE - INTERVAL {{days}} DAY
GROUP BY 1;

Analysis methods
	•	Proportions tests (CTR/CVR): 2-sample z-test or χ²; multiple arms via χ² + post-hoc pairwise with FDR.
	•	Means tests (AOV): Welch t-test or one-way ANOVA if continuous (e.g., AOV on order revenue).
	•	Effects: absolute & relative lift, 95% CI, power heuristics.

System prompt (Claude)

You are the Hypothesis Testing Agent. Convert a natural-language hypothesis into a concrete test plan over the provided schema, selecting the correct statistical test (z/χ²/t/ANOVA), defining groups, metric, time window, and covariates if needed. Output a minimal JSON plan and never run code.

User prompt template

Role: {role}
Hypothesis: {text}
Context: {business_context_excerpt}
Available schema: {schema_json_excerpt}
Return: JSON with {metric, groups, window_days, test_type, tails, alpha, corrections, SQL_specs}

Playbook (steps)
	1.	Parse hypothesis → metric & groups.
	2.	Build SQL aggregations (pushdown); validate counts > threshold.
	3.	Choose test; compute effect size & CI.
	4.	Correct for multiplicity if k>2.
	5.	Diagnostics: sample sizes, variance, assumptions.
	6.	Produce conclusion + caveats.

Outputs

{
  "plan": {...},
  "sql_blocks": ["..."],
  "results": {"p_value":0.023,"effect":{"abs":0.012,"rel":0.07},"ci":[0.003,0.021]},
  "diagnostics":{"n":[...],"assumptions":["..."]},
  "next_actions":["..."]
}


⸻

2) DriverAnalysisAgent

Purpose: Identify drivers of an outcome (CVR, orders, revenue, ROAS).

Typical queries
	•	What drives conversion at session level? (fact_sessions, join dims where available)
	•	What drives margin/revenue per order? (fact_orders + dim_products + optional campaign via session join)
	•	What explains ROAS differences by channel? (joined spend + revenue)

SQL inputs
	•	Session-level feature table:

SELECT
  s.session_id,
  s.device,
  c.channel,
  c.campaign_name,
  s.pages_viewed,
  s.converted_flag::INT AS y
FROM fact_sessions s
LEFT JOIN dim_campaigns c ON s.campaign_id = c.campaign_id
WHERE s.date >= CURRENT_DATE - INTERVAL {{days}} DAY;

Analysis methods
	•	Binary outcome: Logistic regression (L2). Report ORs & 95% CI.
	•	Continuous: Lasso/Ridge linear regression, feature importances, partial dependence.
	•	Categorical features: one-hot with k-limit (top K + “other”).
	•	Interactions optional (channel×device).
	•	Robustness: k-fold CV; report out-of-fold metrics (AUC/R²/MAE).

System prompt

You are the Driver Analysis Agent. Given a target metric and allowed features, produce a driver plan: target definition, feature list, encoding rules, data window, leakage checks, and evaluation metrics. Output JSON only; do not run code.

Playbook
	1.	Define target/aggregation level (session/order/channel-day).
	2.	Select features from schema; cap high-cardinality; prevent leakage.
	3.	Build SQL to assemble feature table; enforce row cap via stratified sampling if needed.
	4.	Fit model; compute importances with CIs (bootstrapped).
	5.	Diagnostics: ROC/AUC or R²; stability across folds.
	6.	Sorted driver table + guidance.

Outputs

{"plan": {...}, "sql":"...", "metrics":{"auc":0.71}, "drivers":[{"feature":"device=mobile","odds_ratio":1.24,"p":0.01}, ...]}


⸻

3) SegmentationAgent

Purpose: Create actionable customer or campaign segments.

Typical queries
	•	Customer segments by region/behaviour/value: aggregate orders per customer.
	•	Campaign segments: group campaigns by efficiency (ROAS/CTR/CVR profiles).

SQL inputs
	•	Customer aggregates:

WITH cust AS (
  SELECT o.customer_id,
         COUNT(DISTINCT o.order_id) AS orders,
         SUM(o.revenue) AS revenue,
         SUM(o.margin)  AS margin
  FROM fact_orders o
  WHERE o.order_timestamp >= NOW() - INTERVAL {{days}} DAY
  GROUP BY 1
),
acq AS (
  SELECT customer_id, acquisition_channel, region
  FROM dim_customers
)
SELECT *
FROM cust LEFT JOIN acq USING(customer_id);

Analysis methods
	•	K-means with k selection via silhouette; standardise features.
	•	Rule-based quick segments (e.g., RFM quantiles).
	•	Optional DBSCAN for anomaly cluster.
	•	Output segment profiles + size + value contribution.

System prompt

You are the Segmentation Agent. Propose a segmentation plan (RFM or k-means), define features, normalisation, k-selection, and stability checks. Return JSON only.

Playbook
	1.	Choose segmentation type (RFM vs k-means) from question.
	2.	Build aggregate table in SQL; cap/clip outliers (winsorise).
	3.	Fit segments; label and profile.
	4.	Validate: silhouette, segment stability (bootstrap).
	5.	Output segment dictionary + business actions.

Outputs

{"plan": {...}, "sql":"...", "segments":[{"name":"High value loyal","n":1240,"share_margin":0.41,"profile":{"R":0.8,"F":0.9,"M":0.95}}]}


⸻

4) CohortAndRetentionAgent

Purpose: First-visit cohorts; measure retention, repeat orders, revenue per cohort.

Typical queries
	•	Monthly cohorts based on dim_customers.first_visit_date, track orders over months since acquisition.

SQL inputs

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
  GROUP BY 1,2
)
SELECT c.cohort_month,
       DATE_DIFF('month', c.cohort_month, o.order_month) AS month_offset,
       SUM(o.orders) AS orders,
       SUM(o.revenue) AS revenue,
       COUNT(DISTINCT o.customer_id) AS active_customers
FROM cohorts c
LEFT JOIN orders o USING(customer_id)
GROUP BY 1,2
ORDER BY 1,2;

Analysis methods
	•	Retention curves (% active), repeat-purchase curves, revenue per retained customer.
	•	Compare cohorts; detect significant deltas.
	•	Simple lifetime proxies (area under retention curve).

System prompt

You are the Cohort & Retention Agent. Produce a cohort plan: cohort key, window, metrics (retention%, orders, revenue), normalisation choices, and comparison tests. Return JSON only.

Playbook
	1.	Define cohort key & horizon (e.g., 6/12 months).
	2.	Build cohort matrix; compute retention & revenue curves.
	3.	Compare latest vs baseline cohorts; significance on means/proportions.
	4.	Output insights + recommended levers.

Outputs

{"plan": {...}, "sql":"...", "curves":[...], "findings":["Retention -6.2pp for May cohort vs baseline"], "actions":["Review onboarding for region=..."]}


⸻

5) AttributionAndROASAgent (Simple Heuristics)

Purpose: Attribute revenue to channels/campaigns and explain ROAS variance.

Typical queries
	•	Last-touch attribution via session→order join; ROAS by channel/campaign.
	•	Heuristic multi-touch: equal split across sessions/touches (if available via UTM in fact_sessions).

SQL inputs
	•	Last-touch:

WITH rev AS (
  SELECT s.campaign_id,
         SUM(o.revenue) AS revenue
  FROM fact_orders o
  JOIN fact_sessions s ON o.session_id = s.session_id
  WHERE o.order_timestamp >= NOW() - INTERVAL {{days}} DAY
  GROUP BY 1
),
spend AS (
  SELECT campaign_id, SUM(spend) AS spend
  FROM fact_ad_spend
  WHERE date >= CURRENT_DATE - INTERVAL {{days}} DAY
  GROUP BY 1
)
SELECT c.channel, c.campaign_name, spend.spend, rev.revenue,
       CASE WHEN spend.spend=0 THEN NULL ELSE rev.revenue::DOUBLE/spend.spend END AS roas
FROM dim_campaigns c
LEFT JOIN spend USING(campaign_id)
LEFT JOIN rev USING(campaign_id)
ORDER BY roas DESC NULLS LAST;

Analysis methods
	•	Decompose ROAS variance: between-channel vs within-channel (ANOVA on ROAS).
	•	Sensitivity to time window & outliers.
	•	Optional constrained regression (budget → revenue).

System prompt

You are the Attribution & ROAS Agent. Define the attribution rule, time window, unit of analysis, and variance decomposition steps. Output JSON only.

Playbook
	1.	Choose rule (last-touch default).
	2.	Assemble spend+revenue table; compute ROAS and CIs (bootstrap).
	3.	Variance decomposition; identify top contributors.
	4.	Sensitivity check (±30 days).
	5.	Insights + trade-offs.

Outputs

{"plan": {...}, "sql":"...", "table":[...], "variance":{"between_channel":0.62}, "insights":["LinkedIn ROAS driven by low spend days..."]}


⸻

6) AnomalyAndTrendBreakAgent

Purpose: Detect spikes/dips or structural breaks (spend, CTR, CVR, orders, revenue).

Typical queries
	•	“Flag abnormal drops in CVR last week”, “Find change-points in spend”.

SQL inputs

SELECT f.date,
       SUM(f.spend) AS spend,
       SUM(f.impressions) AS impressions,
       SUM(f.clicks) AS clicks
FROM fact_ad_spend f
WHERE f.date >= CURRENT_DATE - INTERVAL {{days}} DAY
GROUP BY 1
ORDER BY 1;

(similarly for orders/revenue time series)

Analysis methods
	•	Z-score/robust STL residuals for anomalies.
	•	Change-points: PELT/binary segmentation on mean shift; confirm with pre/post tests.
	•	Output ranked anomalies & breaks with severity.

System prompt

You are the Anomaly & Trend Break Agent. Propose metrics, decomposition method, thresholds, and confirmatory tests. Return JSON only.

Playbook
	1.	Build daily metric series.
	2.	Decompose (trend/seasonal), score residuals; pick anomalies.
	3.	Change-point detection; pre/post significance.
	4.	Summarise incidents with likely causes (campaigns/regions).

Outputs

{"plan": {...}, "sql":"...", "anomalies":[{"date":"2025-10-12","metric":"CVR","z":-3.1}], "breaks":[{"date":"2025-09-01","metric":"spend"}]}


⸻

LLM Prompts (shared scaffolds)

Analysis Plan Builder (orchestrator for Phase 2)

System

You are the Analysis Plan Builder. Given a user question, role, and schema context, select one of the six analysis agents and emit a minimal JSON plan that includes: agent_id, target metric, unit of analysis, time window, required SQL pulls (names + brief purpose), tests/models to run, and expected outputs. Do not write code.

User

Role: {role}
Question: {question}
Schema excerpt: {schema_json_excerpt}
Business context: {biz_excerpt}

Output JSON

{"agent_id":"driver_analysis","target":"converted_flag","unit":"session","window_days":90,"sql_blocks":[...],"methods":[...],"notes":"..."}


⸻

Query-size policy (replacing LIMIT=1000)
	•	Aggregations: never limited; compute at source.
	•	Row-level modelling:
	•	default max_rows = 250_000; sampling strategy: stratified by key dims (channel, device, region).
	•	Report sample_ratio, strata_distribution.
	•	Time windows: agent-specific defaults (e.g., 90 days for tests; 180 for drivers; 365 for cohorts).
	•	Config: config/analysis.yaml

row_caps:
  modelling_max_rows: 250000
  segmentation_max_rows: 200000
windows:
  hypothesis_days: 90
  driver_days: 180
  cohort_months: 12


⸻

Phase 2 — Claude Execution Plan (Playbook)

Scope
	•	Add six agents above (plans + local code).
	•	Add Analysis Plan Builder that selects the right agent.
	•	Extend notebook with “Analysis Mode” section.

Deliverables
	1.	agents/agent_analysis_plan.py — emits JSON plan from question.
	2.	agents/ new files:
	•	agent_hypothesis.py, agent_driver.py, agent_segmentation.py, agent_cohort.py, agent_attribution.py, agent_anomaly.py
	3.	core/analysis_utils.py — sampling, encoding, BH-FDR, CI, bootstraps.
	4.	config/analysis.yaml — row caps, windows, defaults.
	5.	Notebook section: Deep Analytics with 6 runnable demos.
	6.	Tests: unit tests per agent + golden-file plans.

Packages (uv add)
	•	scikit-learn, statsmodels, ruptures (or changepoint alternative), scipy

Implementation steps (for Claude)
	1.	Scaffold configs (config/analysis.yaml) and utilities (analysis_utils.py: sampling, winsorising, encoders, CIs).
	2.	Agent skeletons: implement plan() -> JSON, pull() -> DataFrame, run() -> results, report() -> dict/markdown.
	3.	HypothesisTestingAgent: proportions & t/ANOVA + BH-FDR; tests + fixtures.
	4.	DriverAnalysisAgent: logistic/linear w/ k-fold CV; feature caps; leakage checks.
	5.	SegmentationAgent: RFM + k-means; silhouette; profiles.
	6.	CohortAndRetentionAgent: cohort matrix; retention/revenue curves; comparisons.
	7.	AttributionAndROASAgent: last-touch tables; variance decomposition; sensitivity windows.
	8.	AnomalyAndTrendBreakAgent: STL + z-scores + PELT; confirmations.
	9.	Analysis Plan Builder: map intents→agent; produce plan JSON; fall back to LLM only if ambiguous.
	10.	Notebook updates: a cell per agent (inputs → run → charts → insights).
	11.	Testing: per-agent unit tests (plans, SQL blocks exist; basic correctness on small fixtures).
	12.	Docs: update README/Spec to v0.2.0; add “Analysis Mode” section and examples.

Acceptance tests
	•	For a sample of 2–3 real questions per agent:
	•	No LLM call once templates/plans exist (local only).
	•	End-to-end run < 3s on 250k rows.
	•	Outputs include JSON plan, SQL, diagnostics, result tables, PNG charts, and Next actions.
