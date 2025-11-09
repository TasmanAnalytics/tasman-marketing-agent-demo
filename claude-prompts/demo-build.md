Awesome — here’s a tight, stage-ready spec for a 5-minute, notebook-first demo that sits on top of your current agentic stack. It assumes your DuckDB + config files already exist. Copy/paste friendly, minimal ceremony, zero fluff.

⸻

Five-Minute Demo — Agentic Marketing Analytics (Notebook)

Goal (what the audience sees)

Take a plain-English marketing question → triage → text-to-SQL (template-first) → run in DuckDB → plot → 2-sentence summary, with visible guardrails (semantic/model layer only, tests, observability). No raw tables, no open-ended prompts.

⸻

Run-of-Show (T+00:00 → 05:00)
	•	00:00–00:30 — Set the scene: “Agents are software. We’ll run a micro-agent path end-to-end.”
	•	00:30–01:30 — Inputs & triage: Type a question; show rule-based triage (“search” vs “analysis”).
	•	01:30–02:45 — Template mapping & SQL: Show template hit, render SQL bound to the metrics/semantic layer only.
	•	02:45–04:15 — Execute & visualise: Run SQL, display a chart, and print a crisp summary.
	•	04:15–05:00 — Guardrails & observability: Show logs (routing, template chosen, timings). Show a graceful failure (remove metric -> friendly error).

⸻

Pre-flight (do before going on stage)
	•	config/db.yaml points to your DuckDB (read-only).
	•	config/schema.json, config/business_context.yaml, config/sql_templates.yaml present (as per your stack).
	•	Two questions tested locally (they should hit templates so the live demo avoids LLM calls):
	•	Q1 (primary): “Show ROAS by channel for the last 90 days and plot it.”
	•	Q2 (backup): “Show spend by channel over time (last 90 days).”
	•	Optional: one failure case ready (temporarily comment out roas in business_context.yaml → calculated_measures).

⸻

Notebook structure (cells you’ll show)

File: notebooks/Agentic_5min_Demo.ipynb

0) Bootstrap & validation (Cell 1)

Loads env + configs, validates schema, opens DuckDB (read-only).

from pathlib import Path
import os, json, yaml, time
import duckdb, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parents[0] if Path.cwd().name=="notebooks" else Path.cwd()
CFG  = ROOT/"config"
DB   = yaml.safe_load(open(CFG/"db.yaml"))
SCHEMA = json.load(open(CFG/"schema.json"))
BIZ   = yaml.safe_load(open(CFG/"business_context.yaml"))
TMPL  = yaml.safe_load(open(CFG/"sql_templates.yaml"))

con = duckdb.connect(database=str(Path(DB["duckdb_path"]).resolve()), read_only=True)

def _tables(): return [r[0] for r in con.execute("SHOW TABLES").fetchall()]
def _cols(t): return {r[1]: r[2] for r in con.execute(f"PRAGMA table_info({t})").fetchall()}

missing=[]
for t, cols in SCHEMA.items():
    if t not in _tables(): missing.append(f"Missing table {t}"); continue
    act=_cols(t)
    for c in cols.keys():
        if c not in act: missing.append(f"Missing column {t}.{c}")
assert not missing, "Schema check failed:\n- "+"\n- ".join(missing)
print("✅ Schema OK, DB connected (read-only)")

1) Helpers (Cell 2)

Minimal helpers kept inline (portable for 5-min).

DEFAULTS = {
    "limit": DB.get("default_limit", 1000),
    "window": BIZ.get("defaults", {}).get("time_window_days", 90)
}

def ensure_limit(sql, lim=DEFAULTS["limit"]):
    s=sql.strip().rstrip(";")
    return s if " limit " in s.lower() else f"{s}\nLIMIT {lim};"

PROHIBITED = ("insert ","update ","delete ","create ","alter ","drop ","truncate ","merge ")
def read_only_ok(sql):
    s=sql.strip().lower()
    return s.startswith("select ") and not any(k in s for k in PROHIBITED)

def triage_local(question:str):
    q=question.lower()
    search_kw = ("what","show","list","plot","how much","how many","roas","spend","ctr","cvr","orders","revenue")
    analysis_kw=("why","driver","impact","hypothesis","test","causal","effect")
    score = 0.6 if any(k in q for k in search_kw) else 0.2
    if any(k in q for k in analysis_kw): score = 0.3
    mode = "search" if score>=0.5 else "analysis"
    return {"mode":mode,"confidence":round(score,2),"reason":"keyword rules"}

def match_template(question:str, role="marketer", window_days=None):
    window_days = window_days or DEFAULTS["window"]
    q=question.lower()
    for t in TMPL:
        if t.get("role_hint")==role:
            for utt in t.get("utterances",[]):
                if utt in q:
                    sql=t["sql"].replace("{{time_window_days}}", str(window_days)).replace("{{limit}}", str(DEFAULTS["limit"]))
                    return {"id":t["id"], "sql":sql, "source":"template"}
    return None

def run_sql(sql:str):
    assert read_only_ok(sql), "Blocked non-SELECT SQL"
    sql=ensure_limit(sql)
    t0=time.time()
    df=con.execute(sql).df()
    return df, {"rows":len(df), "ms":int((time.time()-t0)*1000)}

2) Question input (Cell 3)

This is the only thing you type live.

user_role     = "marketer"
user_question = "Show ROAS by channel for the last 90 days and plot it"

3) Agent flow (Cell 4)

Triage → template → SQL → run → visualise → summarise → observability.

OBS = {}

# triage
tri = triage_local(user_question)
OBS["triage"] = tri
print("🔎 Triage ->", tri)

assert tri["mode"]=="search", "Demo path is 'search'; choose a search-style question."

# template mapping
mt = match_template(user_question, role=user_role)
assert mt, "No template matched; pick the backup question."
print("🧩 Template ->", mt["id"])

# execute
df, exec_meta = run_sql(mt["sql"])
OBS["exec"] = exec_meta
display(df.head(8))

# viz
assert df.shape[0]>0, "Empty result; adjust time window or use backup question."
fig = plt.figure(figsize=(7,4))
if {"channel","roas"} <= set(c.lower() for c in df.columns):
    # Normalize columns for robustness
    cols = {c.lower(): c for c in df.columns}
    x = df[cols.get("channel")]
    y = df[cols.get("roas")]
    plt.bar(x, y)
    plt.title("ROAS by Channel (last 90 days)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
elif {"date","total_spend"} <= set(c.lower() for c in df.columns):
    cols = {c.lower(): c for c in df.columns}
    plt.plot(df[cols.get("date")], df[cols.get("total_spend")])
    plt.title("Spend by Channel Over Time")
    plt.tight_layout()
else:
    plt.plot(df.iloc[:,0], df.iloc[:, -1]); plt.title("Auto plot")
plt.show()

# summary (2 lines max)
def short_summary(d: pd.DataFrame):
    lc = {c.lower():c for c in d.columns}
    if "roas" in lc and "channel" in lc:
        top = d.sort_values(lc["roas"], ascending=False).head(1)
        ch = top[lc["channel"]].iloc[0]
        rv = float(top[lc["roas"]].iloc[0])
        return f"Top ROAS channel is **{ch}** at ~{rv:.2f}. Consider shifting a small test budget towards it with guardrails."
    if "total_spend" in lc and "channel" in lc:
        top = d.sort_values(lc["total_spend"], ascending=False).head(1)
        return f"Highest recent spend: **{top[lc['channel']].iloc[0]}**."
    return "Result summarised. See chart."

print("🧠 Summary:", short_summary(df))

OBS["sql"] = mt["sql"]
OBS

4) Guardrail demo (Cell 5, optional 20 seconds)

Temporarily break a metric (comment out roas measure in config and re-load), then re-run to show safe failure.

try:
    # Simulate missing metric (you’ll only run this if you changed config live)
    assert "roas" in (BIZ.get("calculated_measures") or {}), "Simulated: roas measure missing"
except AssertionError as e:
    print("🛑 Guardrail:", "Metric 'roas' undefined in semantic layer. Refusing to run. Add the measure or choose another template.")


⸻

What you’ll say (talk track, ~60 seconds of total narration)
	•	Open (10s): “Agents = tiny, testable functions. We’ll go from a plain question to an answer via micro-agents.”
	•	Triage (10s): “First we classify: this is a search question — we treat retrieval differently from analysis.”
	•	Templates (15s): “We match to a known template bound to the semantic layer (metrics, not raw tables).”
	•	Execute & plot (15s): “SQL runs in DuckDB; we auto-plot and produce a short, reproducible summary.”
	•	Guardrails (10s): “If a metric is undefined, we fail closed with a friendly error rather than hallucinate.”

⸻

Acceptance criteria (for your rehearsal)
	•	No LLM calls on the happy path (the template matches).
	•	SELECT-only SQL with LIMIT injected where appropriate.
	•	Columns referenced exist in schema.json; joins only through the semantic/model layer.
	•	Chart renders in < 1s; summary is ≤ 2 sentences.
	•	OBS log shows: mode, template_id, rows, ms, and the exact SQL used.
	•	Guardrail path prints a clear refusal when a metric is missing.

⸻

Templates you’ll rely on (confirm in config/sql_templates.yaml)
	1.	roas_by_channel (primary)

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
       spend.spend, rev.revenue,
       CASE WHEN spend.spend=0 OR spend.spend IS NULL THEN NULL
            ELSE CAST(rev.revenue AS DOUBLE)/spend.spend END AS roas
FROM spend FULL OUTER JOIN rev USING(channel)
ORDER BY roas DESC NULLS LAST
LIMIT {{limit}};

	2.	spend_by_channel_over_time (backup)

SELECT c.channel, f.date, SUM(f.spend) AS total_spend
FROM fact_ad_spend f
JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
GROUP BY 1,2
ORDER BY 2 ASC, 1
LIMIT {{limit}};


⸻

Observability payload (what you’ll show on screen)

A small dict at the end of Cell 4:

{
  "triage": {"mode":"search","confidence":0.60,"reason":"keyword rules"},
  "exec": {"rows": 8, "ms": 212},
  "sql": "WITH spend AS (... ) SELECT ... ORDER BY roas DESC LIMIT 1000;"
}


⸻

Graceful failure (one-liner you’ll say)

“If the semantic layer doesn’t define ROAS, we don’t guess. The agent refuses, tells you what’s missing, and how to fix it. That’s the difference between a demo and a system.”

⸻

Optional (30-second extension if you have time)

Add a tiny toggle at the top:

DEMO_MODE = "search"  # or "analysis"

If analysis, print: “This would hand off to the Hypothesis Testing Agent next (pre/post or CTR comparisons).” No execution — just show the branching.

⸻

If you want, I can turn this into a ready-to-run .ipynb with your colours (cream BG cells, code font sizing) and drop in a one-page demo checklist slide you can flash before running it.