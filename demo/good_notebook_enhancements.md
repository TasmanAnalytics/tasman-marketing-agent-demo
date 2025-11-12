# Enhancements for Good Notebook

This document describes the additions to `02_good_modular_dspy.ipynb` for better DSPy explanations and LLM hypothesis testing.

## Enhancement 1: DSPy Best Practices Section

Insert after cell that prints "✓ Agent implementations complete" and before "Initialize agents".

### Markdown Cell: "What Makes Good DSPy?"

```markdown
## 3a. 📚 DSPy Best Practices: What Good Looks Like

Before we run these agents, let's understand **what makes them production-grade**.

### The Five Principles

#### 1️⃣ **Clear Input/Output Contracts**

Our signatures explicitly define types and descriptions:

\`\`\`python
class TriageSignature(dspy.Signature):
    """Classify a user question as search or analysis."""  # ← Clear purpose
    question: str = dspy.InputField(desc="User's business question")  # ← Input contract
    mode: str = dspy.OutputField(desc="Either 'search' or 'analysis'")  # ← Output contract
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")  # ← Structured output
\`\`\`

**Why this matters:**
- LLM knows exactly what's expected
- Type safety (string, float, etc.)
- Enables automated testing
- Self-documenting code

**Compare to naive prompting:**
```python
# ❌ BAD: No structure
prompt = "Is this a search or analysis question?"
# What if LLM returns "maybe" or "it depends"?
```

#### 2️⃣ **Local-First, LLM-Fallback Architecture**

Notice our TriageAgent tries keyword matching FIRST:

\`\`\`python
# Try local rules first (deterministic, fast, cheap)
analysis_score = sum(1 for kw in self.analysis_keywords if kw in question_lower)

if analysis_score >= 2:
    return {'mode': 'analysis', 'method': 'local_rules', 'confidence': 0.9}

# Only fallback to LLM if ambiguous
result = self.predictor(question=question)
\`\`\`

**Why this matters:**
- **Faster**: No API call for 80% of queries
- **Cheaper**: No LLM cost for clear cases
- **More reliable**: Deterministic for obvious questions
- **Observable**: Know which path was taken

**In practice:** Most questions hit local rules, LLM only for edge cases.

#### 3️⃣ **Validation Gates After Every LLM Call**

Look at TextToSemanticAgent - after LLM fallback:

\`\`\`python
result = self.predictor(question=question, available_metrics=available_metrics)

# VALIDATE against semantic layer
metric = result.metric.strip()
if metric not in self.semantic.list_available_metrics():
    raise ValueError(f"LLM proposed unknown metric '{metric}'")
\`\`\`

**Why this matters:**
- LLMs can hallucinate invalid values
- Validation prevents downstream errors
- Fails fast with clear error messages
- Maintains data integrity

**Never trust LLM output blindly** - always validate against known constraints.

#### 4️⃣ **Structured, Constrained Prompts**

See how TextToSemanticSignature includes available options:

\`\`\`python
available_metrics: str = dspy.InputField(desc="List of available metrics")
available_dimensions: str = dspy.InputField(desc="List of available dimensions")
\`\`\`

Then we pass the actual catalog:
\`\`\`python
result = self.predictor(
    question=question,
    available_metrics=', '.join(self.semantic.list_available_metrics()),
    available_dimensions=', '.join(self.semantic.get_dimension_names())
)
\`\`\`

**Why this matters:**
- Constrains LLM to valid options
- Reduces hallucination
- Makes validation easier
- Grounds output in real data

#### 5️⃣ **Observable Decision Points**

Every agent returns method and confidence:

\`\`\`python
return {
    'mode': 'analysis',
    'confidence': 0.9,
    'reason': 'Matched 4 analysis keywords',
    'method': 'local_rules'  # ← Know exactly how decision was made
}
\`\`\`

**Why this matters:**
- Debug failures easily
- Measure LLM usage and costs
- Audit decisions later
- Optimize based on what works

### What We DON'T Use LLMs For

Notice what's **deterministic** (no LLM):

1. **SQL Generation** → MetricRunner uses semantic.yml templates
2. **Mathematical Calculations** → HypothesisAgent uses numpy for bootstrap CI
3. **Data Validation** → Explicit rules, not LLM judgment
4. **Join Logic** → Enforced in semantic layer

**Key Insight:** Use LLMs for **ambiguity resolution** (NL understanding), not **logical operations** (SQL, math, validation).

### When to Use LLMs (and When Not To)

| Task | Use LLM? | Why / Why Not |
|------|----------|---------------|
| Classify ambiguous question | ✅ After local rules | Good: Natural language understanding |
| Map "show me CAC" → metric | ✅ After templates | Good: Flexible interpretation |
| Generate SQL | ❌ Never | Bad: Error-prone, use semantic layer |
| Calculate CAC | ❌ Never | Bad: LLMs are bad at math |
| Validate join paths | ❌ Never | Bad: Too complex, use rules |
| Generate narrative | ✅ Yes | Good: Natural language generation |

### Testing DSPy Agents

Our inline tests (see section 10) validate:

1. **Triage accuracy**: Test on 4 canned queries, expect ≥75% accuracy
2. **Semantic validation**: Reject unknown metrics/dimensions
3. **Narration constraints**: Check length and metric references

**This is testable because** we have clear contracts and structured outputs.

### Key Takeaways

✅ **DSPy signatures are contracts**, not just prompts
✅ **Try deterministic first**, LLM as fallback
✅ **Validate everything** from LLMs
✅ **Structure inputs and outputs** explicitly
✅ **Make decisions observable** and loggable
✅ **Use LLMs for ambiguity**, not logic

**See `utils/dspy_best_practices.md` for detailed examples.**

Now let's run these agents and see these principles in action...
```

## Enhancement 2: LLM Hypothesis Testing

Insert after the deterministic hypothesis section (after cell 24) and before narration (before cell 26).

### Markdown Cell: "Advanced: LLM-Based Hypothesis Analysis"

```markdown
## 7a. Advanced: LLM-Based Hypothesis Analysis with Playbook

The bootstrap simulation above was **deterministic** (pure math, no LLM).

But what if we want the LLM to help analyze the data and propose hypotheses?

**The Challenge:** LLMs are bad at math but good at pattern recognition and reasoning.

**The Solution:** Give the LLM:
1. Pre-calculated data (we did the math)
2. Strict instructions (a "playbook")
3. Validation gates (check the output)
4. Fallback to deterministic (if LLM fails)

Let's demonstrate this with our `LLMHypothesisAgent` that uses a **hypothesis testing playbook**.
```

### Code Cell: Load and show playbook

```python
# Import the LLM hypothesis agent
from utils.hypothesis_testing_playbook import LLMHypothesisAgent, get_playbook

# Show the playbook (truncated for space)
playbook = get_playbook()
print("HYPOTHESIS TESTING PLAYBOOK (excerpt):")
print("=" * 80)
print(playbook[:1500] + "\n\n[... see utils/hypothesis_testing_playbook.py for full text ...]")
print("=" * 80)
```

### Markdown Cell: Explain the playbook approach

```markdown
### What's in the Playbook?

The playbook contains:

1. **Clear Objective**: "Analyze channel performance and propose testable hypothesis"

2. **Input Constraints**: What data format to expect

3. **Analysis Requirements**:
   - Step 1: Identify best/worst performers (cite specific numbers)
   - Step 2: Propose hypothesis (3-10% reallocation)
   - Step 3: Calculate expected impact (weighted average formula)
   - Step 4: State confidence factors
   - Step 5: Identify risks

4. **Output Format**: Exact structure required

5. **Validation Rules**: What makes a good vs bad hypothesis

6. **Examples**: Show valid and invalid outputs

**This is what "good prompting" looks like** - not vague instructions, but a complete playbook.
```

### Code Cell: Run LLM hypothesis agent

```python
# Initialize LLM hypothesis agent with fallback
llm_hypothesis_agent = LLMHypothesisAgent(deterministic_fallback=True)

print("Running LLM hypothesis analysis with strict playbook...\n")

# Pass the pre-calculated data to LLM
llm_hypothesis_result = llm_hypothesis_agent(
    cac_df=cac_df,
    roas_df=roas_df,
    business_question=business_question
)

print("LLM Hypothesis Analysis:")
print("=" * 80)
for key, value in llm_hypothesis_result.items():
    if key != 'method':
        print(f"\n{key.upper()}:")
        print(f"  {value}")

print("\n" + "=" * 80)
print(f"Method used: {llm_hypothesis_result['method']}")
print("=" * 80)
```

### Markdown Cell: Compare approaches

```markdown
### Comparison: Deterministic vs LLM Hypothesis Testing

| Aspect | Deterministic (Bootstrap) | LLM (with Playbook) |
|--------|--------------------------|---------------------|
| **Analysis** | Automatic (sort by CAC) | Interprets patterns |
| **Hypothesis** | Best vs worst channel | Considers context |
| **Math** | NumPy bootstrap CI | Provided data |
| **Reasoning** | None (pure calculation) | Explains rationale |
| **Confidence** | Statistical (95% CI) | Qualitative factors |
| **Risks** | None | Lists specific risks |
| **Speed** | Very fast | Slower (API call) |
| **Cost** | Free | Small LLM cost |
| **Reliability** | 100% deterministic | Needs validation |

### When to Use Which?

**Use Deterministic when:**
- Need exact reproducibility
- Speed is critical
- Cost is a concern
- Data patterns are clear

**Use LLM when:**
- Want natural language reasoning
- Need to consider context (e.g., "recent anomaly in referral")
- Benefit from risk analysis
- Value qualitative factors

**Best Practice:** Use both!
- Deterministic for calculations
- LLM for interpretation and narrative

### Key Differences from "Bad" LLM Usage

Compare this to the bad demo:

| Bad Demo | This LLM Usage |
|----------|----------------|
| ❌ Vague prompt | ✅ Detailed playbook |
| ❌ LLM does SQL + math | ✅ LLM gets pre-calculated data |
| ❌ No validation | ✅ Validates output format |
| ❌ No fallback | ✅ Falls back to deterministic |
| ❌ Can't test | ✅ Testable with examples |
| ❌ Not observable | ✅ Logs method used |

**The LLM is a component, not the whole system.**
```

## Enhancement 3: Update Run Record

Add after the LLM hypothesis section:

### Code Cell: Record LLM hypothesis

```python
# Record both hypothesis results in observability
run_record.record_hypothesis(
    params={
        'deterministic_hypothesis': hypothesis_result,
        'llm_hypothesis': llm_hypothesis_result
    },
    result=llm_hypothesis_result  # Use LLM result for the record
)

print("✓ Both hypothesis approaches recorded in observability")
```

## Summary of Changes

These enhancements add:

1. **Clear DSPy education**: Explains the 5 principles with examples from our code
2. **LLM hypothesis testing**: Shows how to use LLMs correctly with playbooks
3. **Comparative analysis**: Shows when to use deterministic vs LLM approaches
4. **Best practices reinforcement**: Contrasts with the bad demo throughout

The notebook now teaches **HOW** to use DSPy effectively, not just shows the code.
