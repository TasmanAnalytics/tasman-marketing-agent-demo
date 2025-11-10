# DSPy Best Practices: What Good Looks Like

This document explains what makes effective DSPy usage and how to build production-grade LLM agents.

## Why DSPy? The Problem It Solves

**Traditional Prompting:**
```python
# ❌ BAD: String-based prompts are fragile
prompt = f"Given this data: {data}, tell me which channel is best"
response = llm.complete(prompt)
# No type safety, no optimization, no validation
```

**DSPy Signatures:**
```python
# ✅ GOOD: Structured contracts with types and validation
class AnalysisSignature(dspy.Signature):
    """Analyze channel performance data."""
    data: str = dspy.InputField(desc="JSON data with metrics")
    best_channel: str = dspy.OutputField(desc="Channel name")
    confidence: float = dspy.OutputField(desc="0-1 score")
    reasoning: str = dspy.OutputField(desc="Why this channel")
```

## The Five Principles of Good DSPy

### 1. Clear Input/Output Contracts

**❌ BAD:**
```python
class VagueSignature(dspy.Signature):
    question: str
    answer: str  # What type? What format? What constraints?
```

**✅ GOOD:**
```python
class ClearSignature(dspy.Signature):
    """Classify marketing question type."""

    question: str = dspy.InputField(
        desc="User's business question about marketing metrics"
    )

    mode: str = dspy.OutputField(
        desc="Either 'search' or 'analysis' - nothing else allowed"
    )

    confidence: float = dspy.OutputField(
        desc="Confidence score between 0 and 1"
    )

    reason: str = dspy.OutputField(
        desc="One sentence explaining the classification"
    )
```

**Why it's good:**
- Each field has a clear description
- Output types are explicit
- Constraints are documented
- LLM knows exactly what's expected

### 2. Local-First, LLM-Fallback Architecture

**❌ BAD: LLM for everything:**
```python
def classify_question(question):
    # Always use LLM, even for obvious cases
    return llm.classify(question)
```

**✅ GOOD: Try deterministic first:**
```python
def classify_question(question):
    # Try keyword matching
    if any(kw in question.lower() for kw in ['cac', 'optimize', 'improve']):
        return {
            'mode': 'analysis',
            'confidence': 0.9,
            'method': 'keyword_match'
        }

    # Only fallback to LLM if ambiguous
    return llm_classifier(question)
```

**Why it's good:**
- Faster (no API call for obvious cases)
- Cheaper (no LLM cost for 80% of queries)
- More reliable (deterministic for clear cases)
- Observable (know which path was taken)

### 3. Validation Gates After Every LLM Call

**❌ BAD: Trust LLM output blindly:**
```python
result = predictor(question=q, metrics=metrics)
# Assume result.metric is valid
query = compile_query(result.metric)
```

**✅ GOOD: Validate against known constraints:**
```python
result = predictor(question=q, metrics=metrics)

# Validate metric exists
if result.metric not in valid_metrics:
    raise ValueError(f"LLM proposed invalid metric: {result.metric}")

# Validate dimensions exist
for dim in result.dimensions:
    if dim not in valid_dimensions:
        raise ValueError(f"LLM proposed invalid dimension: {dim}")

# Now safe to use
query = compile_query(result.metric)
```

**Why it's good:**
- LLMs can hallucinate invalid values
- Validation prevents downstream errors
- Fails fast with clear error messages
- Maintains data integrity

### 4. Structured, Constrained Prompts

**❌ BAD: Open-ended prompts:**
```python
prompt = "Analyze this data and tell me something interesting"
# LLM can return anything - no structure, no constraints
```

**✅ GOOD: Specific instructions with format:**
```python
class HypothesisSignature(dspy.Signature):
    """
    Propose a budget reallocation hypothesis.

    CONSTRAINTS:
    - Shift must be between 3-10%
    - Must cite specific numbers from data
    - Must identify best and worst performers
    - Format: "Shift X% from [channel] to [channel]"
    """
    cac_data: str = dspy.InputField(desc="CAC by channel as JSON")
    hypothesis: str = dspy.OutputField(desc="Reallocation in specified format")
    expected_impact: str = dspy.OutputField(desc="Dollar impact estimate")
```

**Why it's good:**
- Clear constraints reduce hallucination
- Specified format enables parsing
- Explicit requirements in docstring
- Structured outputs enable validation

### 5. Observable Decision Points

**❌ BAD: Black box decisions:**
```python
result = agent(question)
# No idea if LLM was used, what confidence, what method
```

**✅ GOOD: Log every decision:**
```python
result = agent(question)
# Returns: {
#   'answer': '...',
#   'method': 'template_match',  # or 'llm_fallback'
#   'confidence': 0.95,
#   'reasoning': 'Matched template "cac by channel"'
# }

observability.record({
    'agent': 'triage',
    'method': result['method'],
    'confidence': result['confidence'],
    'llm_used': result['method'] == 'llm_fallback'
})
```

**Why it's good:**
- Know exactly when LLMs are used
- Can audit decisions later
- Debug failures easily
- Measure LLM usage costs

## Complete Example: Good vs Bad

### ❌ BAD: Unstructured LLM Query Generation

```python
# No structure, no validation, no fallback
def get_query(question):
    prompt = f"Write SQL for: {question}"
    sql = llm.complete(prompt)
    return execute(sql)  # Hope it works!
```

**Problems:**
- No validation of SQL
- No semantic layer
- No error handling
- Can't test or reproduce
- No way to know what went wrong

### ✅ GOOD: Structured Pipeline with DSPy

```python
class SemanticMappingSignature(dspy.Signature):
    """Map NL question to semantic request."""
    question: str = dspy.InputField(desc="User question")
    available_metrics: str = dspy.InputField(desc="Valid metrics")
    metric: str = dspy.OutputField(desc="Selected metric name")
    dimensions: str = dspy.OutputField(desc="Dimensions to group by")

class SemanticMapper:
    def __init__(self, semantic_layer):
        self.semantic = semantic_layer
        self.predictor = dspy.Predict(SemanticMappingSignature)

        # Try templates first
        self.templates = {
            'cac by channel': ('cac_by_channel', ['channel']),
            'roas by channel': ('roas_by_channel', ['channel'])
        }

    def __call__(self, question):
        # 1. Try template matching (deterministic)
        for template, (metric, dims) in self.templates.items():
            if template in question.lower():
                return {
                    'metric': metric,
                    'dimensions': dims,
                    'method': 'template'
                }

        # 2. Fallback to LLM (structured)
        available = ', '.join(self.semantic.list_metrics())
        result = self.predictor(
            question=question,
            available_metrics=available
        )

        # 3. Validate (safety gate)
        if result.metric not in self.semantic.list_metrics():
            raise ValueError(f"Invalid metric: {result.metric}")

        return {
            'metric': result.metric,
            'dimensions': result.dimensions.split(','),
            'method': 'llm'
        }

# 4. Compile SQL (deterministic, from semantic layer)
query = semantic_layer.compile(request['metric'])

# 5. Execute with observability
result = execute_with_logging(query)
```

**Why it's good:**
- Clear signature defines contract
- Template matching tries deterministic first
- LLM fallback is constrained and validated
- Semantic layer ensures correct SQL
- Observable at every step
- Testable components
- Reproducible results

## When to Use LLMs (and When Not To)

### ✅ GOOD Use Cases for LLMs:

1. **Ambiguous Classification**
   - "Is this a search or analysis question?"
   - After trying keyword rules first

2. **Natural Language Mapping**
   - Map "show me CAC" → `cac_by_channel`
   - After trying template matching first

3. **Natural Language Generation**
   - Generate narrative summaries
   - With strict length and format constraints

### ❌ BAD Use Cases for LLMs:

1. **SQL Generation**
   - Use semantic layer instead
   - SQL should come from tested templates

2. **Mathematical Calculations**
   - Use numpy/pandas
   - LLMs are bad at math

3. **Data Validation**
   - Use explicit rules
   - LLMs can't reliably validate constraints

4. **Join Logic**
   - Use semantic layer
   - Joins are too complex and error-prone

## Testing DSPy Agents

### Unit Tests
```python
def test_triage_agent():
    agent = TriageAgent()

    # Test keyword path (deterministic)
    result = agent("What is the CAC by channel?")
    assert result['mode'] == 'analysis'
    assert result['method'] == 'keyword_match'

    # Test ambiguous case (LLM fallback)
    result = agent("Tell me about performance")
    assert result['method'] == 'llm_fallback'
    assert 'confidence' in result
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    question = "Which channel should I optimize?"

    # Triage
    triage = triage_agent(question)
    assert triage['mode'] == 'analysis'

    # Semantic mapping
    request = semantic_agent(question)
    assert request['metric'] in valid_metrics

    # Query compilation
    query = semantic.compile(request)
    assert 'SELECT' in query['sql']

    # Execution
    result = execute(query)
    assert len(result) > 0
```

## Key Takeaways

1. **DSPy signatures are contracts**, not just prompts
2. **Try deterministic first**, LLM as fallback
3. **Validate everything** from LLMs
4. **Structure inputs and outputs** explicitly
5. **Make decisions observable** and loggable
6. **Test each component** independently
7. **Use LLMs for ambiguity**, not logic
8. **Keep prompts declarative** and constraint-focused

## Further Reading

- DSPy Documentation: https://github.com/stanfordnlp/dspy
- This implementation: See `02_good_modular_dspy.ipynb`
- Hypothesis Playbook: See `utils/hypothesis_testing_playbook.py`
