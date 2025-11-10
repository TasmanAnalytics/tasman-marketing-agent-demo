# Additional Example for Bad Notebook: Vague Acceptance Criteria

This content should be inserted after the first example in `01_bad_oneshot_raw.ipynb`.

## New Section: Example 2 - Vague Acceptance Criteria

### Markdown Cell

```markdown
## Example 2: Vague Request with No Acceptance Criteria

Let's try another anti-pattern: asking the LLM to "analyze" data without:
- Specific questions
- Clear success criteria
- Defined scope
- Expected output format
- Constraints on what to look for

This is common in practice: analysts ask LLMs to "find something interesting" or "analyze our data".
```

### Code Cell 1

```python
# Ask a vague, open-ended question
vague_question = """Analyze our marketing data and tell me something interesting that could help improve performance."""

vague_prompt = f"""{schema_description}

Task: {vague_question}

Look at our marketing data and provide insights. Write SQL queries as needed and explain what you found.
"""

# Call LLM with no constraints
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": vague_prompt}],
    temperature=0.7
)

vague_output = response.choices[0].message.content
print("LLM Response to Vague Request:")
print("=" * 80)
print(vague_output)
```

### Markdown Cell

```markdown
### What's Wrong with This Approach?

Even if the SQL executes, this approach fails because:

1. **No Clear Goal**: "Something interesting" is subjective - interesting to who? For what purpose?

2. **No Validation Criteria**: How do we know if the insight is:
   - Correct?
   - Actionable?
   - Novel?
   - Relevant to business goals?

3. **No Scope**: The LLM might:
   - Look at irrelevant metrics
   - Use wrong time windows
   - Mix incompatible data
   - Generate obvious or useless insights

4. **No Output Format**: The response could be:
   - A single number without context
   - A complex visualization we can't implement
   - Multiple contradictory insights
   - Generic advice that doesn't use our data

5. **Not Reproducible**: Running this again with the same data might produce completely different "insights"

6. **Not Testable**: How do you write a test for "tell me something interesting"?
```

### Code Cell 2

```python
# Demonstrate the problems
print("\n" + "=" * 80)
print("PROBLEMS WITH VAGUE REQUESTS")
print("=" * 80)

problems = [
    {
        'problem': 'No Clear Success Criteria',
        'example': 'LLM says "Snapchat performs well" - is that interesting? Actionable? True?',
        'impact': 'Cannot validate if the insight is useful or correct'
    },
    {
        'problem': 'Generic, Obvious Insights',
        'example': 'LLM might say "Channels with lower CAC are more efficient" (duh!)',
        'impact': 'Waste time on insights that provide no value'
    },
    {
        'problem': 'Mixing Time Windows',
        'example': 'Compares last month Snapchat to last year Google Search',
        'impact': 'Invalid comparisons that mislead decision-making'
    },
    {
        'problem': 'No Prioritization',
        'example': 'LLM lists 10 "interesting" things without ranking importance',
        'impact': 'Cannot decide which insight to act on first'
    },
    {
        'problem': 'Hallucinated Patterns',
        'example': 'LLM "finds" a trend that is just random noise',
        'impact': 'False insights lead to wrong business decisions'
    }
]

for i, p in enumerate(problems, 1):
    print(f"\n{i}. {p['problem']}")
    print(f"   Example: {p['example']}")
    print(f"   Impact: {p['impact']}")

print("\n" + "=" * 80)
print("WHAT GOOD LOOKS LIKE")
print("=" * 80)

print("""
Instead of vague requests, use:

✅ SPECIFIC QUESTION:
   "Which channel mix change is most likely to improve CAC next month?"

✅ CLEAR CONSTRAINTS:
   - Time window: Last 90 days
   - Metrics: CAC, ROAS by channel
   - Output: Specific reallocation percentage
   - Success criteria: Quantified CAC improvement with confidence interval

✅ VALIDATION:
   - Check that channels mentioned exist in data
   - Verify that percentage is reasonable (3-10%)
   - Ensure output includes specific dollar amounts
   - Validate that math is correct

✅ STRUCTURED OUTPUT:
   - Hypothesis: "Shift X% from [channel] to [channel]"
   - Expected impact: "$X.XX improvement"
   - Confidence: "95% CI: [$Y, $Z]"
   - Risks: 2-3 specific risks

✅ REPRODUCIBLE:
   - Same question → same answer (or within CI)
   - Documented methodology
   - Versioned semantic definitions
   - Logged decision process

See 02_good_modular_dspy.ipynb for this approach in action.
""")
```

### Markdown Cell

```markdown
### Comparison: Vague vs Specific

| Aspect | Vague Request | Specific Request (Good Demo) |
|--------|--------------|------------------------------|
| **Question** | "Tell me something interesting" | "Which channel mix change improves CAC?" |
| **Success Criteria** | ??? | Quantified CAC improvement with CI |
| **Scope** | Entire database, any metric | CAC and ROAS by channel, 90 days |
| **Output Format** | Freeform text | Structured JSON with hypothesis |
| **Validation** | Impossible | Check against semantic catalog |
| **Actionability** | Maybe? | Clear: shift X% budget from A to B |
| **Reproducibility** | No | Yes (run_id + versioned specs) |
| **Testability** | No | Yes (inline tests) |

**Key Insight**: The specificity of the request determines the quality of the output.

Vague inputs → vague outputs, even with perfect LLMs.
Specific, constrained inputs → actionable, validated outputs.
```

## Summary of Changes

This example demonstrates:

1. **What happens with vague acceptance criteria**: Generic, unvalidated, non-actionable output
2. **Why specificity matters**: Clear constraints lead to better results
3. **The contrast**: Shows the problem, then points to the good demo as the solution

This reinforces the key message: **It's not just about using LLMs vs not using them - it's about HOW you use them.**
