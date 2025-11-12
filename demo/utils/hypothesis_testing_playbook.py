"""
Hypothesis testing playbook - strict instructions for LLM-based hypothesis analysis.

This module demonstrates how to use LLMs effectively for analytical reasoning
with clear constraints, validation, and structured outputs.
"""

import dspy
from typing import Dict, Any, List
import pandas as pd


# =============================================================================
# BEST PRACTICE: Clear, Constrained DSPy Signatures
# =============================================================================

class HypothesisAnalysisSignature(dspy.Signature):
    """
    Analyze channel performance and propose budget reallocation hypothesis.

    ✅ WHAT MAKES THIS SIGNATURE GOOD:

    1. **Clear Input/Output Contract**: Each field has explicit type and description
    2. **Constrained Output Format**: JSON structure prevents hallucination
    3. **Domain Constraints**: Specifies valid ranges and requirements
    4. **Validation Rules**: Explicit criteria for what makes a good hypothesis
    5. **Grounded in Data**: Requires citing specific metrics from input

    This is a DSPy signature, not just a prompt. It defines the contract
    between the LLM and our system, enabling:
    - Type checking
    - Automated prompt optimization (if we run DSPy optimization)
    - Clear documentation of expectations
    - Consistent behavior across model changes
    """

    # Inputs with clear descriptions
    cac_data: str = dspy.InputField(
        desc="CAC by channel as JSON: [{channel, cac, spend, conversions}]"
    )
    roas_data: str = dspy.InputField(
        desc="ROAS by channel as JSON: [{channel, roas, revenue, spend}]"
    )
    business_context: str = dspy.InputField(
        desc="Business question and any known anomalies"
    )

    # Structured outputs with validation criteria
    analysis: str = dspy.OutputField(
        desc="Brief analysis (max 3 sentences) identifying best and worst performing channels with specific numbers"
    )
    hypothesis: str = dspy.OutputField(
        desc="Proposed budget reallocation in format: 'Shift X% from [channel] to [channel]' where X is between 3-10"
    )
    expected_impact: str = dspy.OutputField(
        desc="Expected CAC change in format: 'Projected CAC: $X.XX (current: $Y.YY, delta: $Z.ZZ)'"
    )
    confidence_factors: str = dspy.OutputField(
        desc="2-3 factors affecting confidence (e.g., data quality, seasonality, sample size)"
    )
    risks: str = dspy.OutputField(
        desc="2-3 specific risks in bullet format"
    )


# =============================================================================
# PLAYBOOK: Strict Instructions for Hypothesis Testing
# =============================================================================

HYPOTHESIS_TESTING_PLAYBOOK = """
# Hypothesis Testing Playbook for Marketing Budget Optimization

## Objective
Analyze channel performance data and propose a testable budget reallocation hypothesis
that is likely to improve customer acquisition cost (CAC).

## Input Data Constraints
You will receive:
1. CAC by channel (cost per conversion)
2. ROAS by channel (return on ad spend)
3. Business context (question and known anomalies)

## Analysis Requirements

### Step 1: Identify Best and Worst Performers
- BEST performers: Channels with LOW CAC and HIGH ROAS
- WORST performers: Channels with HIGH CAC and LOW ROAS
- ALWAYS cite specific numbers from the data
- DO NOT invent or assume data not provided

### Step 2: Propose Hypothesis
Your hypothesis MUST:
- Shift budget FROM worst performer TO best performer
- Specify exact percentage (between 3% and 10%)
- Follow format: "Shift X% from [channel] to [channel]"
- Be testable and reversible

INVALID hypotheses:
- ❌ "Increase budget overall" (not a reallocation)
- ❌ "Shift 50% from X to Y" (too aggressive)
- ❌ "Maybe try optimizing" (not specific)

VALID hypotheses:
- ✅ "Shift 5% from display to email"
- ✅ "Reallocate 7% from search to referral"

### Step 3: Calculate Expected Impact
- Use weighted average formula: blended_CAC = Σ(weight_i × CAC_i)
- Show current blended CAC
- Calculate projected blended CAC after shift
- Report delta (improvement should be negative)

### Step 4: State Confidence Factors
Factors that INCREASE confidence:
- Large sample size (many conversions)
- Consistent performance over time
- Clear performance gap between channels
- Known data quality

Factors that DECREASE confidence:
- Small sample size
- Recent anomalies (e.g., "recent anomaly in referral traffic")
- Seasonality concerns
- Data quality issues

### Step 5: Identify Risks
Always include 2-3 specific risks:
- Data quality risks (attribution, measurement)
- Market risks (seasonality, competition)
- Execution risks (implementation challenges)

## Output Format

Your response must be structured with these exact fields:
1. analysis: Brief summary with specific numbers
2. hypothesis: Specific reallocation statement
3. expected_impact: Current CAC, projected CAC, delta
4. confidence_factors: 2-3 factors affecting confidence
5. risks: 2-3 specific risks

## Validation Rules

Before returning your analysis:
1. ✓ Check that hypothesis cites channels from the data
2. ✓ Verify percentage is between 3-10
3. ✓ Confirm expected_impact includes specific dollar amounts
4. ✓ Ensure analysis references actual data points
5. ✓ Verify risks are specific, not generic

## Example Output

{
  "analysis": "Email has the lowest CAC at $15.32 with strong ROAS of 5.2x, while display shows highest CAC at $65.18 with weak ROAS of 1.9x. This 4.2x CAC gap suggests clear reallocation opportunity.",
  "hypothesis": "Shift 5% from display to email",
  "expected_impact": "Projected CAC: $41.23 (current: $43.87, delta: -$2.64)",
  "confidence_factors": "High confidence due to: (1) Large sample size (1000+ conversions per channel), (2) Consistent 90-day performance trend, (3) Clear 4x performance gap",
  "risks": "• Attribution model may not capture full customer journey\n• Display may have brand-building effects not measured in direct CAC\n• Email performance may not scale with increased budget"
}

## Critical Don'ts

❌ DO NOT invent data points not in the input
❌ DO NOT propose shifts larger than 10%
❌ DO NOT use vague language ("might", "could", "possibly")
❌ DO NOT ignore anomalies mentioned in business context
❌ DO NOT skip validation checks

## Remember

This hypothesis will be:
1. Reviewed by analysts
2. Tested with bootstrap confidence intervals
3. Implemented with real budget
4. Monitored for 2-4 weeks

Be specific. Be conservative. Be honest about uncertainty.
"""


class LLMHypothesisAgent:
    """
    LLM-based hypothesis agent with strict playbook constraints.

    ✅ WHAT MAKES THIS IMPLEMENTATION GOOD:

    1. **Playbook-Driven**: LLM receives explicit instructions (not just a vague prompt)
    2. **Structured I/O**: Uses DSPy signatures for type safety
    3. **Validation**: Every output is validated against rules
    4. **Fallback**: Can fall back to deterministic method if LLM fails
    5. **Observable**: Logs method used (LLM vs deterministic)

    This demonstrates the RIGHT way to use LLMs for analytical tasks:
    - Clear instructions
    - Structured outputs
    - Validation gates
    - Graceful degradation
    """

    def __init__(self, deterministic_fallback: bool = True):
        """
        Initialize hypothesis agent.

        Args:
            deterministic_fallback: If True, fall back to deterministic calculation
                                   if LLM output fails validation
        """
        self.deterministic_fallback = deterministic_fallback
        self.predictor = dspy.Predict(HypothesisAnalysisSignature)

    def __call__(
        self,
        cac_df: pd.DataFrame,
        roas_df: pd.DataFrame,
        business_question: str
    ) -> Dict[str, Any]:
        """
        Generate hypothesis using LLM with strict playbook.

        Args:
            cac_df: DataFrame with columns [channel, cac, spend, conversions]
            roas_df: DataFrame with columns [channel, roas, revenue, spend]
            business_question: Business context

        Returns:
            Dictionary with hypothesis analysis
        """
        # Prepare structured inputs
        cac_data = cac_df.to_json(orient='records')
        roas_data = roas_df.to_json(orient='records')

        # Construct context with playbook
        business_context = f"{business_question}\n\nPlaybook Instructions:\n{HYPOTHESIS_TESTING_PLAYBOOK}"

        try:
            # Call LLM with strict signature
            result = self.predictor(
                cac_data=cac_data,
                roas_data=roas_data,
                business_context=business_context
            )

            # Extract and validate
            hypothesis_result = {
                'analysis': result.analysis,
                'hypothesis': result.hypothesis,
                'expected_impact': result.expected_impact,
                'confidence_factors': result.confidence_factors,
                'risks': result.risks,
                'method': 'llm_playbook'
            }

            # Validate hypothesis format
            if not self._validate_hypothesis(hypothesis_result, cac_df):
                if self.deterministic_fallback:
                    return self._deterministic_fallback(cac_df, roas_df, business_question)
                else:
                    raise ValueError("LLM hypothesis failed validation and fallback disabled")

            return hypothesis_result

        except Exception as e:
            if self.deterministic_fallback:
                return self._deterministic_fallback(cac_df, roas_df, business_question)
            else:
                raise

    def _validate_hypothesis(self, result: Dict[str, Any], cac_df: pd.DataFrame) -> bool:
        """
        Validate LLM output against playbook rules.

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['analysis', 'hypothesis', 'expected_impact', 'confidence_factors', 'risks']
        if not all(field in result for field in required_fields):
            return False

        # Check hypothesis mentions channels from data
        channels = set(cac_df['channel'].str.lower())
        hypothesis_lower = result['hypothesis'].lower()

        # Should mention at least 2 channels (from and to)
        mentioned_channels = sum(1 for ch in channels if ch in hypothesis_lower)
        if mentioned_channels < 2:
            return False

        # Check for percentage mention (3-10%)
        import re
        percentage_match = re.search(r'(\d+)%', result['hypothesis'])
        if not percentage_match:
            return False

        percentage = int(percentage_match.group(1))
        if not (3 <= percentage <= 10):
            return False

        # Check expected_impact has dollar amounts
        if '$' not in result['expected_impact']:
            return False

        return True

    def _deterministic_fallback(
        self,
        cac_df: pd.DataFrame,
        roas_df: pd.DataFrame,
        business_question: str
    ) -> Dict[str, Any]:
        """
        Fallback to deterministic hypothesis generation.

        This ensures the system never fails even if LLM output is invalid.
        """
        # Sort by CAC to find best/worst
        sorted_cac = cac_df.sort_values('cac')
        best_channel = sorted_cac.iloc[0]
        worst_channel = sorted_cac.iloc[-1]

        # Calculate current blended CAC
        total_spend = cac_df['spend'].sum()
        current_cac = (cac_df['spend'] * cac_df['cac']).sum() / total_spend

        # Simulate 5% shift
        shift_pct = 5.0

        return {
            'analysis': f"{best_channel['channel']} has lowest CAC at ${best_channel['cac']:.2f}, while {worst_channel['channel']} has highest at ${worst_channel['cac']:.2f}.",
            'hypothesis': f"Shift {shift_pct}% from {worst_channel['channel']} to {best_channel['channel']}",
            'expected_impact': f"Current blended CAC: ${current_cac:.2f} (deterministic calculation)",
            'confidence_factors': "Deterministic fallback used due to LLM validation failure",
            'risks': "• LLM analysis unavailable\n• Using simplified deterministic model\n• May miss nuanced factors",
            'method': 'deterministic_fallback'
        }


def get_playbook() -> str:
    """Return the hypothesis testing playbook for reference."""
    return HYPOTHESIS_TESTING_PLAYBOOK
