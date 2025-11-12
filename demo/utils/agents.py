"""
Production-grade agent implementations using DSPy.

All agents follow these principles:
1. Local-first logic (deterministic rules before LLM)
2. Validation gates after every LLM call
3. Observable decision points (method, confidence, reasoning)
4. Structured inputs/outputs via DSPy signatures
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import dspy


# ==================== AGENT 1: TRIAGE ====================

class TriageAgent:
    """Classify questions as search or analysis using DSPy."""

    def __init__(self, signature):
        """
        Args:
            signature: DSPy TriageSignature class
        """
        self.predictor = dspy.Predict(signature)

        # Local keyword rules for fast path
        self.analysis_keywords = [
            'cac', 'roas', 'improve', 'optimize', 'compare', 'which',
            'recommend', 'should', 'best', 'worst', 'trend', 'anomaly'
        ]
        self.search_keywords = [
            'what is', 'show me', 'list', 'find', 'get', 'display'
        ]

    def __call__(self, question: str) -> Dict[str, Any]:
        question_lower = question.lower()

        # Try local rules first (faster, cheaper)
        analysis_score = sum(1 for kw in self.analysis_keywords if kw in question_lower)
        search_score = sum(1 for kw in self.search_keywords if kw in question_lower)

        if analysis_score > search_score and analysis_score >= 2:
            return {
                'mode': 'analysis',
                'confidence': 0.9,
                'reason': f'Matched {analysis_score} analysis keywords',
                'method': 'local_rules'
            }
        elif search_score > analysis_score and search_score >= 2:
            return {
                'mode': 'search',
                'confidence': 0.9,
                'reason': f'Matched {search_score} search keywords',
                'method': 'local_rules'
            }

        # DSPy fallback for ambiguous cases
        print("  → Using DSPy TriageSignature for ambiguous question")
        result = self.predictor(question=question)
        return {
            'mode': result.mode,
            'confidence': float(result.confidence),
            'reason': result.reason,
            'method': 'dspy_fallback'
        }


# ==================== AGENT 2: TEXT-TO-SEMANTIC ====================

class TextToSemanticAgent:
    """Map natural language to semantic request using DSPy."""

    def __init__(self, semantic_layer, signature):
        """
        Args:
            semantic_layer: SemanticLayer instance
            signature: DSPy TextToSemanticSignature class
        """
        self.semantic = semantic_layer
        self.predictor = dspy.Predict(signature)

    def __call__(self, question: str) -> Dict[str, Any]:
        # Use DSPy with constrained inputs
        available_metrics = ', '.join(self.semantic.list_available_metrics())
        available_dimensions = ', '.join(self.semantic.get_dimension_names())

        print(f"  → Using DSPy TextToSemanticSignature")
        print(f"     Available metrics: {available_metrics}")
        print(f"     Available dimensions: {available_dimensions}")

        result = self.predictor(
            question=question,
            available_metrics=available_metrics,
            available_dimensions=available_dimensions
        )

        # Validate against semantic layer
        metric = result.metric.strip()
        if metric not in self.semantic.list_available_metrics():
            raise ValueError(
                f"❌ DSPy proposed unknown metric '{metric}'. "
                f"Available: {available_metrics}"
            )

        dimensions = [d.strip() for d in result.dimensions.split(',')]
        for dim in dimensions:
            if not self.semantic.validate_dimension(dim):
                raise ValueError(
                    f"❌ DSPy proposed unknown dimension '{dim}'. "
                    f"Available: {available_dimensions}"
                )

        print(f"  ✓ DSPy output validated: {metric} by {dimensions}")

        return {
            'metric': metric,
            'dimensions': dimensions,
            'filters': {},
            'window_days': int(result.window_days),
            'method': 'dspy_constrained'
        }


# ==================== AGENT 3: METRIC RUNNER ====================

class MetricRunner:
    """Compile and execute queries from semantic layer (deterministic, no LLM)."""

    def __init__(self, db_conn, semantic_layer):
        """
        Args:
            db_conn: Database connection
            semantic_layer: SemanticLayer instance
        """
        self.conn = db_conn
        self.semantic = semantic_layer

    def __call__(self, metric: str, window_days: int, limit: int = 1000) -> Dict[str, Any]:
        # Compile query from semantic layer (no LLM)
        query_info = self.semantic.compile_query(metric, window_days, limit)

        # Execute with timing
        start_time = time.time()
        result_df = self.conn.execute(query_info['sql']).df()
        elapsed_ms = (time.time() - start_time) * 1000

        return {
            'query_info': query_info,
            'df': result_df,
            'elapsed_ms': elapsed_ms,
            'row_count': len(result_df)
        }


# ==================== AGENT 4: HYPOTHESIS ====================

class HypothesisAgent:
    """Analyze channel performance and propose budget shifts using DSPy."""

    def __init__(self, signature, n_bootstrap: int = 1000):
        """
        Args:
            signature: DSPy HypothesisSignature class
            n_bootstrap: Number of bootstrap samples for CI
        """
        self.n_bootstrap = n_bootstrap
        self.predictor = dspy.Predict(signature)

    def __call__(self, cac_df: pd.DataFrame, roas_df: pd.DataFrame = None,
                 question: str = "", spend_col: str = 'spend',
                 cac_col: str = 'cac', channel_col: str = 'channel') -> Dict[str, Any]:
        """
        Use DSPy to analyze channel data and propose budget reallocation.
        Then validate with bootstrap confidence intervals.
        """
        # Prepare data strings for DSPy
        cac_data_str = cac_df.to_string(index=False)
        roas_data_str = roas_df.to_string(index=False) if roas_df is not None else "N/A"

        # Use DSPy to analyze and propose hypothesis
        print("  → Using DSPy HypothesisSignature to analyze channels")
        print(f"     Analyzing {len(cac_df)} channels...")

        result = self.predictor(
            question=question,
            cac_data=cac_data_str,
            roas_data=roas_data_str
        )

        best_channel = result.best_channel.strip()
        worst_channel = result.worst_channel.strip()
        shift_percentage = float(result.shift_percentage)

        print(f"  ✓ DSPy recommendation: Shift {shift_percentage}% from {worst_channel} to {best_channel}")
        print(f"     Reasoning: {result.reasoning}")

        # Validate DSPy output against actual data
        valid_df = cac_df[cac_df[cac_col].notna()].copy()
        valid_channels = set(valid_df[channel_col].values)

        if best_channel not in valid_channels or worst_channel not in valid_channels:
            print(f"  ⚠️  DSPy suggested invalid channels, falling back to deterministic")
            # Fallback: sort by CAC
            valid_df = valid_df.sort_values(cac_col)
            best_channel = valid_df.iloc[0][channel_col]
            worst_channel = valid_df.iloc[-1][channel_col]

        # Get CAC values
        best_cac = valid_df[valid_df[channel_col] == best_channel][cac_col].values[0]
        worst_cac = valid_df[valid_df[channel_col] == worst_channel][cac_col].values[0]

        # Calculate current blended CAC
        total_spend = valid_df[spend_col].sum()
        weights = valid_df[spend_col] / total_spend
        current_blended_cac = (weights * valid_df[cac_col]).sum()

        # Simulate shift
        shift_fraction = shift_percentage / 100.0
        new_weights = weights.copy()

        best_idx = valid_df[valid_df[channel_col] == best_channel].index[0]
        worst_idx = valid_df[valid_df[channel_col] == worst_channel].index[0]

        new_weights[worst_idx] -= shift_fraction
        new_weights[best_idx] += shift_fraction

        projected_cac = (new_weights * valid_df[cac_col]).sum()
        delta_cac = projected_cac - current_blended_cac

        # Bootstrap confidence interval
        bootstrap_deltas = []
        for _ in range(self.n_bootstrap):
            sample_df = valid_df.sample(n=len(valid_df), replace=True)
            sample_weights = sample_df[spend_col] / sample_df[spend_col].sum()

            current_sample = (sample_weights * sample_df[cac_col]).sum()

            new_sample_weights = sample_weights.copy()
            sample_best_idx = sample_df[sample_df[cac_col] == sample_df[cac_col].min()].index[0]
            sample_worst_idx = sample_df[sample_df[cac_col] == sample_df[cac_col].max()].index[0]

            new_sample_weights[sample_worst_idx] -= shift_fraction
            new_sample_weights[sample_best_idx] += shift_fraction

            projected_sample = (new_sample_weights * sample_df[cac_col]).sum()
            bootstrap_deltas.append(projected_sample - current_sample)

        ci_lower = current_blended_cac + np.percentile(bootstrap_deltas, 2.5)
        ci_upper = current_blended_cac + np.percentile(bootstrap_deltas, 97.5)

        print(f"  ✓ Bootstrap validation complete: 95% CI = [${ci_lower:.2f}, ${ci_upper:.2f}]")

        return {
            'best_channel': best_channel,
            'best_cac': float(best_cac),
            'worst_channel': worst_channel,
            'worst_cac': float(worst_cac),
            'current_blended_cac': float(current_blended_cac),
            'shift_percentage': shift_percentage,
            'projected_blended_cac': float(projected_cac),
            'delta_cac': float(delta_cac),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstrap': self.n_bootstrap,
            'recommendation': f"Shift {shift_percentage}pp budget from {worst_channel} to {best_channel}",
            'dspy_reasoning': result.reasoning
        }


# ==================== AGENT 5: NARRATOR ====================

class NarratorAgent:
    """Generate decision memo using DSPy."""

    def __init__(self, signature):
        """
        Args:
            signature: DSPy NarratorSignature class
        """
        self.predictor = dspy.Predict(signature)

    def __call__(self, question: str, metrics_used: List[str],
                 key_findings: str, recommendation: str) -> Dict[str, str]:

        # Use DSPy to generate narrative
        print("  → Using DSPy NarratorSignature to generate decision memo")

        result = self.predictor(
            question=question,
            metrics_used=', '.join(metrics_used),
            key_findings=key_findings,
            recommendation=recommendation
        )

        memo = result.memo
        word_count = len(memo.split())

        # Validate constraints
        if word_count > 200:
            print(f"  ⚠️  Memo too long ({word_count} words), truncating...")
            words = memo.split()[:150]
            memo = ' '.join(words) + '...'
            word_count = 150

        # Check that it references at least one metric
        memo_lower = memo.lower()
        metric_referenced = any(m.lower() in memo_lower for m in metrics_used)
        if not metric_referenced:
            memo = f"[Metrics: {', '.join(metrics_used)}] " + memo

        print(f"  ✓ Memo generated: {word_count} words")

        return {
            'memo': memo,
            'word_count': word_count,
            'constraints_met': word_count <= 200 and metric_referenced
        }
