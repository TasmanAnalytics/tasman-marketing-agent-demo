"""Local rule-based triage for query classification."""

import re
from typing import Dict, Any, Optional


class LocalTriage:
    """Rule-based query triage without LLM calls."""

    # Keywords indicating "search" queries
    SEARCH_KEYWORDS = [
        "what", "how many", "how much", "when", "show", "list", "plot",
        "display", "get", "find", "count", "sum", "average", "total",
        "by", "per", "top", "bottom", "highest", "lowest"
    ]

    # Keywords indicating "analysis" queries
    ANALYSIS_KEYWORDS = [
        "why", "driver", "drivers", "impact", "causal", "cause",
        "segment", "segmentation", "cluster", "clustering", "cohort",
        "correlation", "trend", "pattern", "anomaly", "outlier",
        "compare", "comparison", "versus", "vs", "difference"
    ]

    def __init__(self, business_context: Dict[str, Any]):
        """
        Initialize local triage.

        Args:
            business_context: Business context from config
        """
        self.business_context = business_context
        self.roles = business_context.get('roles', {})

    def normalize_question(self, question: str) -> str:
        """
        Normalize question for analysis.

        Args:
            question: Raw question

        Returns:
            Normalized question (lowercase, stripped)
        """
        return question.strip().lower()

    def count_keyword_matches(self, question: str, keywords: list) -> int:
        """
        Count keyword matches in question.

        Args:
            question: Normalized question
            keywords: List of keywords to match

        Returns:
            Count of matches
        """
        count = 0
        for keyword in keywords:
            # Use word boundary matching for better accuracy
            if re.search(rf'\b{re.escape(keyword)}\b', question):
                count += 1
        return count

    def infer_role_from_question(self, question: str) -> Optional[str]:
        """
        Infer user role from question content.

        Args:
            question: Normalized question

        Returns:
            Role name or None
        """
        role_scores = {}

        for role_name, role_config in self.roles.items():
            score = 0

            # Check KPIs
            kpis = role_config.get('kpis', [])
            for kpi in kpis:
                if kpi in question:
                    score += 2

            # Check dimensions
            dims = role_config.get('dims', [])
            for dim in dims:
                if dim in question:
                    score += 1

            # Check synonyms
            synonyms = role_config.get('synonyms', {})
            for canonical, syn_list in synonyms.items():
                for syn in syn_list:
                    if syn in question:
                        score += 1

            role_scores[role_name] = score

        # Return role with highest score (if any)
        if role_scores and max(role_scores.values()) > 0:
            return max(role_scores, key=role_scores.get)

        return None

    def triage(
        self,
        question: str,
        role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Triage query using local rules.

        Args:
            question: User question
            role: User role (optional, can be inferred)

        Returns:
            Dict with keys: mode, analysis_type, confidence, reason, inferred_role
        """
        norm_q = self.normalize_question(question)

        # Infer role if not provided
        if not role:
            role = self.infer_role_from_question(norm_q)

        # Count keyword matches
        search_matches = self.count_keyword_matches(norm_q, self.SEARCH_KEYWORDS)
        analysis_matches = self.count_keyword_matches(norm_q, self.ANALYSIS_KEYWORDS)

        # Simple heuristic: more search keywords = search mode
        if search_matches > analysis_matches:
            mode = "search"
            analysis_type = None
            confidence = min(0.9, 0.5 + (search_matches * 0.1))
            reason = f"Detected {search_matches} search keyword(s): descriptive query pattern"
        elif analysis_matches > 0:
            mode = "analysis"
            # Sub-classify analysis type
            if any(kw in norm_q for kw in ["why", "driver", "impact", "causal", "cause"]):
                analysis_type = "driver_analysis"
            elif any(kw in norm_q for kw in ["segment", "cluster", "cohort"]):
                analysis_type = "segmentation"
            elif any(kw in norm_q for kw in ["compare", "versus", "vs", "difference"]):
                analysis_type = "comparison"
            else:
                analysis_type = "hypothesis_testing"

            confidence = min(0.9, 0.5 + (analysis_matches * 0.15))
            reason = f"Detected {analysis_matches} analysis keyword(s): {analysis_type}"
        else:
            # Ambiguous - default to search but low confidence
            mode = "search"
            analysis_type = None
            confidence = 0.4
            reason = "No strong keyword matches; defaulting to search mode"

        return {
            "mode": mode,
            "analysis_type": analysis_type,
            "confidence": confidence,
            "reason": reason,
            "inferred_role": role,
            "method": "local_rules"
        }

    def should_call_llm(self, triage_result: Dict[str, Any], threshold: float = 0.6) -> bool:
        """
        Determine if LLM should be called based on confidence.

        Args:
            triage_result: Result from triage()
            threshold: Confidence threshold

        Returns:
            True if LLM should be called
        """
        return triage_result['confidence'] < threshold
