"""Triage agent: local rules first, LLM fallback."""

from typing import Dict, Any, Optional
from core.triage_local import LocalTriage
from core.llm_clients import LLMClient, triage_llm


class TriageAgent:
    """Agent for triaging user queries."""

    def __init__(
        self,
        business_context: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        llm_threshold: float = 0.6
    ):
        """
        Initialize triage agent.

        Args:
            business_context: Business context config
            llm_client: Optional LLM client for fallback
            llm_threshold: Confidence threshold for LLM fallback
        """
        self.local_triage = LocalTriage(business_context)
        self.llm_client = llm_client
        self.llm_threshold = llm_threshold
        self.business_context = business_context

    def triage(
        self,
        question: str,
        role: Optional[str] = None,
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        Triage user question.

        Args:
            question: User question
            role: User role
            force_local: Force local-only (no LLM)

        Returns:
            Triage result dict
        """
        # Try local triage first
        local_result = self.local_triage.triage(question, role)

        # Use inferred role if not provided
        if not role and local_result.get('inferred_role'):
            role = local_result['inferred_role']

        # Check if LLM should be called
        should_call_llm = (
            not force_local and
            self.llm_client is not None and
            self.local_triage.should_call_llm(local_result, self.llm_threshold)
        )

        if should_call_llm:
            # Call LLM for better triage
            llm_result = triage_llm(
                question,
                role,
                self.business_context,
                self.llm_client
            )

            # Merge results (prefer LLM but keep local insights)
            return {
                **llm_result,
                'local_result': local_result,
                'used_llm': True
            }
        else:
            # Use local result
            return {
                **local_result,
                'used_llm': False
            }
