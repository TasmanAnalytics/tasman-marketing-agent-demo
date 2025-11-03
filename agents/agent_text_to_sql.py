"""Text-to-SQL agent: local templates first, LLM fallback."""

import json
from typing import Dict, Any, Optional
from core.local_text_to_sql import LocalTextToSQL
from core.llm_clients import LLMClient, text_to_sql_llm


class TextToSQLAgent:
    """Agent for converting natural language to SQL."""

    def __init__(
        self,
        templates: list,
        schema: Dict[str, Dict[str, str]],
        business_context: Dict[str, Any],
        llm_client: Optional[LLMClient] = None,
        llm_threshold: float = 0.6,
        default_limit: int = 1000
    ):
        """
        Initialize text-to-SQL agent.

        Args:
            templates: SQL templates
            schema: Database schema
            business_context: Business context
            llm_client: Optional LLM client
            llm_threshold: Confidence threshold for LLM
            default_limit: Default row limit
        """
        self.local_engine = LocalTextToSQL(
            templates,
            schema,
            business_context,
            default_limit
        )
        self.llm_client = llm_client
        self.llm_threshold = llm_threshold
        self.schema = schema
        self.business_context = business_context

    def generate_sql(
        self,
        question: str,
        role: Optional[str] = None,
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        Generate SQL from natural language question.

        Args:
            question: User question
            role: User role
            force_local: Force local-only (no LLM)

        Returns:
            Dict with SQL and metadata
        """
        # Try local engine first
        local_result = self.local_engine.generate_sql(question, role)

        # Check if LLM should be called
        should_call_llm = (
            not force_local and
            self.llm_client is not None and
            self.local_engine.should_call_llm(local_result, self.llm_threshold)
        )

        if should_call_llm:
            # Call LLM for SQL generation
            schema_json = json.dumps(self.schema, indent=2)
            llm_result = text_to_sql_llm(
                question,
                role,
                schema_json,
                self.business_context,
                self.llm_client
            )

            # Validate LLM-generated SQL
            if llm_result['sql']:
                is_valid, errors = self.local_engine.validate_sql_schema(llm_result['sql'])
                llm_result['valid'] = is_valid
                llm_result['validation_errors'] = errors

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
