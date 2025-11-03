"""Local text-to-SQL engine with template matching and heuristics."""

import re
from typing import Dict, Any, Optional, List, Tuple
from difflib import SequenceMatcher


class LocalTextToSQL:
    """Local text-to-SQL using templates and heuristics."""

    def __init__(
        self,
        templates: List[Dict[str, Any]],
        schema: Dict[str, Dict[str, str]],
        business_context: Dict[str, Any],
        default_limit: int = 1000
    ):
        """
        Initialize local text-to-SQL engine.

        Args:
            templates: SQL templates from config
            schema: Database schema
            business_context: Business context (roles, KPIs, etc.)
            default_limit: Default row limit
        """
        self.templates = templates
        self.schema = schema
        self.business_context = business_context
        self.default_limit = default_limit
        self.roles = business_context.get('roles', {})

    def normalize_question(self, question: str) -> str:
        """Normalize question text."""
        return question.strip().lower()

    def match_template(
        self,
        question: str,
        role: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Match question to SQL template.

        Args:
            question: User question
            role: User role

        Returns:
            Tuple of (matched_template, confidence_score)
        """
        norm_q = self.normalize_question(question)
        best_match = None
        best_score = 0.0

        for template in self.templates:
            # Check role hint
            role_hint = template.get('role_hint')
            if role and role_hint and role != role_hint:
                # Reduce score for role mismatch but don't eliminate
                role_penalty = 0.2
            else:
                role_penalty = 0.0

            # Match against utterances
            utterances = template.get('utterances', [])
            for utterance in utterances:
                # Calculate similarity
                similarity = SequenceMatcher(None, norm_q, utterance.lower()).ratio()

                # Boost score if key terms match
                template_terms = set(utterance.lower().split())
                question_terms = set(norm_q.split())
                term_overlap = len(template_terms & question_terms) / max(len(template_terms), 1)

                # Combined score
                score = (similarity * 0.6 + term_overlap * 0.4) - role_penalty

                if score > best_score:
                    best_score = score
                    best_match = template

        return best_match, best_score

    def fill_template_params(
        self,
        template_sql: str,
        role: Optional[str] = None,
        time_window_days: Optional[int] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Fill template parameters.

        Args:
            template_sql: SQL template with {{placeholders}}
            role: User role
            time_window_days: Time window in days
            limit: Row limit

        Returns:
            SQL with parameters filled
        """
        # Get defaults from role or business context
        if time_window_days is None:
            if role and role in self.roles:
                time_window_days = self.roles[role].get('defaults', {}).get('time_window_days', 90)
            else:
                time_window_days = self.business_context.get('defaults', {}).get('time_window_days', 90)

        if limit is None:
            limit = self.default_limit

        # Replace placeholders
        sql = template_sql.replace('{{time_window_days}}', str(time_window_days))
        sql = sql.replace('{{limit}}', str(limit))

        return sql

    def extract_entities(self, question: str) -> Dict[str, Any]:
        """
        Extract entities (KPIs, dimensions, time ranges) from question.

        Args:
            question: User question

        Returns:
            Dict of extracted entities
        """
        norm_q = self.normalize_question(question)
        entities = {
            'kpis': [],
            'dimensions': [],
            'time_ranges': []
        }

        # Extract KPIs from all roles
        all_kpis = set()
        for role_config in self.roles.values():
            all_kpis.update(role_config.get('kpis', []))

        for kpi in all_kpis:
            if kpi in norm_q:
                entities['kpis'].append(kpi)

        # Extract dimensions
        all_dims = set()
        for role_config in self.roles.values():
            all_dims.update(role_config.get('dims', []))

        for dim in all_dims:
            if dim in norm_q:
                entities['dimensions'].append(dim)

        # Extract time ranges (simple patterns)
        time_patterns = {
            r'last (\d+) days?': lambda m: int(m.group(1)),
            r'past (\d+) days?': lambda m: int(m.group(1)),
            r'(\d+) days?': lambda m: int(m.group(1)),
            r'last month': lambda m: 30,
            r'last quarter': lambda m: 90,
            r'last year': lambda m: 365,
        }

        for pattern, extractor in time_patterns.items():
            match = re.search(pattern, norm_q)
            if match:
                entities['time_ranges'].append(extractor(match))
                break

        return entities

    def validate_sql_schema(self, sql: str) -> Tuple[bool, List[str]]:
        """
        Validate SQL against schema.

        Args:
            sql: SQL query

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Extract table references (simple regex)
        # Pattern: FROM/JOIN <table_name>
        table_pattern = r'(?:FROM|JOIN)\s+(\w+)'
        referenced_tables = re.findall(table_pattern, sql, re.IGNORECASE)

        for table in referenced_tables:
            if table not in self.schema:
                errors.append(f"Unknown table: {table}")

        # Extract column references (pattern: <table>.<column>)
        column_pattern = r'(\w+)\.(\w+)'
        referenced_columns = re.findall(column_pattern, sql)

        for table, column in referenced_columns:
            if table in self.schema:
                if column not in self.schema[table]:
                    errors.append(f"Unknown column: {table}.{column}")

        return (len(errors) == 0, errors)

    def generate_sql(
        self,
        question: str,
        role: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate SQL from question using local logic.

        Args:
            question: User question
            role: User role

        Returns:
            Dict with keys: sql, template_id, confidence, method, entities
        """
        # Try template matching first
        template, match_score = self.match_template(question, role)

        if template and match_score > 0.6:
            # High confidence template match
            template_sql = template['sql']
            entities = self.extract_entities(question)

            # Fill parameters
            time_window = entities['time_ranges'][0] if entities['time_ranges'] else None
            sql = self.fill_template_params(template_sql, role, time_window)

            # Validate against schema
            is_valid, errors = self.validate_sql_schema(sql)

            return {
                'sql': sql,
                'template_id': template['id'],
                'confidence': match_score,
                'method': 'template_match',
                'entities': entities,
                'valid': is_valid,
                'validation_errors': errors
            }

        # Moderate confidence - try heuristic generation
        elif template and match_score > 0.4:
            # Use template as base but lower confidence
            template_sql = template['sql']
            entities = self.extract_entities(question)
            time_window = entities['time_ranges'][0] if entities['time_ranges'] else None
            sql = self.fill_template_params(template_sql, role, time_window)
            is_valid, errors = self.validate_sql_schema(sql)

            return {
                'sql': sql,
                'template_id': template['id'],
                'confidence': match_score,
                'method': 'template_match_low_confidence',
                'entities': entities,
                'valid': is_valid,
                'validation_errors': errors
            }

        # No good template match - return failure (LLM should be called)
        else:
            entities = self.extract_entities(question)
            return {
                'sql': None,
                'template_id': None,
                'confidence': 0.0,
                'method': 'no_match',
                'entities': entities,
                'valid': False,
                'validation_errors': ['No template match found']
            }

    def should_call_llm(self, result: Dict[str, Any], threshold: float = 0.6) -> bool:
        """
        Determine if LLM should be called.

        Args:
            result: Result from generate_sql()
            threshold: Confidence threshold

        Returns:
            True if LLM should be called
        """
        return result['sql'] is None or result['confidence'] < threshold
