"""Search agent: orchestrates triage, SQL generation, execution, and visualization."""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from agents.agent_triage import TriageAgent
from agents.agent_text_to_sql import TextToSQLAgent
from core.duckdb_connector import DuckDBConnector
from core.viz import AutoViz


class SearchAgent:
    """Agent for handling search queries end-to-end."""

    def __init__(
        self,
        triage_agent: TriageAgent,
        text_to_sql_agent: TextToSQLAgent,
        db_connector: DuckDBConnector,
        output_dir: Path
    ):
        """
        Initialize search agent.

        Args:
            triage_agent: Triage agent
            text_to_sql_agent: Text-to-SQL agent
            db_connector: Database connector
            output_dir: Output directory for visualizations
        """
        self.triage_agent = triage_agent
        self.text_to_sql_agent = text_to_sql_agent
        self.db_connector = db_connector
        self.viz = AutoViz(output_dir)

    def search(
        self,
        question: str,
        role: Optional[str] = None,
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        Execute search query end-to-end.

        Args:
            question: User question
            role: User role
            force_local: Force local-only (no LLM)

        Returns:
            Dict with results and metadata
        """
        result = {
            'question': question,
            'role': role,
            'steps': []
        }

        # Step 1: Triage
        triage_result = self.triage_agent.triage(question, role, force_local)
        result['triage'] = triage_result
        result['steps'].append({
            'step': 'triage',
            'mode': triage_result['mode'],
            'confidence': triage_result['confidence'],
            'used_llm': triage_result.get('used_llm', False)
        })

        # Use inferred role if available
        if not role and triage_result.get('inferred_role'):
            role = triage_result['inferred_role']
            result['role'] = role

        # Only proceed if mode is "search"
        if triage_result['mode'] != 'search':
            result['status'] = 'unsupported_mode'
            result['message'] = (
                f"Query requires '{triage_result['mode']}' mode which is not "
                f"implemented in this phase. Analysis type: {triage_result.get('analysis_type')}"
            )
            return result

        # Step 2: Generate SQL
        sql_result = self.text_to_sql_agent.generate_sql(question, role, force_local)
        result['sql_generation'] = sql_result
        result['steps'].append({
            'step': 'sql_generation',
            'method': sql_result['method'],
            'confidence': sql_result['confidence'],
            'used_llm': sql_result.get('used_llm', False),
            'template_id': sql_result.get('template_id')
        })

        # Check if SQL was generated
        if not sql_result['sql'] or not sql_result['valid']:
            result['status'] = 'sql_generation_failed'
            result['message'] = "Failed to generate valid SQL"
            result['errors'] = sql_result.get('validation_errors', [])
            return result

        sql = sql_result['sql']
        result['sql'] = sql

        # Step 3: Execute SQL
        try:
            df = self.db_connector.execute(sql)
            result['data'] = df
            result['row_count'] = len(df)
            result['steps'].append({
                'step': 'execution',
                'row_count': len(df),
                'success': True
            })
        except Exception as e:
            result['status'] = 'execution_failed'
            result['message'] = f"SQL execution failed: {str(e)}"
            result['steps'].append({
                'step': 'execution',
                'success': False,
                'error': str(e)
            })
            return result

        # Step 4: Visualize
        chart_filename = f"chart_{hash(question) % 100000}.png"
        try:
            chart_path, chart_type = self.viz.visualize(df, chart_filename)
            result['chart_path'] = chart_path
            result['chart_type'] = chart_type
            result['steps'].append({
                'step': 'visualization',
                'chart_type': chart_type,
                'success': chart_path is not None
            })
        except Exception as e:
            result['chart_path'] = None
            result['chart_type'] = 'error'
            result['steps'].append({
                'step': 'visualization',
                'success': False,
                'error': str(e)
            })

        # Step 5: Summarize
        summary = self.viz.summarize_result(df, result.get('chart_path'))
        result['summary'] = summary

        result['status'] = 'success'
        return result
