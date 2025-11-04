"""Base class for all Phase 2 analysis agents.

Provides common interface for plan → pull → run → report workflow.
All specialized analysis agents (hypothesis testing, driver analysis, etc.) inherit from this.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import pandas as pd
import yaml

from core.analysis_utils import StratifiedSampler


logger = logging.getLogger(__name__)


class BaseAnalysisAgent(ABC):
    """Base class for all analysis agents.

    Workflow:
        1. plan() - Generate analysis plan (JSON) using LLM
        2. pull() - Execute SQL and apply sampling if needed
        3. run() - Perform statistical analysis (local computation)
        4. report() - Generate final report with insights
        5. execute() - End-to-end orchestration

    Attributes:
        db: Database connector (DuckDB)
        config: Analysis configuration from config/analysis.yaml
        llm_client: LLM client for plan generation (optional)
        agent_name: Name of the agent (e.g., "HypothesisTestingAgent")
        max_rows: Maximum rows for this agent type
    """

    def __init__(
        self,
        db_connector,
        config: Optional[Dict[str, Any]] = None,
        llm_client=None,
        agent_name: str = "BaseAnalysisAgent"
    ):
        """Initialize analysis agent.

        Args:
            db_connector: Database connector with execute_query method
            config: Analysis configuration (if None, loads from config/analysis.yaml)
            llm_client: LLM client for plan generation
            agent_name: Name of the agent for logging
        """
        self.db = db_connector
        self.llm_client = llm_client
        self.agent_name = agent_name

        # Load config if not provided
        if config is None:
            config_path = Path(__file__).parent.parent / "config" / "analysis.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self.config = config

        # Get max rows for this agent type
        self.max_rows = self._get_max_rows()

        logger.info(f"Initialized {self.agent_name} (max_rows={self.max_rows})")

    def _get_max_rows(self) -> int:
        """Get max rows for this agent type from config."""
        # Subclasses can override this to specify their row cap type
        return self.config.get('defaults', {}).get('max_rows', 250000)

    @abstractmethod
    def plan(self, question: str, role: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis plan (JSON).

        This method should use the LLM to generate a structured plan that includes:
        - SQL query to pull data
        - Analysis parameters (test type, confidence level, etc.)
        - Expected outputs

        Args:
            question: User's question (e.g., "Is campaign A better than B?")
            role: User's role for context
            context: Additional context (schema, previous results, etc.)

        Returns:
            Plan dict with keys: sql, parameters, expected_outputs, metadata
        """
        pass

    @abstractmethod
    def run(self, df: pd.DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis (local computation).

        This is where the actual analysis happens - hypothesis tests, modeling, etc.
        Should be deterministic and reproducible given the same data and plan.

        Args:
            df: DataFrame from pull()
            plan: Plan dict from plan()

        Returns:
            Results dict with analysis outputs
        """
        pass

    @abstractmethod
    def report(self, results: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final report with insights.

        Uses LLM to convert statistical results into human-readable insights.

        Args:
            results: Results dict from run()
            plan: Original plan dict

        Returns:
            Report dict with keys: summary, insights, visualizations, raw_results
        """
        pass

    def pull(self, plan: Dict[str, Any]) -> pd.DataFrame:
        """Execute SQL and apply sampling if needed.

        This method:
        1. Executes the SQL query from the plan
        2. Checks if row count exceeds max_rows
        3. Applies stratified sampling if needed
        4. Logs sampling metadata

        Args:
            plan: Plan dict with 'sql' key

        Returns:
            DataFrame (sampled if needed)
        """
        sql = plan.get('sql')
        if not sql:
            raise ValueError("Plan must contain 'sql' key")

        logger.info(f"Executing query: {sql[:100]}...")

        # Execute query
        df = self.db.execute_query(sql)

        original_rows = len(df)
        logger.info(f"Query returned {original_rows} rows")

        # Check if sampling is needed
        if original_rows > self.max_rows:
            logger.warning(
                f"Row count ({original_rows}) exceeds max_rows ({self.max_rows}). "
                f"Applying stratified sampling..."
            )

            # Get stratification columns from plan or config
            strata_cols = plan.get('sampling', {}).get('strata_cols')
            if not strata_cols:
                strata_cols = self.config.get('sampling', {}).get('strata_dimensions', [])

            # Filter to columns that exist in df
            strata_cols = [col for col in strata_cols if col in df.columns]

            if not strata_cols:
                logger.warning("No valid stratification columns found. Using random sampling.")
                df = df.sample(n=self.max_rows, random_state=self.config['defaults']['random_seed'])
            else:
                # Stratified sampling
                sampler = StratifiedSampler(
                    max_rows=self.max_rows,
                    strata_cols=strata_cols,
                    min_stratum_size=self.config['sampling'].get('min_stratum_size', 100),
                    random_state=self.config['defaults']['random_seed']
                )

                df = sampler.fit_sample(df)

                # Log sampling metadata
                logger.info(
                    f"Sampled to {len(df)} rows "
                    f"(ratio={len(df)/original_rows:.2%}, strata={strata_cols})"
                )

        return df

    def execute(
        self,
        question: str,
        role: str = "analyst",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """End-to-end execution: plan → pull → run → report.

        Args:
            question: User's question
            role: User's role
            context: Additional context

        Returns:
            Complete response dict with:
                - plan: The generated plan
                - data_shape: Shape of pulled data
                - results: Statistical results
                - report: Final report with insights
                - metadata: Execution metadata
        """
        if context is None:
            context = {}

        start_time = datetime.now()

        logger.info(f"[{self.agent_name}] Starting execution for: {question}")

        # Step 1: Plan
        logger.info("Step 1/4: Generating plan...")
        plan = self.plan(question, role, context)

        # Save plan if configured
        if self.config.get('reproducibility', {}).get('save_plans', False):
            self._save_plan(plan, question)

        # Step 2: Pull
        logger.info("Step 2/4: Pulling data...")
        df = self.pull(plan)

        # Step 3: Run
        logger.info("Step 3/4: Running analysis...")
        results = self.run(df, plan)

        # Step 4: Report
        logger.info("Step 4/4: Generating report...")
        report = self.report(results, plan)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"[{self.agent_name}] Execution complete ({duration:.2f}s)")

        # Compile complete response
        response = {
            'plan': plan,
            'data_shape': {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            },
            'results': results,
            'report': report,
            'metadata': {
                'agent': self.agent_name,
                'question': question,
                'role': role,
                'duration_seconds': duration,
                'timestamp': end_time.isoformat(),
                'config_snapshot': {
                    'max_rows': self.max_rows,
                    'random_seed': self.config['defaults']['random_seed'],
                    'confidence_level': self.config['stats']['confidence_level']
                }
            }
        }

        return response

    def _save_plan(self, plan: Dict[str, Any], question: str):
        """Save plan to JSON file for reproducibility."""
        plan_dir = Path(self.config['reproducibility']['plan_output_dir'])
        plan_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from timestamp and question
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_question = "".join(c if c.isalnum() else "_" for c in question[:50])
        filename = f"{timestamp}_{self.agent_name}_{safe_question}.json"

        filepath = plan_dir / filename

        with open(filepath, 'w') as f:
            json.dump(plan, f, indent=2)

        logger.info(f"Saved plan to {filepath}")

    def validate_plan(self, plan: Dict[str, Any]) -> List[str]:
        """Validate plan structure and return list of errors.

        Args:
            plan: Plan dict to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Required top-level keys
        required_keys = ['sql', 'parameters']
        for key in required_keys:
            if key not in plan:
                errors.append(f"Missing required key: '{key}'")

        # Validate SQL is non-empty
        if 'sql' in plan and not plan['sql'].strip():
            errors.append("SQL query is empty")

        # Validate parameters is dict
        if 'parameters' in plan and not isinstance(plan['parameters'], dict):
            errors.append("'parameters' must be a dict")

        return errors

    def get_schema_context(self) -> str:
        """Get database schema as context for plan generation.

        Returns:
            String describing available tables and columns
        """
        try:
            # Get list of tables
            tables = self.db.execute_query("SHOW TABLES")

            schema_parts = ["Available tables and columns:\n"]

            for table_row in tables.itertuples():
                table_name = table_row.name

                # Get columns for this table
                columns = self.db.execute_query(f"DESCRIBE {table_name}")
                col_names = ", ".join(columns['column_name'].tolist())

                schema_parts.append(f"- {table_name}: {col_names}")

            return "\n".join(schema_parts)

        except Exception as e:
            logger.error(f"Error fetching schema: {e}")
            return "Schema information unavailable"

    def get_config_context(self) -> str:
        """Get relevant config parameters as context.

        Returns:
            String describing key configuration values
        """
        return f"""Configuration:
- Confidence level: {self.config['stats']['confidence_level']}
- Significance threshold (alpha): {self.config['stats']['alpha']}
- Max rows for this agent: {self.max_rows}
- Bootstrap iterations: {self.config['stats']['bootstrap_iterations']}
- Random seed: {self.config['defaults']['random_seed']}
"""
