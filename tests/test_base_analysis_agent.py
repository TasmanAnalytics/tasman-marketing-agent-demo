"""Tests for BaseAnalysisAgent base class."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pandas as pd
import yaml

from agents.base_analysis_agent import BaseAnalysisAgent


# Concrete implementation for testing
class ConcreteAnalysisAgent(BaseAnalysisAgent):
    """Concrete implementation of BaseAnalysisAgent for testing."""

    def __init__(self, db_connector, config=None, llm_client=None):
        super().__init__(db_connector, config, llm_client, agent_name="ConcreteAnalysisAgent")

    def plan(self, question: str, role: str, context: dict) -> dict:
        """Simple plan generation for testing."""
        return {
            'sql': f"SELECT * FROM test_table WHERE question = '{question}'",
            'parameters': {
                'test_type': 'z_test',
                'confidence_level': 0.95
            },
            'expected_outputs': ['p_value', 'effect_size']
        }

    def run(self, df: pd.DataFrame, plan: dict) -> dict:
        """Simple analysis for testing."""
        return {
            'row_count': len(df),
            'column_count': len(df.columns),
            'test_type': plan['parameters'].get('test_type')
        }

    def report(self, results: dict, plan: dict) -> dict:
        """Simple report for testing."""
        return {
            'summary': f"Analyzed {results['row_count']} rows",
            'insights': ['Test insight 1', 'Test insight 2'],
            'raw_results': results
        }


class TestBaseAnalysisAgentInit:
    """Test initialization and configuration loading."""

    def test_init_with_config_dict(self):
        """Test initialization with config dict."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'confidence_level': 0.95}
        }

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        assert agent.db == mock_db
        assert agent.config == config
        assert agent.agent_name == "ConcreteAnalysisAgent"
        assert agent.max_rows == 100000

    def test_init_loads_config_from_file(self):
        """Test initialization loads config from file if not provided."""
        mock_db = Mock()

        agent = ConcreteAnalysisAgent(mock_db)

        assert agent.config is not None
        assert 'defaults' in agent.config
        assert 'stats' in agent.config
        assert agent.max_rows > 0

    def test_init_with_llm_client(self):
        """Test initialization with LLM client."""
        mock_db = Mock()
        mock_llm = Mock()
        config = {'defaults': {'max_rows': 50000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config, llm_client=mock_llm)

        assert agent.llm_client == mock_llm


class TestPull:
    """Test data pulling and sampling."""

    def test_pull_no_sampling_when_under_max(self):
        """Test that no sampling occurs when data is under max_rows."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 1000, 'random_seed': 42},
            'sampling': {'strata_dimensions': ['channel'], 'min_stratum_size': 10}
        }

        # Create test data under max_rows
        test_data = pd.DataFrame({
            'channel': ['Google'] * 500 + ['Facebook'] * 500,
            'clicks': range(1000)
        })

        mock_db.execute_query.return_value = test_data

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {'sql': 'SELECT * FROM test'}
        result_df = agent.pull(plan)

        # Should return all rows (no sampling)
        assert len(result_df) == 1000
        mock_db.execute_query.assert_called_once_with('SELECT * FROM test')

    def test_pull_sampling_when_over_max(self):
        """Test stratified sampling when data exceeds max_rows."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 500, 'random_seed': 42},
            'sampling': {'strata_dimensions': ['channel'], 'min_stratum_size': 50}
        }

        # Create test data over max_rows
        test_data = pd.DataFrame({
            'channel': ['Google'] * 1000 + ['Facebook'] * 1000,
            'clicks': range(2000)
        })

        mock_db.execute_query.return_value = test_data

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {'sql': 'SELECT * FROM test'}
        result_df = agent.pull(plan)

        # Should sample down to max_rows
        assert len(result_df) == 500
        assert len(result_df) < len(test_data)

        # Should preserve both strata
        assert 'Google' in result_df['channel'].values
        assert 'Facebook' in result_df['channel'].values

    def test_pull_with_plan_strata_override(self):
        """Test that plan can override stratification columns."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 500, 'random_seed': 42},
            'sampling': {'strata_dimensions': ['channel'], 'min_stratum_size': 50}
        }

        test_data = pd.DataFrame({
            'channel': ['Google'] * 1000,
            'device': ['mobile'] * 500 + ['desktop'] * 500,
            'clicks': range(1000)
        })

        mock_db.execute_query.return_value = test_data

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        # Plan specifies different strata
        plan = {
            'sql': 'SELECT * FROM test',
            'sampling': {'strata_cols': ['device']}
        }

        result_df = agent.pull(plan)

        # Should use 'device' for stratification
        assert len(result_df) == 500
        assert 'mobile' in result_df['device'].values
        assert 'desktop' in result_df['device'].values

    def test_pull_random_sampling_when_no_valid_strata(self):
        """Test random sampling when no valid stratification columns exist."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 500, 'random_seed': 42},
            'sampling': {'strata_dimensions': ['nonexistent_col'], 'min_stratum_size': 50}
        }

        test_data = pd.DataFrame({
            'value': range(1000)
        })

        mock_db.execute_query.return_value = test_data

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {'sql': 'SELECT * FROM test'}
        result_df = agent.pull(plan)

        # Should fall back to random sampling
        assert len(result_df) == 500

    def test_pull_raises_error_when_no_sql(self):
        """Test that pull raises error when plan has no SQL."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {'parameters': {}}  # Missing 'sql'

        with pytest.raises(ValueError, match="Plan must contain 'sql' key"):
            agent.pull(plan)


class TestExecute:
    """Test end-to-end execute workflow."""

    def test_execute_full_workflow(self):
        """Test complete execute workflow: plan → pull → run → report."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 10000, 'random_seed': 42},
            'stats': {'confidence_level': 0.95, 'bootstrap_iterations': 1000},
            'sampling': {'strata_dimensions': []},
            'reproducibility': {'save_plans': False}
        }

        test_data = pd.DataFrame({
            'channel': ['Google'] * 50,
            'clicks': range(50)
        })

        mock_db.execute_query.return_value = test_data

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        response = agent.execute(
            question="Is Google effective?",
            role="analyst",
            context={"previous_test": "yes"}
        )

        # Validate response structure
        assert 'plan' in response
        assert 'data_shape' in response
        assert 'results' in response
        assert 'report' in response
        assert 'metadata' in response

        # Validate plan
        assert 'sql' in response['plan']
        assert 'Google effective' in response['plan']['sql']

        # Validate data_shape
        assert response['data_shape']['rows'] == 50
        assert response['data_shape']['columns'] == 2

        # Validate results
        assert response['results']['row_count'] == 50
        assert response['results']['test_type'] == 'z_test'

        # Validate report
        assert 'summary' in response['report']
        assert '50 rows' in response['report']['summary']

        # Validate metadata
        assert response['metadata']['agent'] == "ConcreteAnalysisAgent"
        assert response['metadata']['question'] == "Is Google effective?"
        assert 'duration_seconds' in response['metadata']

    def test_execute_with_default_context(self):
        """Test execute with no context provided."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 10000, 'random_seed': 42},
            'stats': {'confidence_level': 0.95},
            'sampling': {'strata_dimensions': []},
            'reproducibility': {'save_plans': False}
        }

        test_data = pd.DataFrame({'value': [1, 2, 3]})
        mock_db.execute_query.return_value = test_data

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        # Should not raise error
        response = agent.execute(question="Test question")

        assert response is not None


class TestValidatePlan:
    """Test plan validation."""

    def test_validate_plan_valid(self):
        """Test validation of valid plan."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {
            'sql': 'SELECT * FROM test',
            'parameters': {'test_type': 'z_test'},
            'expected_outputs': ['p_value']
        }

        errors = agent.validate_plan(plan)
        assert errors == []

    def test_validate_plan_missing_sql(self):
        """Test validation catches missing SQL."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {
            'parameters': {'test_type': 'z_test'}
        }

        errors = agent.validate_plan(plan)
        assert len(errors) == 1
        assert 'sql' in errors[0].lower()

    def test_validate_plan_empty_sql(self):
        """Test validation catches empty SQL."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {
            'sql': '   ',  # Empty/whitespace
            'parameters': {}
        }

        errors = agent.validate_plan(plan)
        assert any('empty' in e.lower() for e in errors)

    def test_validate_plan_missing_parameters(self):
        """Test validation catches missing parameters."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {
            'sql': 'SELECT * FROM test'
        }

        errors = agent.validate_plan(plan)
        assert any('parameters' in e.lower() for e in errors)

    def test_validate_plan_invalid_parameters_type(self):
        """Test validation catches non-dict parameters."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {
            'sql': 'SELECT * FROM test',
            'parameters': ['list', 'not', 'dict']
        }

        errors = agent.validate_plan(plan)
        assert any('dict' in e.lower() for e in errors)


class TestSchemaContext:
    """Test schema context generation."""

    def test_get_schema_context(self):
        """Test schema context generation."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        # Mock SHOW TABLES
        mock_db.execute_query.side_effect = [
            pd.DataFrame({'name': ['fact_orders', 'dim_campaigns']}),  # SHOW TABLES
            pd.DataFrame({'column_name': ['order_id', 'revenue', 'date']}),  # DESCRIBE fact_orders
            pd.DataFrame({'column_name': ['campaign_id', 'channel']})  # DESCRIBE dim_campaigns
        ]

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        schema = agent.get_schema_context()

        assert 'fact_orders' in schema
        assert 'dim_campaigns' in schema
        assert 'order_id, revenue, date' in schema
        assert 'campaign_id, channel' in schema

    def test_get_schema_context_error_handling(self):
        """Test schema context handles errors gracefully."""
        mock_db = Mock()
        config = {'defaults': {'max_rows': 1000, 'random_seed': 42}}

        mock_db.execute_query.side_effect = Exception("Database error")

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        schema = agent.get_schema_context()

        assert 'unavailable' in schema.lower()


class TestConfigContext:
    """Test config context generation."""

    def test_get_config_context(self):
        """Test config context generation."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {
                'confidence_level': 0.95,
                'alpha': 0.05,
                'bootstrap_iterations': 1000
            }
        }

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        context = agent.get_config_context()

        assert '0.95' in context
        assert '0.05' in context
        assert '250000' in context
        assert '1000' in context
        assert '42' in context


class TestSavePlan:
    """Test plan saving for reproducibility."""

    def test_save_plan_creates_file(self, tmp_path):
        """Test that plan is saved to JSON file."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 1000, 'random_seed': 42},
            'reproducibility': {
                'save_plans': True,
                'plan_output_dir': str(tmp_path)
            }
        }

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {
            'sql': 'SELECT * FROM test',
            'parameters': {'test_type': 'z_test'}
        }

        agent._save_plan(plan, "Test question")

        # Check file was created
        saved_files = list(tmp_path.glob("*.json"))
        assert len(saved_files) == 1

        # Check file content
        with open(saved_files[0], 'r') as f:
            loaded_plan = json.load(f)

        assert loaded_plan == plan

    def test_save_plan_filename_format(self, tmp_path):
        """Test plan filename format."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 1000, 'random_seed': 42},
            'reproducibility': {
                'save_plans': True,
                'plan_output_dir': str(tmp_path)
            }
        }

        agent = ConcreteAnalysisAgent(mock_db, config=config)

        plan = {'sql': 'SELECT * FROM test', 'parameters': {}}
        agent._save_plan(plan, "Is campaign A better than B?")

        saved_files = list(tmp_path.glob("*.json"))
        filename = saved_files[0].name

        # Check filename components
        assert 'ConcreteAnalysisAgent' in filename
        assert 'Is_campaign_A_better' in filename  # Sanitized question
        assert filename.endswith('.json')
