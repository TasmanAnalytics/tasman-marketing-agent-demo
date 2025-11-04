"""Tests for DriverAnalysisAgent."""

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from agents.agent_driver import DriverAnalysisAgent
from tests.fixtures.synthetic_data import generate_driver_analysis_data


class TestDriverAgentInit:
    """Test initialization."""

    def test_init_sets_correct_max_rows(self):
        """Test that max_rows is set from driver config."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }

        agent = DriverAnalysisAgent(mock_db, config=config)

        assert agent.max_rows == 250000
        assert agent.agent_name == "DriverAnalysisAgent"


class TestPlanGeneration:
    """Test plan generation from questions."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return DriverAnalysisAgent(mock_db, config=config)

    def test_plan_conversion_drivers(self, agent):
        """Test plan for conversion drivers."""
        question = "What drives conversion?"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['target'] == 'converted_flag'
        assert plan['target_type'] == 'binary'
        assert plan['unit'] == 'session'
        assert plan['model_type'] == 'logistic'
        assert 'device' in plan['features']
        assert 'channel' in plan['features']
        assert plan['window_days'] == 180

    def test_plan_revenue_drivers(self, agent):
        """Test plan for revenue drivers."""
        question = "What are the key drivers of revenue?"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['target'] == 'revenue'
        assert plan['target_type'] == 'continuous'
        assert plan['model_type'] == 'linear'
        assert 'sql' in plan

    def test_plan_with_device_filter(self, agent):
        """Test plan with device filter."""
        question = "What drives conversion on mobile?"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['filters']['device'] == 'mobile'
        assert 'mobile' in plan['sql'].lower()

    def test_plan_with_time_window(self, agent):
        """Test plan with explicit time window."""
        question = "What drives conversion in the last 90 days?"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['window_days'] == 90

    def test_plan_customer_level(self, agent):
        """Test plan for customer-level analysis."""
        question = "What drives total revenue per customer?"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['target'] == 'total_revenue'
        assert plan['unit'] == 'customer'


class TestDriverAnalysisLogistic:
    """Test logistic regression for binary outcomes."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return DriverAnalysisAgent(mock_db, config=config)

    def test_run_logistic_binary_outcome(self, agent):
        """Test logistic regression on synthetic binary data."""
        # Generate data with known drivers
        X, y = generate_driver_analysis_data(
            n_samples=1000,
            n_features=5,
            feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            target_type='binary',
            true_coefficients={'feature_1': 1.5, 'feature_2': -0.8, 'feature_3': 0.3},
            random_state=42
        )

        # Add target to dataframe
        df = X.copy()
        df['converted_flag'] = y

        plan = {
            'target': 'converted_flag',
            'target_type': 'binary',
            'unit': 'session',
            'features': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
            'model_type': 'logistic',
            'cv_folds': 3,
            'max_features': 50,
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['model_type'] == 'logistic'
        assert 'model_metrics' in result
        assert 'auc' in result['model_metrics']['metric']
        assert result['model_metrics']['test_auc'] > 0.5  # Better than random
        assert len(result['drivers']) > 0
        assert result['drivers'][0]['rank'] == 1

        # Top driver should be feature_1 (strongest coefficient)
        top_driver = result['drivers'][0]['feature']
        assert 'feature_1' in top_driver


class TestDriverAnalysisLinear:
    """Test linear regression for continuous outcomes."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return DriverAnalysisAgent(mock_db, config=config)

    def test_run_linear_continuous_outcome(self, agent):
        """Test linear regression on synthetic continuous data."""
        # Generate data with known drivers
        X, y = generate_driver_analysis_data(
            n_samples=1000,
            n_features=5,
            feature_names=['price', 'quantity', 'discount', 'feature_4', 'feature_5'],
            target_type='continuous',
            true_coefficients={'price': 2.0, 'quantity': 1.5, 'discount': -0.5},
            noise_level=0.5,
            random_state=42
        )

        # Add target to dataframe
        df = X.copy()
        df['revenue'] = y

        plan = {
            'target': 'revenue',
            'target_type': 'continuous',
            'unit': 'order',
            'features': ['price', 'quantity', 'discount', 'feature_4', 'feature_5'],
            'model_type': 'linear',
            'cv_folds': 3,
            'max_features': 50,
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['model_type'] == 'linear'
        assert 'r2' in result['model_metrics']['metric']
        assert result['model_metrics']['test_r2'] > 0.3  # Should explain some variance
        assert len(result['drivers']) > 0

        # Check coefficients
        assert 'coefficients' in result
        assert len(result['coefficients']) > 0


class TestCategoricalFeatures:
    """Test handling of categorical features."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return DriverAnalysisAgent(mock_db, config=config)

    def test_run_with_categorical_features(self, agent):
        """Test driver analysis with categorical features."""
        np.random.seed(42)

        # Create data with categorical features
        n_samples = 1000
        df = pd.DataFrame({
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
            'channel': np.random.choice(['Google', 'Facebook', 'LinkedIn'], n_samples),
            'pages_viewed': np.random.randint(1, 20, n_samples),
            'converted_flag': np.random.randint(0, 2, n_samples)
        })

        plan = {
            'target': 'converted_flag',
            'target_type': 'binary',
            'unit': 'session',
            'features': ['device', 'channel', 'pages_viewed'],
            'model_type': 'logistic',
            'cv_folds': 3,
            'max_features': 50,
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['model_type'] == 'logistic'
        assert result['diagnostics']['n_categorical'] == 2
        assert result['diagnostics']['n_numeric'] == 1
        # One-hot encoding should create multiple features
        assert result['diagnostics']['n_features'] > 3


class TestReportGeneration:
    """Test report generation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return DriverAnalysisAgent(mock_db, config=config)

    def test_report_good_model(self, agent):
        """Test report for good model performance."""
        results = {
            'model_type': 'logistic',
            'model_metrics': {
                'cv_mean': 0.82,
                'cv_std': 0.03,
                'test_auc': 0.85,
                'metric': 'auc'
            },
            'drivers': [
                {'feature': 'channel_Google', 'importance': 0.15, 'importance_std': 0.02, 'rank': 1},
                {'feature': 'pages_viewed', 'importance': 0.12, 'importance_std': 0.01, 'rank': 2},
                {'feature': 'device_mobile', 'importance': 0.08, 'importance_std': 0.01, 'rank': 3}
            ],
            'coefficients': {
                'channel_Google': 0.5,
                'pages_viewed': 0.3,
                'device_mobile': -0.2
            },
            'diagnostics': {
                'n_samples': 5000,
                'n_features': 8,
                'n_categorical': 2,
                'n_numeric': 1,
                'train_size': 4000,
                'test_size': 1000
            }
        }

        plan = {
            'target': 'converted_flag',
            'unit': 'session'
        }

        report = agent.report(results, plan)

        assert 'summary' in report
        assert 'Strong predictive performance' in report['insights'][0]
        assert 'Top drivers' in report['insights'][1]
        assert len(report['next_actions']) > 0

    def test_report_poor_model(self, agent):
        """Test report for poor model performance."""
        results = {
            'model_type': 'logistic',
            'model_metrics': {
                'cv_mean': 0.58,
                'cv_std': 0.05,
                'test_auc': 0.55,
                'metric': 'auc'
            },
            'drivers': [
                {'feature': 'feature_1', 'importance': 0.05, 'importance_std': 0.02, 'rank': 1}
            ],
            'coefficients': {'feature_1': 0.1},
            'diagnostics': {
                'n_samples': 500,
                'n_features': 3,
                'n_categorical': 0,
                'n_numeric': 3,
                'train_size': 400,
                'test_size': 100
            }
        }

        plan = {
            'target': 'converted_flag',
            'unit': 'session'
        }

        report = agent.report(results, plan)

        assert any('Low AUC' in caveat for caveat in report['caveats'])
        assert any('Small sample size' in caveat for caveat in report['caveats'])
        assert any('Add more features' in action for action in report['next_actions'])


class TestExtractors:
    """Test helper extraction methods."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'cv_folds': 5},
            'row_caps': {'modelling_max_rows': 250000},
            'driver': {
                'max_features': 50,
                'regularization_alpha': 1.0,
                'categorical_encoding': {
                    'max_categories': 20,
                    'min_category_frequency': 0.01
                }
            },
            'windows': {'driver_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return DriverAnalysisAgent(mock_db, config=config)

    def test_extract_target_conversion(self, agent):
        """Test conversion target extraction."""
        target, target_type = agent._extract_target("What drives conversion?")
        assert target == 'converted_flag'
        assert target_type == 'binary'

    def test_extract_target_revenue(self, agent):
        """Test revenue target extraction."""
        target, target_type = agent._extract_target("What drives revenue?")
        assert target == 'revenue'
        assert target_type == 'continuous'

    def test_extract_unit_session(self, agent):
        """Test session unit extraction."""
        unit = agent._extract_unit("What drives session conversion?", 'converted_flag')
        assert unit == 'session'

    def test_extract_unit_customer(self, agent):
        """Test customer unit extraction."""
        unit = agent._extract_unit("What drives customer lifetime value?", 'total_revenue')
        assert unit == 'customer'

    def test_extract_filters(self, agent):
        """Test filter extraction."""
        filters = agent._extract_filters("What drives conversion on mobile in the US?")
        assert filters.get('device') == 'mobile'
        # Note: US region filter not yet implemented in current version

    def test_extract_window(self, agent):
        """Test window extraction."""
        assert agent._extract_window("in the last 90 days") == 90
        assert agent._extract_window("last 2 weeks") == 14
        assert agent._extract_window("no time mentioned") == 180  # default
