"""Tests for HypothesisTestingAgent."""

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from agents.agent_hypothesis import HypothesisTestingAgent


class TestHypothesisAgentInit:
    """Test initialization."""

    def test_init_sets_correct_max_rows(self):
        """Test that max_rows is set from hypothesis config."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 250000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'confidence_level': 0.95},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }

        agent = HypothesisTestingAgent(mock_db, config=config)

        assert agent.max_rows == 100000
        assert agent.agent_name == "HypothesisTestingAgent"


class TestPlanGeneration:
    """Test plan generation from questions."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'confidence_level': 0.95},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_plan_ctr_comparison_two_channels(self, agent):
        """Test plan for CTR comparison between two channels."""
        question = "Is LinkedIn CTR better than Facebook CTR?"

        plan = agent.plan(question, role="marketer", context={})

        assert plan['metric'] == 'ctr'
        assert plan['metric_type'] == 'proportion'
        assert 'LinkedIn' in plan['groups']
        assert 'Facebook' in plan['groups']
        assert 'channel' in plan['dimension'].lower()
        assert plan['test_type'] == 'z_test'
        assert plan['alternative'] == 'greater'
        assert plan['window_days'] == 90
        assert 'sql' in plan

    def test_plan_cvr_comparison_devices(self, agent):
        """Test plan for CVR comparison across devices."""
        question = "Compare conversion rates between mobile and desktop"

        plan = agent.plan(question, role="marketer", context={})

        assert plan['metric'] == 'cvr'
        assert 'mobile' in plan['groups']
        assert 'desktop' in plan['groups']
        assert 'device' in plan['dimension'].lower()
        assert plan['test_type'] == 'z_test'
        assert plan['alternative'] == 'two-sided'

    def test_plan_aov_comparison(self, agent):
        """Test plan for AOV comparison."""
        question = "Is mobile AOV lower than desktop AOV?"

        plan = agent.plan(question, role="marketer", context={})

        assert plan['metric'] == 'aov'
        assert plan['metric_type'] == 'continuous'
        assert plan['test_type'] == 't_test'
        assert plan['alternative'] == 'less'

    def test_plan_multiple_groups_triggers_chi_squared(self, agent):
        """Test that >2 groups triggers chi-squared for proportions."""
        question = "Compare CTR across Google, Facebook, and LinkedIn"

        plan = agent.plan(question, role="marketer", context={})

        assert plan['metric'] == 'ctr'
        assert len(plan['groups']) == 3
        assert plan['test_type'] == 'chi_squared'
        assert plan['corrections'] is True

    def test_plan_extracts_time_window(self, agent):
        """Test that time window is extracted from question."""
        question = "Is LinkedIn CTR better than Facebook CTR in the last 30 days?"

        plan = agent.plan(question, role="marketer", context={})

        assert plan['window_days'] == 30

    def test_plan_defaults_to_config_window(self, agent):
        """Test that window defaults to config if not specified."""
        question = "Is LinkedIn CTR better than Facebook CTR?"

        plan = agent.plan(question, role="marketer", context={})

        assert plan['window_days'] == 90  # From config


class TestProportionTest:
    """Test two-sample proportion tests (z-test)."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'min_sample_size': 30},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_run_proportion_test_ctr(self, agent):
        """Test proportion test with CTR data."""
        # LinkedIn: 500/10000 = 5.0%
        # Facebook: 300/10000 = 3.0%
        # This should be a significant difference
        df = pd.DataFrame({
            'channel': ['LinkedIn', 'Facebook'],
            'clicks': [500, 300],
            'impressions': [10000, 10000],
            'ctr': [0.05, 0.03]
        })

        plan = {
            'metric': 'ctr',
            'metric_type': 'proportion',
            'groups': ['LinkedIn', 'Facebook'],
            'dimension': 'channel',
            'test_type': 'z_test',
            'alternative': 'two-sided',
            'alpha': 0.05
        }

        result = agent.run(df, plan)

        assert result['test_type'] == 'z_test'
        assert 'p_value' in result
        assert 'z_stat' in result
        # 500/10000 vs 300/10000 should be highly significant
        assert result['p_value'] < 0.001
        assert len(result['group_stats']) == 2
        assert result['group_stats'][0]['group'] == 'LinkedIn'
        assert result['group_stats'][0]['rate'] == pytest.approx(0.05, abs=0.001)

    def test_run_proportion_test_cvr(self, agent):
        """Test proportion test with CVR data."""
        df = pd.DataFrame({
            'channel': ['Google', 'TikTok'],
            'conversions': [100, 50],
            'sessions': [2000, 2000],
            'cvr': [0.05, 0.025]
        })

        plan = {
            'metric': 'cvr',
            'metric_type': 'proportion',
            'groups': ['Google', 'TikTok'],
            'dimension': 'channel',
            'test_type': 'z_test',
            'alternative': 'two-sided',
            'alpha': 0.05
        }

        result = agent.run(df, plan)

        assert result['test_type'] == 'z_test'
        assert 'p_value' in result
        # With n=2000 each, 5% vs 2.5% should be significant
        assert result['p_value'] < 0.05
        assert result['effect']['abs'] == pytest.approx(0.025, abs=0.001)
        assert result['effect']['rel'] > 0  # Google has higher CVR


class TestTTest:
    """Test Welch's t-test for continuous metrics."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'min_sample_size': 30},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_run_t_test_aov(self, agent):
        """Test t-test with AOV data."""
        np.random.seed(42)

        # Mobile: lower AOV
        mobile_aov = np.random.normal(loc=80, scale=20, size=500)

        # Desktop: higher AOV
        desktop_aov = np.random.normal(loc=100, scale=25, size=500)

        df = pd.DataFrame({
            'device': ['mobile'] * 500 + ['desktop'] * 500,
            'aov': np.concatenate([mobile_aov, desktop_aov])
        })

        plan = {
            'metric': 'aov',
            'metric_type': 'continuous',
            'groups': ['mobile', 'desktop'],
            'dimension': 'device',
            'test_type': 't_test',
            'alternative': 'two-sided',
            'alpha': 0.05
        }

        result = agent.run(df, plan)

        assert result['test_type'] == 't_test'
        assert 'p_value' in result
        assert 't_stat' in result
        assert result['p_value'] < 0.05  # Significant difference
        assert len(result['group_stats']) == 2
        assert result['group_stats'][0]['mean'] < result['group_stats'][1]['mean']
        assert 'confidence_interval' in result


class TestChiSquared:
    """Test chi-squared test for multiple proportions."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'min_sample_size': 30},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_run_chi_squared_multiple_channels(self, agent):
        """Test chi-squared with multiple channels."""
        df = pd.DataFrame({
            'channel': ['Google', 'Facebook', 'LinkedIn', 'TikTok'],
            'clicks': [1000, 800, 500, 300],
            'impressions': [20000, 20000, 10000, 10000],
            'ctr': [0.05, 0.04, 0.05, 0.03]
        })

        plan = {
            'metric': 'ctr',
            'metric_type': 'proportion',
            'groups': ['Google', 'Facebook', 'LinkedIn', 'TikTok'],
            'dimension': 'channel',
            'test_type': 'chi_squared',
            'alternative': 'two_sided',
            'alpha': 0.05,
            'corrections': True
        }

        result = agent.run(df, plan)

        assert result['test_type'] == 'chi_squared'
        assert 'p_value' in result
        assert 'chi2_stat' in result
        assert 'dof' in result
        assert len(result['group_stats']) == 4
        assert 'pairwise' in result
        assert result['pairwise'] is not None
        assert len(result['pairwise']['comparisons']) == 6  # 4 choose 2

    def test_chi_squared_fdr_correction(self, agent):
        """Test that FDR correction is applied to pairwise comparisons."""
        df = pd.DataFrame({
            'channel': ['Google', 'Facebook', 'LinkedIn'],
            'conversions': [100, 95, 90],
            'sessions': [2000, 2000, 2000],
            'cvr': [0.05, 0.0475, 0.045]
        })

        plan = {
            'metric': 'cvr',
            'metric_type': 'proportion',
            'groups': ['Google', 'Facebook', 'LinkedIn'],
            'dimension': 'channel',
            'test_type': 'chi_squared',
            'alternative': 'two_sided',
            'alpha': 0.05,
            'corrections': True
        }

        result = agent.run(df, plan)

        assert result['pairwise'] is not None
        assert 'adjusted_p_values' in result['pairwise']
        assert len(result['pairwise']['adjusted_p_values']) == 3  # 3 choose 2


class TestANOVA:
    """Test one-way ANOVA for continuous metrics with >2 groups."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'min_sample_size': 30},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_run_anova_revenue(self, agent):
        """Test ANOVA with revenue data."""
        np.random.seed(42)

        # Three channels with different means
        google_rev = np.random.normal(loc=100, scale=20, size=300)
        facebook_rev = np.random.normal(loc=90, scale=20, size=300)
        linkedin_rev = np.random.normal(loc=110, scale=20, size=300)

        df = pd.DataFrame({
            'channel': ['Google'] * 300 + ['Facebook'] * 300 + ['LinkedIn'] * 300,
            'revenue': np.concatenate([google_rev, facebook_rev, linkedin_rev])
        })

        plan = {
            'metric': 'revenue',
            'metric_type': 'continuous',
            'groups': ['Google', 'Facebook', 'LinkedIn'],
            'dimension': 'channel',
            'test_type': 'anova',
            'alternative': 'two_sided',
            'alpha': 0.05,
            'corrections': True
        }

        result = agent.run(df, plan)

        assert result['test_type'] == 'anova'
        assert 'p_value' in result
        assert 'f_stat' in result
        assert result['p_value'] < 0.05  # Significant difference
        assert len(result['group_stats']) == 3
        assert 'pairwise' in result
        assert len(result['pairwise']['comparisons']) == 3  # 3 choose 2


class TestReportGeneration:
    """Test report generation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05, 'min_sample_size': 30},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_report_significant_difference(self, agent):
        """Test report for significant difference."""
        results = {
            'test_type': 'z_test',
            'p_value': 0.001,
            'z_stat': 3.29,
            'effect': {'abs': 0.02, 'rel': 0.66},
            'group_stats': [
                {'group': 'LinkedIn', 'rate': 0.05, 'successes': 500, 'trials': 10000},
                {'group': 'Facebook', 'rate': 0.03, 'successes': 300, 'trials': 10000}
            ],
            'confidence_level': 0.05
        }

        plan = {
            'metric': 'ctr',
            'groups': ['LinkedIn', 'Facebook'],
            'alpha': 0.05,
            'corrections': False
        }

        report = agent.report(results, plan)

        assert 'summary' in report
        assert 'significant' in report['summary'].lower()
        assert len(report['insights']) > 0
        assert len(report['next_actions']) > 0
        assert 'raw_results' in report

    def test_report_nonsignificant_difference(self, agent):
        """Test report for non-significant difference."""
        results = {
            'test_type': 't_test',
            'p_value': 0.25,
            't_stat': 1.15,
            'effect': {'abs': 2.5, 'rel': 0.025},
            'confidence_interval': (95.0, 102.5),
            'group_stats': [
                {'group': 'mobile', 'mean': 100.0, 'std': 20.0, 'n': 500},
                {'group': 'desktop', 'mean': 97.5, 'std': 22.0, 'n': 500}
            ],
            'equal_variance': True,
            'confidence_level': 0.05
        }

        plan = {
            'metric': 'aov',
            'groups': ['mobile', 'desktop'],
            'alpha': 0.05,
            'corrections': False
        }

        report = agent.report(results, plan)

        assert 'not statistically significant' in report['summary'].lower()
        assert 'No statistically significant difference' in report['insights'][0]

    def test_report_includes_caveats_for_small_sample(self, agent):
        """Test that report includes caveats for small samples."""
        results = {
            'test_type': 'z_test',
            'p_value': 0.04,
            'z_stat': 2.05,
            'effect': {'abs': 0.05, 'rel': 0.5},
            'group_stats': [
                {'group': 'A', 'rate': 0.15, 'successes': 3, 'trials': 20},
                {'group': 'B', 'rate': 0.10, 'successes': 2, 'trials': 20}
            ],
            'confidence_level': 0.05
        }

        plan = {
            'metric': 'cvr',
            'groups': ['A', 'B'],
            'alpha': 0.05,
            'corrections': False
        }

        report = agent.report(results, plan)

        assert any('small sample size' in caveat.lower() for caveat in report['caveats'])

    def test_report_includes_multiple_testing_caveat(self, agent):
        """Test that report includes multiple testing caveat."""
        results = {
            'test_type': 'chi_squared',
            'p_value': 0.01,
            'chi2_stat': 15.2,
            'dof': 3,
            'group_stats': [
                {'group': 'G1', 'rate': 0.05, 'successes': 100, 'trials': 2000},
                {'group': 'G2', 'rate': 0.04, 'successes': 80, 'trials': 2000},
                {'group': 'G3', 'rate': 0.045, 'successes': 90, 'trials': 2000},
                {'group': 'G4', 'rate': 0.035, 'successes': 70, 'trials': 2000}
            ],
            'pairwise': {
                'comparisons': ['G1 vs G2', 'G1 vs G3', 'G1 vs G4', 'G2 vs G3', 'G2 vs G4', 'G3 vs G4'],
                'p_values': [0.1, 0.2, 0.01, 0.3, 0.05, 0.08],
                'adjusted_p_values': [0.15, 0.24, 0.06, 0.30, 0.08, 0.12],
                'significant': [False, False, True, False, False, False]
            },
            'confidence_level': 0.05
        }

        plan = {
            'metric': 'ctr',
            'groups': ['G1', 'G2', 'G3', 'G4'],
            'alpha': 0.05,
            'corrections': True
        }

        report = agent.report(results, plan)

        assert any('multiple comparisons' in caveat.lower() for caveat in report['caveats'])


class TestExtractors:
    """Test helper extraction methods."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_extract_metric_ctr(self, agent):
        """Test CTR metric extraction."""
        assert agent._extract_metric("What's the CTR for Google?") == 'ctr'
        assert agent._extract_metric("Compare click-through rates") == 'ctr'

    def test_extract_metric_cvr(self, agent):
        """Test CVR metric extraction."""
        assert agent._extract_metric("What's the conversion rate?") == 'cvr'
        assert agent._extract_metric("Compare CVR across channels") == 'cvr'

    def test_extract_metric_aov(self, agent):
        """Test AOV metric extraction."""
        assert agent._extract_metric("Is AOV higher on desktop?") == 'aov'
        assert agent._extract_metric("Compare average order value") == 'aov'

    def test_extract_dimension_channel(self, agent):
        """Test channel dimension extraction."""
        dim = agent._extract_dimension("Compare across channels")
        assert 'channel' in dim.lower()

    def test_extract_dimension_device(self, agent):
        """Test device dimension extraction."""
        dim = agent._extract_dimension("Compare mobile vs desktop")
        assert 'device' in dim.lower()

    def test_extract_groups(self, agent):
        """Test group extraction."""
        groups = agent._extract_groups("LinkedIn vs Facebook", "c.channel")
        assert 'LinkedIn' in groups
        assert 'Facebook' in groups

    def test_extract_window_days(self, agent):
        """Test window extraction with days."""
        assert agent._extract_window("in the last 30 days") == 30
        assert agent._extract_window("last 60 days") == 60

    def test_extract_window_weeks(self, agent):
        """Test window extraction with weeks."""
        assert agent._extract_window("in the last 2 weeks") == 14

    def test_extract_window_months(self, agent):
        """Test window extraction with months."""
        assert agent._extract_window("last 3 months") == 90

    def test_extract_alternative_greater(self, agent):
        """Test alternative extraction for greater."""
        assert agent._extract_alternative("Is A better than B?") == 'greater'
        assert agent._extract_alternative("Is A higher than B?") == 'greater'

    def test_extract_alternative_less(self, agent):
        """Test alternative extraction for less."""
        assert agent._extract_alternative("Is A worse than B?") == 'less'
        assert agent._extract_alternative("Is A lower than B?") == 'less'

    def test_extract_alternative_two_sided(self, agent):
        """Test alternative extraction for two-sided."""
        assert agent._extract_alternative("Compare A and B") == 'two-sided'
        assert agent._extract_alternative("Is A different from B?") == 'two-sided'


class TestSQLGeneration:
    """Test SQL query generation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'stats': {'alpha': 0.05},
            'row_caps': {'hypothesis_max_rows': 100000},
            'hypothesis': {'multiple_comparison_correction': True},
            'windows': {'hypothesis_days': 90},
            'sampling': {'strata_dimensions': []}
        }
        return HypothesisTestingAgent(mock_db, config=config)

    def test_build_sql_ctr(self, agent):
        """Test SQL generation for CTR."""
        sql = agent._build_sql('ctr', 'c.channel', 90, 'proportion', 2)

        assert 'fact_ad_spend' in sql
        assert 'clicks' in sql.lower()
        assert 'impressions' in sql.lower()
        assert 'c.channel' in sql
        assert "90" in sql

    def test_build_sql_cvr(self, agent):
        """Test SQL generation for CVR."""
        sql = agent._build_sql('cvr', 's.device', 30, 'proportion', 2)

        assert 'fact_sessions' in sql
        assert 'converted_flag' in sql
        assert 's.device' in sql
        assert "30" in sql

    def test_build_sql_aov(self, agent):
        """Test SQL generation for AOV."""
        sql = agent._build_sql('aov', 's.device', 60, 'continuous', 2)

        assert 'fact_orders' in sql
        assert 'revenue' in sql.lower()
        assert 's.device' in sql
        assert "60" in sql
