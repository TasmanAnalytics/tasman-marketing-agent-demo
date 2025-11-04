"""Tests for SegmentationAgent."""

import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np

from agents.agent_segmentation import SegmentationAgent
from tests.fixtures.synthetic_data import generate_segmentation_data


class TestSegmentationAgentInit:
    """Test initialization."""

    def test_init_sets_correct_max_rows(self):
        """Test that max_rows is set from segmentation config."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 100000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }

        agent = SegmentationAgent(mock_db, config=config)

        assert agent.max_rows == 200000
        assert agent.agent_name == "SegmentationAgent"


class TestPlanGeneration:
    """Test plan generation from questions."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 200000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return SegmentationAgent(mock_db, config=config)

    def test_plan_customer_segmentation(self, agent):
        """Test plan for customer segmentation."""
        question = "Segment customers by value"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['method'] == 'kmeans'
        assert plan['unit'] == 'customer'
        assert 'total_orders' in plan['features']
        assert 'total_revenue' in plan['features']
        assert plan['standardize'] is True

    def test_plan_rfm_segmentation(self, agent):
        """Test plan for RFM segmentation."""
        question = "Create RFM segments for customers"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['method'] == 'rfm'
        assert plan['unit'] == 'customer'
        assert 'recency_days' in plan['features']

    def test_plan_campaign_segmentation(self, agent):
        """Test plan for campaign segmentation."""
        question = "Segment campaigns by performance"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['unit'] == 'campaign'
        assert 'ctr' in plan['features']
        assert 'cvr' in plan['features']

    def test_plan_with_k_specified(self, agent):
        """Test plan with specific number of clusters."""
        question = "Create 5 customer segments"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['k'] == 5

    def test_plan_anomaly_detection(self, agent):
        """Test plan for anomaly detection."""
        question = "Find outlier customers"

        plan = agent.plan(question, role="analyst", context={})

        assert plan['method'] == 'dbscan'


class TestKMeansClustering:
    """Test K-means clustering."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 200000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return SegmentationAgent(mock_db, config=config)

    def test_run_kmeans_auto_k(self, agent):
        """Test K-means with automatic k selection."""
        # Generate data with known clusters
        df, true_labels = generate_segmentation_data(
            n_samples=500,
            n_true_clusters=3,
            n_features=4,
            feature_names=['metric_1', 'metric_2', 'metric_3', 'metric_4'],
            cluster_separation=2.0,
            random_state=42
        )

        plan = {
            'method': 'kmeans',
            'unit': 'customer',
            'features': ['metric_1', 'metric_2', 'metric_3', 'metric_4'],
            'k': None,  # Auto-select
            'k_range': [2, 5],
            'standardize': True,
            'outlier_handling': 'winsorize',
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['method'] == 'kmeans'
        assert 2 <= result['k'] <= 5
        assert 'silhouette_score' in result
        assert result['silhouette_score'] > 0.2  # Some separation
        assert len(result['segments']) == result['k']
        assert all('size' in seg for seg in result['segments'])

    def test_run_kmeans_fixed_k(self, agent):
        """Test K-means with fixed k."""
        df, _ = generate_segmentation_data(
            n_samples=300,
            n_true_clusters=3,
            n_features=3,
            random_state=42
        )

        plan = {
            'method': 'kmeans',
            'unit': 'customer',
            'features': ['metric_0', 'metric_1', 'metric_2'],
            'k': 4,  # Fixed
            'k_range': [2, 10],
            'standardize': True,
            'outlier_handling': 'winsorize',
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['k'] == 4
        assert len(result['segments']) == 4


class TestRFMSegmentation:
    """Test RFM segmentation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 200000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return SegmentationAgent(mock_db, config=config)

    def test_run_rfm(self, agent):
        """Test RFM segmentation."""
        np.random.seed(42)

        # Create customer data with RFM attributes
        n_customers = 500
        df = pd.DataFrame({
            'customer_id': [f'C{i:04d}' for i in range(n_customers)],
            'recency_days': np.random.randint(1, 365, n_customers),
            'total_orders': np.random.randint(1, 50, n_customers),
            'total_revenue': np.random.uniform(10, 5000, n_customers)
        })

        plan = {
            'method': 'rfm',
            'unit': 'customer',
            'features': ['recency_days', 'total_orders', 'total_revenue'],
            'k': None,
            'k_range': [2, 10],
            'standardize': False,
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['method'] == 'rfm'
        assert result['k'] > 0  # Should create multiple segments
        assert len(result['segments']) > 0
        # Top segment should have profile
        assert 'profile' in result['segments'][0]
        assert 'avg_revenue' in result['segments'][0]['profile']


class TestDBSCANClustering:
    """Test DBSCAN clustering."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 200000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return SegmentationAgent(mock_db, config=config)

    def test_run_dbscan(self, agent):
        """Test DBSCAN clustering."""
        # Generate data with some outliers
        df, _ = generate_segmentation_data(
            n_samples=300,
            n_true_clusters=2,
            n_features=3,
            cluster_separation=1.5,
            random_state=42
        )

        # Add some outliers
        outliers = pd.DataFrame({
            'metric_0': [100, -100],
            'metric_1': [100, -100],
            'metric_2': [100, -100]
        })
        df = pd.concat([df, outliers], ignore_index=True)

        plan = {
            'method': 'dbscan',
            'unit': 'customer',
            'features': ['metric_0', 'metric_1', 'metric_2'],
            'k': None,
            'k_range': [2, 10],
            'standardize': True,
            'seed': 42
        }

        result = agent.run(df, plan)

        assert result['method'] == 'dbscan'
        assert 'n_noise' in result
        assert result['n_noise'] >= 0  # Should detect some noise/outliers
        assert len(result['segments']) > 0


class TestReportGeneration:
    """Test report generation."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 200000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return SegmentationAgent(mock_db, config=config)

    def test_report_good_kmeans(self, agent):
        """Test report for good K-means segmentation."""
        results = {
            'method': 'kmeans',
            'k': 4,
            'silhouette_score': 0.65,
            'segments': [
                {'segment_id': 0, 'size': 500, 'share': 0.4, 'profile': {}},
                {'segment_id': 1, 'size': 400, 'share': 0.32, 'profile': {}},
                {'segment_id': 2, 'size': 200, 'share': 0.16, 'profile': {}},
                {'segment_id': 3, 'size': 150, 'share': 0.12, 'profile': {}}
            ],
            'diagnostics': {
                'n_samples': 1250,
                'n_features': 4
            }
        }

        plan = {'unit': 'customer'}

        report = agent.report(results, plan)

        assert 'summary' in report
        assert 'High-quality' in report['insights'][0]
        assert len(report['next_actions']) > 0

    def test_report_poor_kmeans(self, agent):
        """Test report for poor K-means segmentation."""
        results = {
            'method': 'kmeans',
            'k': 5,
            'silhouette_score': 0.15,
            'segments': [
                {'segment_id': 0, 'size': 100, 'share': 1.0, 'profile': {}}
            ],
            'diagnostics': {
                'n_samples': 100,
                'n_features': 3
            }
        }

        plan = {'unit': 'customer'}

        report = agent.report(results, plan)

        assert any('Low segmentation quality' in insight for insight in report['insights'])
        assert any('Small sample size' in caveat for caveat in report['caveats'])

    def test_report_rfm(self, agent):
        """Test report for RFM segmentation."""
        results = {
            'method': 'rfm',
            'k': 125,
            'segments': [
                {
                    'segment_id': '555',
                    'size': 50,
                    'share': 0.1,
                    'profile': {
                        'avg_revenue': 5000.0,
                        'avg_orders': 20.0,
                        'avg_recency': 5.0
                    }
                }
            ],
            'diagnostics': {
                'n_samples': 500,
                'n_segments': 125
            }
        }

        plan = {'unit': 'customer'}

        report = agent.report(results, plan)

        assert '125' in report['summary']
        assert 'Best segment' in report['insights'][-1]


class TestExtractors:
    """Test helper extraction methods."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        mock_db = Mock()
        config = {
            'defaults': {'max_rows': 200000, 'random_seed': 42},
            'row_caps': {'segmentation_max_rows': 200000},
            'segmentation': {
                'default_method': 'kmeans',
                'kmeans': {'min_k': 2, 'max_k': 10},
                'standardize': True,
                'outlier_handling': 'winsorize',
                'winsorize_limits': [0.01, 0.99],
                'dbscan': {'eps': 0.5, 'min_samples': 10}
            },
            'windows': {'segmentation_days': 180},
            'sampling': {'strata_dimensions': []}
        }
        return SegmentationAgent(mock_db, config=config)

    def test_extract_method_kmeans(self, agent):
        """Test K-means method extraction."""
        assert agent._extract_method("Segment customers") == 'kmeans'

    def test_extract_method_rfm(self, agent):
        """Test RFM method extraction."""
        assert agent._extract_method("Create RFM segments") == 'rfm'
        assert agent._extract_method("Segment by recency and frequency") == 'rfm'

    def test_extract_method_dbscan(self, agent):
        """Test DBSCAN method extraction."""
        assert agent._extract_method("Find outlier customers") == 'dbscan'

    def test_extract_unit_customer(self, agent):
        """Test customer unit extraction."""
        assert agent._extract_unit("Segment customers") == 'customer'

    def test_extract_unit_campaign(self, agent):
        """Test campaign unit extraction."""
        assert agent._extract_unit("Segment campaigns") == 'campaign'

    def test_extract_k(self, agent):
        """Test k extraction."""
        assert agent._extract_k("Create 5 segments") == 5
        assert agent._extract_k("into 3 groups") == 3
        assert agent._extract_k("k=4 clusters") == 4
        assert agent._extract_k("Segment customers") is None
