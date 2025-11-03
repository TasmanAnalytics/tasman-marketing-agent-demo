"""Tests for local triage module."""

import pytest
from core.triage_local import LocalTriage


@pytest.fixture
def business_context():
    """Sample business context for testing."""
    return {
        'roles': {
            'marketer': {
                'kpis': ['spend', 'impressions', 'clicks', 'ctr'],
                'dims': ['channel', 'campaign_name'],
                'synonyms': {
                    'spend': ['cost', 'ad_spend'],
                    'impressions': ['views']
                },
                'defaults': {'time_window_days': 90}
            }
        }
    }


@pytest.fixture
def triage(business_context):
    """Create LocalTriage instance."""
    return LocalTriage(business_context)


def test_search_classification(triage):
    """Test that search queries are classified correctly."""
    search_queries = [
        "show me revenue",
        "how many orders",
        "what is the total spend",
        "list campaigns",
        "plot spend by channel"
    ]

    for query in search_queries:
        result = triage.triage(query)
        assert result['mode'] == 'search', f"Expected 'search' for query: {query}"
        assert result['confidence'] > 0.5


def test_analysis_classification(triage):
    """Test that analysis queries are classified correctly."""
    analysis_queries = [
        "why did revenue drop",
        "identify drivers of conversion",
        "segment customers by behavior",
        "compare performance across channels"
    ]

    for query in analysis_queries:
        result = triage.triage(query)
        assert result['mode'] == 'analysis', f"Expected 'analysis' for query: {query}"
        assert result['analysis_type'] is not None


def test_role_inference(triage):
    """Test that role can be inferred from question."""
    query = "show me spend and impressions by channel"
    result = triage.triage(query)

    assert result['inferred_role'] == 'marketer'


def test_confidence_scoring(triage):
    """Test that confidence scores are reasonable."""
    clear_query = "show me total revenue by month"
    result = triage.triage(clear_query)

    assert 0.0 <= result['confidence'] <= 1.0
    assert result['confidence'] > 0.5  # Should be high confidence


def test_ambiguous_query(triage):
    """Test handling of ambiguous queries."""
    ambiguous_query = "tell me about the data"
    result = triage.triage(ambiguous_query)

    assert result['confidence'] < 0.6  # Should be low confidence
    assert result['mode'] in ['search', 'analysis']
