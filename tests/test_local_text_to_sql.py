"""Tests for local text-to-SQL module."""

import pytest
from core.local_text_to_sql import LocalTextToSQL


@pytest.fixture
def schema():
    """Sample schema for testing."""
    return {
        'fact_ad_spend': {
            'date': 'DATE',
            'campaign_id': 'INT',
            'spend': 'FLOAT',
            'impressions': 'INT',
            'clicks': 'INT'
        },
        'dim_campaigns': {
            'campaign_id': 'INT',
            'channel': 'TEXT',
            'campaign_name': 'TEXT'
        }
    }


@pytest.fixture
def business_context():
    """Sample business context."""
    return {
        'roles': {
            'marketer': {
                'kpis': ['spend', 'impressions', 'clicks'],
                'dims': ['channel', 'campaign_name'],
                'synonyms': {},
                'defaults': {'time_window_days': 90}
            }
        },
        'defaults': {
            'time_window_days': 90,
            'limit': 1000
        }
    }


@pytest.fixture
def templates():
    """Sample SQL templates."""
    return [
        {
            'id': 'spend_by_channel',
            'utterances': ['spend by channel', 'channel spend'],
            'role_hint': 'marketer',
            'sql': '''
                SELECT c.channel, SUM(f.spend) AS total_spend
                FROM fact_ad_spend f
                JOIN dim_campaigns c ON f.campaign_id = c.campaign_id
                WHERE f.date >= CURRENT_DATE - INTERVAL {{time_window_days}} DAY
                GROUP BY 1
                LIMIT {{limit}};
            ''',
            'dims': ['channel'],
            'measures': ['spend']
        }
    ]


@pytest.fixture
def engine(templates, schema, business_context):
    """Create LocalTextToSQL instance."""
    return LocalTextToSQL(templates, schema, business_context, default_limit=1000)


def test_template_matching(engine):
    """Test that templates are matched correctly."""
    result = engine.generate_sql("spend by channel", role="marketer")

    assert result['sql'] is not None
    assert result['template_id'] == 'spend_by_channel'
    assert result['method'] == 'template_match'
    assert result['confidence'] > 0.6


def test_parameter_filling(engine):
    """Test that template parameters are filled correctly."""
    result = engine.generate_sql("spend by channel", role="marketer")

    sql = result['sql']
    assert '{{time_window_days}}' not in sql
    assert '{{limit}}' not in sql
    assert '90' in sql  # Default time window
    assert '1000' in sql  # Default limit


def test_schema_validation(engine):
    """Test that SQL is validated against schema."""
    # Valid SQL
    valid_sql = "SELECT c.channel, f.spend FROM fact_ad_spend f JOIN dim_campaigns c ON f.campaign_id = c.campaign_id"
    is_valid, errors = engine.validate_sql_schema(valid_sql)
    assert is_valid
    assert len(errors) == 0

    # Invalid SQL (unknown table)
    invalid_sql = "SELECT * FROM unknown_table"
    is_valid, errors = engine.validate_sql_schema(invalid_sql)
    assert not is_valid
    assert len(errors) > 0


def test_entity_extraction(engine):
    """Test that entities are extracted from questions."""
    question = "show me spend and impressions by channel for last 30 days"
    entities = engine.extract_entities(question)

    assert 'spend' in entities['kpis']
    assert 'impressions' in entities['kpis']
    assert 'channel' in entities['dimensions']
    assert 30 in entities['time_ranges']


def test_no_template_match(engine):
    """Test handling when no template matches."""
    result = engine.generate_sql("some completely unrelated question", role="marketer")

    assert result['sql'] is None
    assert result['confidence'] == 0.0
    assert result['method'] == 'no_match'


def test_llm_threshold(engine):
    """Test that LLM should be called for low confidence."""
    result = engine.generate_sql("completely unknown query pattern", role="marketer")

    should_call = engine.should_call_llm(result, threshold=0.6)
    assert should_call
