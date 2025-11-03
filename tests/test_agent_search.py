"""Tests for search agent module."""

import pytest
import tempfile
from pathlib import Path
import duckdb

from agents.agent_search import SearchAgent
from agents.agent_triage import TriageAgent
from agents.agent_text_to_sql import TextToSQLAgent
from core.duckdb_connector import DuckDBConnector


@pytest.fixture
def temp_db():
    """Create temporary test database."""
    db_path = tempfile.mktemp(suffix='.duckdb')

    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE fact_ad_spend (
            date DATE,
            campaign_id INTEGER,
            spend FLOAT
        )
    """)
    conn.execute("""
        CREATE TABLE dim_campaigns (
            campaign_id INTEGER,
            channel TEXT
        )
    """)
    # Insert data within the last 90 days
    conn.execute("INSERT INTO fact_ad_spend VALUES (CURRENT_DATE - INTERVAL 30 DAY, 1, 100.0)")
    conn.execute("INSERT INTO dim_campaigns VALUES (1, 'Google')")
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def business_context():
    """Sample business context."""
    return {
        'roles': {
            'marketer': {
                'kpis': ['spend'],
                'dims': ['channel'],
                'synonyms': {},
                'defaults': {'time_window_days': 90}
            }
        },
        'defaults': {'time_window_days': 90, 'limit': 1000}
    }


@pytest.fixture
def schema():
    """Sample schema."""
    return {
        'fact_ad_spend': {
            'date': 'DATE',
            'campaign_id': 'INTEGER',
            'spend': 'FLOAT'
        },
        'dim_campaigns': {
            'campaign_id': 'INTEGER',
            'channel': 'TEXT'
        }
    }


@pytest.fixture
def templates():
    """Sample templates with LIMIT already included."""
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
def search_agent(temp_db, templates, schema, business_context):
    """Create search agent with test dependencies."""
    db_connector = DuckDBConnector(temp_db, default_limit=1000)
    db_connector.connect()

    triage_agent = TriageAgent(business_context, llm_client=None)

    text_to_sql_agent = TextToSQLAgent(
        templates=templates,
        schema=schema,
        business_context=business_context,
        llm_client=None,
        default_limit=1000
    )

    output_dir = Path(tempfile.mkdtemp())

    agent = SearchAgent(
        triage_agent=triage_agent,
        text_to_sql_agent=text_to_sql_agent,
        db_connector=db_connector,
        output_dir=output_dir
    )

    yield agent

    db_connector.close()


def test_search_agent_no_duplicate_limit(search_agent):
    """Test that search agent doesn't add duplicate LIMIT clauses."""
    result = search_agent.search(
        question="spend by channel",
        role="marketer"
    )

    # Should succeed
    assert result['status'] == 'success'

    # SQL should have exactly one LIMIT
    sql = result['sql']
    assert sql.count('LIMIT') == 1

    # Verify data was returned
    assert 'data' in result
    assert len(result['data']) > 0


def test_search_agent_template_match(search_agent):
    """Test that search agent uses template matching."""
    result = search_agent.search(
        question="channel spend",
        role="marketer"
    )

    # Should use template (no LLM)
    assert result['sql_generation']['method'] == 'template_match'
    assert result['sql_generation']['template_id'] == 'spend_by_channel'

    # Should not call LLM
    assert result['sql_generation']['used_llm'] is False


def test_search_agent_execution_success(search_agent):
    """Test successful query execution."""
    result = search_agent.search(
        question="spend by channel",
        role="marketer"
    )

    assert result['status'] == 'success'
    assert result['row_count'] == 1
    assert 'summary' in result

    # Check steps
    steps = {s['step']: s for s in result['steps']}
    assert 'triage' in steps
    assert 'sql_generation' in steps
    assert 'execution' in steps
    assert steps['execution']['success'] is True
