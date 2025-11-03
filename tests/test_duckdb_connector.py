"""Tests for DuckDB connector module."""

import pytest
import tempfile
from pathlib import Path
import duckdb
from core.duckdb_connector import DuckDBConnector


@pytest.fixture
def temp_db():
    """Create temporary DuckDB for testing."""
    # Create temp file path without creating the file
    db_path = tempfile.mktemp(suffix='.duckdb')

    # Create test database with sample table
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE test_table (
            id INTEGER,
            name VARCHAR,
            value FLOAT
        )
    """)
    conn.execute("INSERT INTO test_table VALUES (1, 'test', 10.5)")
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def connector(temp_db):
    """Create DuckDBConnector instance."""
    connector = DuckDBConnector(temp_db, default_limit=100)
    connector.connect()
    yield connector
    connector.close()


def test_connection(connector):
    """Test database connection."""
    assert connector.conn is not None


def test_list_tables(connector):
    """Test listing tables."""
    tables = connector.list_tables()
    assert 'test_table' in tables


def test_list_columns(connector):
    """Test listing columns."""
    columns = connector.list_columns('test_table')
    assert 'id' in columns
    assert 'name' in columns
    assert 'value' in columns


def test_select_only_validation(connector):
    """Test SELECT-only validation."""
    # Valid SELECT
    assert connector.is_select_only("SELECT * FROM test_table")
    assert connector.is_select_only("select id, name from test_table")

    # Invalid (DDL/DML)
    assert not connector.is_select_only("INSERT INTO test_table VALUES (2, 'x', 5)")
    assert not connector.is_select_only("UPDATE test_table SET value = 0")
    assert not connector.is_select_only("DELETE FROM test_table")
    assert not connector.is_select_only("DROP TABLE test_table")


def test_limit_injection(connector):
    """Test LIMIT injection."""
    sql = "SELECT * FROM test_table"
    sql_with_limit = connector.ensure_limit(sql, limit=50)

    assert "LIMIT 50" in sql_with_limit

    # Should not double-add LIMIT if already present
    sql_already_limited = "SELECT * FROM test_table LIMIT 100"
    sql_result = connector.ensure_limit(sql_already_limited, limit=50)
    assert sql_result.count("LIMIT") == 1
    assert "LIMIT 100" in sql_result  # Original limit preserved


def test_query_execution(connector):
    """Test query execution."""
    df = connector.execute("SELECT * FROM test_table")

    assert len(df) == 1
    assert df.iloc[0]['id'] == 1
    assert df.iloc[0]['name'] == 'test'
    assert df.iloc[0]['value'] == 10.5


def test_unsafe_query_rejection(connector):
    """Test that unsafe queries are rejected."""
    with pytest.raises(ValueError):
        connector.execute("DROP TABLE test_table")

    with pytest.raises(ValueError):
        connector.execute("INSERT INTO test_table VALUES (2, 'bad', 0)")


def test_schema_validation(connector):
    """Test schema validation."""
    schema = {
        'test_table': {
            'id': 'INTEGER',
            'name': 'VARCHAR',
            'value': 'FLOAT'
        }
    }

    is_valid, errors = connector.validate_schema(schema)
    assert is_valid
    assert len(errors) == 0

    # Test with missing table
    bad_schema = {
        'missing_table': {'col': 'INT'}
    }
    is_valid, errors = connector.validate_schema(bad_schema)
    assert not is_valid
    assert len(errors) > 0


def test_context_manager(temp_db):
    """Test context manager usage."""
    with DuckDBConnector(temp_db) as conn:
        tables = conn.list_tables()
        assert 'test_table' in tables

    # Connection should be closed after context
    assert conn.conn is None
