"""
Database connector - provides read-only DuckDB connections with validation.
"""

import duckdb
from typing import List, Dict, Any


def get_db_connection(db_path: str, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB connection.

    Args:
        db_path: Path to the DuckDB database file
        read_only: If True, open in read-only mode (default: True)

    Returns:
        DuckDB connection object

    Raises:
        FileNotFoundError: If database file doesn't exist
        Exception: If connection fails
    """
    try:
        conn = duckdb.connect(db_path, read_only=read_only)
        return conn
    except Exception as e:
        raise Exception(f"Failed to connect to database at {db_path}: {e}")


def validate_schema(conn: duckdb.DuckDBPyConnection) -> Dict[str, Any]:
    """
    Validate that required tables and columns exist in the database.

    Args:
        conn: DuckDB connection

    Returns:
        Dictionary with validation results and schema info

    Raises:
        ValueError: If required tables or columns are missing
    """
    # Expected tables and their required columns
    expected_schema = {
        'dim_campaigns': ['campaign_id', 'campaign_name', 'channel'],
        'dim_adgroups': ['adgroup_id', 'campaign_id'],
        'dim_creatives': ['creative_id', 'adgroup_id'],
        'dim_products': ['product_id'],
        'dim_customers': ['customer_id', 'region'],
        'fact_ad_spend': ['date', 'campaign_id', 'spend', 'impressions', 'clicks'],
        'fact_sessions': ['session_id', 'campaign_id', 'session_start', 'converted_flag', 'device_type'],
        'fact_orders': ['order_id', 'session_id', 'order_timestamp', 'revenue'],
    }

    # Get actual tables
    tables_result = conn.execute("SHOW TABLES").fetchall()
    actual_tables = {row[0] for row in tables_result}

    # Check for missing tables
    missing_tables = set(expected_schema.keys()) - actual_tables
    if missing_tables:
        raise ValueError(f"Missing required tables: {', '.join(sorted(missing_tables))}")

    # Check columns for each table
    schema_info = {}
    missing_columns = {}

    for table, required_cols in expected_schema.items():
        # Get actual columns
        cols_result = conn.execute(f"DESCRIBE {table}").fetchall()
        actual_cols = {row[0] for row in cols_result}

        # Check for missing columns
        missing = set(required_cols) - actual_cols
        if missing:
            missing_columns[table] = missing

        schema_info[table] = {
            'columns': list(actual_cols),
            'required_present': not bool(missing)
        }

    if missing_columns:
        error_msg = "Missing required columns:\n"
        for table, cols in missing_columns.items():
            error_msg += f"  {table}: {', '.join(sorted(cols))}\n"
        raise ValueError(error_msg)

    return {
        'valid': True,
        'tables': list(actual_tables),
        'schema_info': schema_info
    }


def get_table_row_counts(conn: duckdb.DuckDBPyConnection) -> Dict[str, int]:
    """
    Get row counts for all tables in the database.

    Args:
        conn: DuckDB connection

    Returns:
        Dictionary mapping table names to row counts
    """
    tables_result = conn.execute("SHOW TABLES").fetchall()
    tables = [row[0] for row in tables_result]

    row_counts = {}
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        row_counts[table] = count

    return row_counts
