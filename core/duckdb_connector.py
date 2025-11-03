"""DuckDB connector with read-only execution and LIMIT enforcement."""

import re
from typing import Any, Dict, List, Tuple
import duckdb
import pandas as pd


class DuckDBConnector:
    """Read-only DuckDB connector with safety guardrails."""

    # Prohibited SQL keywords for read-only mode
    PROHIBITED_KEYWORDS = (
        "insert ", "update ", "delete ", "create ", "alter ",
        "drop ", "truncate ", "merge ", "replace "
    )

    def __init__(self, db_path: str, default_limit: int = 1000):
        """
        Initialize DuckDB connector.

        Args:
            db_path: Path to DuckDB file
            default_limit: Default LIMIT to inject if missing
        """
        self.db_path = db_path
        self.default_limit = default_limit
        self.conn = None

    def connect(self, read_only: bool = True) -> None:
        """Establish connection to DuckDB."""
        self.conn = duckdb.connect(database=self.db_path, read_only=read_only)

    def close(self) -> None:
        """Close DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def is_select_only(self, sql: str) -> bool:
        """
        Check if SQL is a safe SELECT query.

        Args:
            sql: SQL query to check

        Returns:
            True if safe SELECT query, False otherwise
        """
        sql_lower = sql.strip().lower()
        return (
            sql_lower.startswith("select ") or
            sql_lower.startswith("with ")
        ) and not any(kw in sql_lower for kw in self.PROHIBITED_KEYWORDS)

    def ensure_limit(self, sql: str, limit: int = None) -> str:
        """
        Ensure SQL has a LIMIT clause.

        Args:
            sql: SQL query
            limit: LIMIT value (uses default if not provided)

        Returns:
            SQL with LIMIT clause
        """
        limit = limit or self.default_limit
        sql_lower = sql.lower()

        # Check if LIMIT already exists
        if " limit " in sql_lower:
            return sql

        # Remove trailing semicolon and add LIMIT
        sql_clean = sql.strip().rstrip(";")
        return f"{sql_clean}\nLIMIT {limit};"

    def execute(
        self,
        sql: str,
        enforce_limit: bool = True,
        enforce_select_only: bool = True
    ) -> pd.DataFrame:
        """
        Execute SQL query with safety checks.

        Args:
            sql: SQL query to execute
            enforce_limit: Whether to enforce LIMIT clause
            enforce_select_only: Whether to enforce SELECT-only queries

        Returns:
            Query results as DataFrame

        Raises:
            ValueError: If query fails safety checks
            RuntimeError: If connection not established
        """
        if not self.conn:
            raise RuntimeError("Database connection not established. Call connect() first.")

        # Safety check: SELECT only
        if enforce_select_only and not self.is_select_only(sql):
            raise ValueError(
                "Only SELECT queries are allowed. "
                f"Query appears to contain DDL/DML: {sql[:100]}"
            )

        # Inject LIMIT if needed
        if enforce_limit:
            sql = self.ensure_limit(sql)

        # Execute query
        try:
            result = self.conn.execute(sql).fetchdf()
            return result
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}\nSQL: {sql}")

    def list_tables(self) -> List[str]:
        """List all tables in database."""
        if not self.conn:
            raise RuntimeError("Database connection not established.")
        return [row[0] for row in self.conn.execute("SHOW TABLES").fetchall()]

    def list_columns(self, table: str) -> Dict[str, str]:
        """
        List columns in a table.

        Args:
            table: Table name

        Returns:
            Dict mapping column name to type
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")

        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        # Row format: (cid, name, type, notnull, dflt_value, pk)
        return {row[1]: row[2] for row in rows}

    def validate_schema(self, schema_spec: Dict[str, Dict[str, str]]) -> Tuple[bool, List[str]]:
        """
        Validate that database schema matches specification.

        Args:
            schema_spec: Schema specification (table -> columns)

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        if not self.conn:
            raise RuntimeError("Database connection not established.")

        errors = []
        actual_tables = self.list_tables()

        for table, expected_cols in schema_spec.items():
            # Check table exists
            if table not in actual_tables:
                errors.append(f"Missing table: {table}")
                continue

            # Check columns exist
            actual_cols = self.list_columns(table)
            for col in expected_cols.keys():
                if col not in actual_cols:
                    errors.append(f"Missing column: {table}.{col}")

        return (len(errors) == 0, errors)
