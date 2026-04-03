"""Low-level query helpers that return plain Python structures."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .connection import get_connection


def run(
    sql: str,
    params: tuple = (),
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Execute arbitrary SQL and return results as a list of dicts.

    Args:
        sql: SQL statement to execute.
        params: Optional positional parameters for parameterised queries.
        db_path: Path to SQLite file (defaults to project database).

    Example:
        >>> from database_utils import run
        >>> run("SELECT name, industry FROM customers LIMIT 3")
    """
    conn = get_connection(db_path)
    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def query_df(
    sql: str,
    params: tuple = (),
    db_path: str | Path | None = None,
):
    """Execute SQL and return a pandas DataFrame.

    Requires pandas to be installed.

    Example:
        >>> from database_utils import query_df
        >>> df = query_df("SELECT * FROM scenarios")
        >>> df.head()
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas is required for query_df. Run: pip install pandas") from e

    import sqlite3 as _sqlite3
    from .connection import _DEFAULT_DB

    path = Path(db_path) if db_path else _DEFAULT_DB
    conn = _sqlite3.connect(path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
