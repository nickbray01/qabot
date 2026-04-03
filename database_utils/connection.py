"""Shared SQLite connection factory."""

import sqlite3
from pathlib import Path

_DEFAULT_DB = Path(__file__).parent.parent / "data" / "synthetic_startup.sqlite"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set to Row.

    Args:
        db_path: Path to the SQLite file. Defaults to the project database.
    """
    path = Path(db_path) if db_path else _DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn
