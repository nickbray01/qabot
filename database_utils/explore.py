"""High-level exploration helpers for the synthetic_startup database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .query import run


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_SKIP_TABLES = {"sqlite_sequence", "sqlite_stat1"}
_FTS_SUFFIX = ("_fts", "_fts_config", "_fts_content", "_fts_data",
               "_fts_idx", "_fts_docsize")


def _is_fts(name: str) -> bool:
    return any(name.endswith(s) or name == s[1:] for s in _FTS_SUFFIX) or \
           "_fts" in name


def tables(db_path: str | Path | None = None, include_fts: bool = False) -> list[str]:
    """Return a list of table names in the database.

    Args:
        include_fts: If False (default) the FTS shadow tables are excluded.

    Example:
        >>> from database_utils import tables
        >>> tables()
        ['artifacts', 'company_profile', 'competitors', ...]
    """
    rows = run(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        db_path=db_path,
    )
    result = [r["name"] for r in rows if r["name"] not in _SKIP_TABLES]
    if not include_fts:
        result = [t for t in result if not _is_fts(t)]
    return result


def schema(table: str, db_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return column info for a table (name, type, not-null, default, pk).

    Example:
        >>> from database_utils import schema
        >>> schema("artifacts")
        [{'cid': 0, 'name': 'artifact_id', 'type': 'TEXT', ...}, ...]
    """
    rows = run(f"PRAGMA table_info({table})", db_path=db_path)
    return [dict(r) for r in rows]


def count(table: str, db_path: str | Path | None = None) -> int:
    """Return the number of rows in a table.

    Example:
        >>> from database_utils import count
        >>> count("artifacts")
        250
    """
    rows = run(f"SELECT COUNT(*) AS n FROM {table}", db_path=db_path)
    return rows[0]["n"]


def sample(
    table: str,
    n: int = 5,
    columns: list[str] | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return up to *n* rows from a table.

    Args:
        table: Table name.
        n: Number of rows to return.
        columns: Optional list of column names to include. Defaults to all.

    Example:
        >>> from database_utils import sample
        >>> sample("customers", n=3, columns=["name", "industry", "crm_stage"])
    """
    cols = ", ".join(columns) if columns else "*"
    return run(f"SELECT {cols} FROM {table} LIMIT {n}", db_path=db_path)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _group_counts(
    table: str,
    column: str,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    rows = run(
        f"SELECT {column}, COUNT(*) AS n FROM {table} GROUP BY {column} ORDER BY n DESC",
        db_path=db_path,
    )
    return {r[column]: r["n"] for r in rows}


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def overview(db_path: str | Path | None = None) -> None:
    """Print a human-readable summary of every main table.

    Example:
        >>> from database_utils import overview
        >>> overview()
    """
    _print_separator("DATABASE OVERVIEW")

    all_tables = tables(db_path=db_path)
    print(f"Tables: {', '.join(all_tables)}\n")

    for tbl in all_tables:
        n = count(tbl, db_path=db_path)
        cols = [c["name"] for c in schema(tbl, db_path=db_path)]
        print(f"  {tbl:<25} {n:>5} rows   columns: {', '.join(cols)}")

    print()

    # --- company ---
    _print_separator("COMPANY")
    rows = run("SELECT name, category, headquarters, founding_year FROM company_profile", db_path=db_path)
    for r in rows:
        print(f"  {r['name']}  |  {r['category']}  |  HQ: {r['headquarters']}  |  Founded: {r['founding_year']}")

    # --- products ---
    _print_separator("PRODUCTS")
    for r in run("SELECT name, category, pricing_model FROM products", db_path=db_path):
        print(f"  {r['name']:<20} {r['category']:<35} pricing: {r['pricing_model']}")

    # --- competitors ---
    _print_separator("COMPETITORS")
    for r in run("SELECT name, segment, pricing_position FROM competitors ORDER BY segment", db_path=db_path):
        print(f"  {r['name']:<20} {r['segment']:<40} price: {r['pricing_position']}")

    # --- employees ---
    _print_separator("EMPLOYEES  (by department)")
    for dept, n in _group_counts("employees", "department", db_path=db_path).items():
        print(f"  {dept:<25} {n} employees")

    # --- scenarios ---
    _print_separator("SCENARIOS  (by industry)")
    for industry, n in _group_counts("scenarios", "industry", db_path=db_path).items():
        print(f"  {industry:<25} {n} scenarios")

    # --- customers ---
    _print_separator("CUSTOMERS")
    print("  CRM stages:")
    for stage, n in _group_counts("customers", "crm_stage", db_path=db_path).items():
        print(f"    {stage:<30} {n}")
    print("  Regions:")
    for region, n in _group_counts("customers", "region", db_path=db_path).items():
        print(f"    {region:<30} {n}")

    # --- implementations ---
    _print_separator("IMPLEMENTATIONS")
    rows = run(
        "SELECT MIN(contract_value) as mn, MAX(contract_value) as mx, "
        "ROUND(AVG(contract_value),0) as avg, SUM(contract_value) as total "
        "FROM implementations",
        db_path=db_path,
    )
    r = rows[0]
    print(f"  Contract values — min: ${r['mn']:,}  max: ${r['mx']:,}  "
          f"avg: ${int(r['avg']):,}  total: ${r['total']:,}")
    print("  Deployment models:")
    for model, n in _group_counts("implementations", "deployment_model", db_path=db_path).items():
        print(f"    {model:<30} {n}")

    # --- artifacts ---
    _print_separator("ARTIFACTS  (by type)")
    for atype, n in _group_counts("artifacts", "artifact_type", db_path=db_path).items():
        print(f"  {atype:<30} {n}")

    print()


# ---------------------------------------------------------------------------
# Artifact search (wraps the FTS index)
# ---------------------------------------------------------------------------

def search_artifacts(
    query: str,
    limit: int = 10,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Full-text search across artifact title, summary, and content.

    Uses the ``artifacts_fts`` virtual table.

    Args:
        query: FTS5 query string (e.g. ``"schema rollout"``).
        limit: Maximum number of results.

    Example:
        >>> from database_utils.explore import search_artifacts
        >>> search_artifacts("mTLS certificate", limit=5)
    """
    return run(
        """
        SELECT a.artifact_id, a.artifact_type, a.title, a.summary,
               a.scenario_id, a.customer_id
        FROM artifacts_fts f
        JOIN artifacts a ON a.artifact_id = f.artifact_id
        WHERE artifacts_fts MATCH ?
        LIMIT ?
        """,
        params=(query, limit),
        db_path=db_path,
    )


def artifacts_for_customer(
    customer_name_or_id: str,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return all artifacts linked to a customer (by name or id).

    Example:
        >>> from database_utils.explore import artifacts_for_customer
        >>> artifacts_for_customer("MapleHarvest Grocers")
    """
    # Try by name first, then by id
    rows = run(
        """
        SELECT a.artifact_id, a.artifact_type, a.title, a.created_at, a.summary
        FROM artifacts a
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE c.name = ? OR c.customer_id = ?
        ORDER BY a.created_at
        """,
        params=(customer_name_or_id, customer_name_or_id),
        db_path=db_path,
    )
    return rows


def scenario_summary(
    scenario_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a dict with all related records for a single scenario.

    Includes the scenario, customer, implementation, and artifact list.

    Example:
        >>> from database_utils.explore import scenario_summary
        >>> scenario_summary("scn_abc123")
    """
    scen_rows = run("SELECT * FROM scenarios WHERE scenario_id = ?",
                    params=(scenario_id,), db_path=db_path)
    if not scen_rows:
        return {}
    scen = scen_rows[0]

    cust_rows = run("SELECT * FROM customers WHERE scenario_id = ?",
                    params=(scenario_id,), db_path=db_path)
    impl_rows = run("SELECT * FROM implementations WHERE scenario_id = ?",
                    params=(scenario_id,), db_path=db_path)
    art_rows = run(
        "SELECT artifact_id, artifact_type, title, created_at, summary "
        "FROM artifacts WHERE scenario_id = ? ORDER BY created_at",
        params=(scenario_id,), db_path=db_path,
    )

    return {
        "scenario": scen,
        "customer": cust_rows[0] if cust_rows else None,
        "implementation": impl_rows[0] if impl_rows else None,
        "artifacts": art_rows,
    }


# ---------------------------------------------------------------------------
# Internal formatting
# ---------------------------------------------------------------------------

def _print_separator(title: str, width: int = 60) -> None:
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
