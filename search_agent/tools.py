"""LangChain tool wrappers around database_utils for the search agent."""

from __future__ import annotations

import json
import re
from collections import defaultdict

from langchain_core.tools import tool


def _sanitize_fts(query: str) -> str:
    """Strip characters that cause FTS5 parse errors (colon filters, operators, etc.)."""
    return re.sub(r'[-":*()\[\]{}|&^!]', ' ', query).strip()

from database_utils.explore import (
    artifacts_for_customer,
    scenario_summary,
    search_artifacts as _search_artifacts,
)
from database_utils.query import run


@tool
def search_artifacts(query: str, limit: int = 10) -> str:
    """Full-text keyword search across artifact titles, summaries, and content.

    Use this as your first step for most questions. Returns a flat list of
    matching artifacts (id, type, title, summary, customer_id, scenario_id).
    Pass plain keywords — e.g. 'schema rollout', 'mTLS certificate'.

    Args:
        query: Plain keywords to search for (special characters are stripped automatically).
        limit: Maximum number of results (default 10).
    """
    rows = _search_artifacts(_sanitize_fts(query), limit=limit)
    return json.dumps([dict(r) for r in rows], indent=2)


@tool
def find_pattern_across_customers(query: str, limit: int = 50) -> str:
    """FTS search grouped by customer — use this to answer 'is this widespread or a one-off?'.

    Runs a full-text search and groups results by customer, returning a count
    of affected customers and the artifact titles per customer. Always use this
    before concluding that an issue is isolated to a single customer.

    Args:
        query: Keywords or FTS5 expression describing the pattern.
        limit: Maximum FTS hits to scan before grouping (default 50).
    """
    rows = run(
        """
        SELECT a.artifact_id, a.artifact_type, a.title, a.customer_id, c.name AS customer_name
        FROM artifacts_fts f
        JOIN artifacts a ON a.artifact_id = f.artifact_id
        JOIN customers c ON c.customer_id = a.customer_id
        WHERE artifacts_fts MATCH ?
        LIMIT ?
        """,
        params=(_sanitize_fts(query), limit),
    )
    grouped: dict[str, dict] = defaultdict(lambda: {"count": 0, "artifacts": []})
    for row in rows:
        name = row["customer_name"]
        grouped[name]["count"] += 1
        grouped[name]["artifacts"].append(row["title"])

    result = {
        "customers_affected": len(grouped),
        "by_customer": {k: v for k, v in sorted(grouped.items())},
    }
    return json.dumps(result, indent=2)


@tool
def customer_artifacts(name_or_id: str) -> str:
    """Return all artifacts linked to a specific customer (by name or customer_id).

    Use this when you already know which customer to investigate and want a
    complete list of their artifacts (id, type, title, summary, created_at).

    Args:
        name_or_id: Customer name (e.g. 'MapleHarvest Grocers') or customer_id.
    """
    rows = artifacts_for_customer(name_or_id)
    return json.dumps([dict(r) for r in rows], indent=2)


@tool
def scenario_summary_tool(scenario_id: str) -> str:
    """Return everything about a scenario in one shot: scenario details, customer,
    implementation, and a list of all associated artifacts.

    Use this when you have a scenario_id and want full context without making
    multiple separate calls.

    Args:
        scenario_id: The scenario_id to look up (e.g. 'scn_abc123').
    """
    result = scenario_summary(scenario_id)
    return json.dumps({k: dict(v) if hasattr(v, "keys") else v for k, v in result.items()}, indent=2)


@tool
def artifact_full_text(artifact_id: str) -> str:
    """Fetch the full content_text of a single artifact by its artifact_id.

    Use this after search_artifacts or customer_artifacts to read the actual
    document content. Artifacts are 300–1500 tokens, so fetch only the ones
    directly relevant to the answer.

    Args:
        artifact_id: The artifact_id to fetch (e.g. 'art_xyz789').
    """
    rows = run(
        "SELECT artifact_id, title, artifact_type, content_text FROM artifacts WHERE artifact_id = ?",
        params=(artifact_id,),
    )
    if not rows:
        return json.dumps({"error": f"No artifact found with id '{artifact_id}'"})
    return json.dumps(dict(rows[0]), indent=2)


@tool
def sql_query(sql: str) -> str:
    """Execute a read-only SQL SELECT query against the database.

    Use this for structured questions that require filtering, aggregation, or
    joins that the other tools don't cover — e.g. counting implementations by
    status, filtering by contract value, or joining across multiple tables.

    Only SELECT statements are allowed.

    Key tables: scenarios, customers, implementations, artifacts, products,
    company_profile, competitors, employees.

    Args:
        sql: A SQL SELECT statement.
    """
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT statements are permitted."})
    rows = run(sql)
    return json.dumps([dict(r) for r in rows], indent=2)
