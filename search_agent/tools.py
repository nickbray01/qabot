"""Three composable tools for the search agent.

search  — FTS over artifact content, returns snippets (+ optional customer grouping)
read    — fetch full content for one or more artifact IDs in a single call
lookup  — structured rows from customers / competitors / employees / artifacts tables
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from langchain_core.tools import tool

from database_utils.query import run


def _sanitize_fts(query: str) -> str:
    """Strip characters that cause FTS5 parse errors."""
    return re.sub(r'[-":*()\[\]{}|&^!.]', ' ', query).strip()


def _fts_candidates(query: str) -> list[str]:
    """Return a sequence of progressively relaxed FTS5 AND-mode queries to try.

    FTS5 AND-mode is precise but brittle when the agent uses words not verbatim
    in documents (e.g. 'issue' vs 'degradation'). We build a fallback ladder:
      1. Full sanitized query (highest precision)
      2. Drop pure numeric tokens (date fragments like '02', '20', '2026')
      3. Drop trailing tokens one at a time until only 2 tokens remain

    The search() function tries each in order and stops at the first that returns
    enough results. This keeps AND precision for specific queries while recovering
    gracefully when the agent paraphrases.
    """
    cleaned = re.sub(r'[-":*()\[\]{}|&^!.]', ' ', query)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return [query]

    candidates: list[str] = []

    # 1. Full query
    candidates.append(' '.join(tokens))

    # 2. Drop pure numeric tokens (date components)
    content = [t for t in tokens if not t.isdigit()]
    if content != tokens and len(content) >= 2:
        candidates.append(' '.join(content))
    else:
        content = tokens

    # 3. Progressive truncation: drop last token down to 2-token minimum
    current = list(content)
    while len(current) > 2:
        current = current[:-1]
        candidates.append(' '.join(current))

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


# ── 1. search ─────────────────────────────────────────────────────────────────

@tool
def search(
    query: str,
    customer_name: str = "",
    region: str = "",
    crm_stage: str = "",
    artifact_type: str = "",
    limit: int = 10,
    snippet_chars: int = 400,
    group_by_customer: bool = False,
) -> str:
    """Full-text search over all artifact content (titles, summaries, and body text).

    Returns matching artifacts with a content snippet so you can often answer
    without a follow-up read() call. Use read() only when you need the complete
    document for exact quotes, dates, commands, or action-item lists.

    Set group_by_customer=True for pattern questions ("is this widespread?",
    "which accounts share this problem?"). Results are grouped by customer,
    each entry includes the count of matching artifacts, their titles, the IDs
    you can pass to read(), and a snippet from the top-matching artifact.

    Optional filters narrow results before FTS runs (all are partial-match):
      customer_name — restrict to a specific customer
      region        — restrict to a geographic region, e.g. "Nordics", "Canada"
      crm_stage     — restrict by CRM stage, e.g. "escalation recovery", "renewal"
      artifact_type — one of: customer_call, internal_document,
                      internal_communication, support_ticket, competitor_research

    Examples:
      search("conditional credit NordChemica")
      search("root cause alert noise", group_by_customer=True)
      search("action items owners", customer_name="NordFryst", artifact_type="internal_communication")
      search("approval bypass", region="Canada", group_by_customer=True)
      search("pilot threshold scale", region="Nordics", snippet_chars=600)
    """
    extra_conditions: list[str] = []
    extra_params: list = []

    if customer_name:
        extra_conditions.append("c.name LIKE ?")
        extra_params.append(f"%{customer_name}%")
    if region:
        extra_conditions.append("c.region LIKE ?")
        extra_params.append(f"%{region}%")
    if crm_stage:
        extra_conditions.append("c.crm_stage LIKE ?")
        extra_params.append(f"%{crm_stage}%")
    if artifact_type:
        extra_conditions.append("a.artifact_type = ?")
        extra_params.append(artifact_type)

    snippet_tokens = max(16, int(snippet_chars) // 6)

    def _run_fts(fts_expr: str) -> list:
        conditions = ["artifacts_fts MATCH ?"] + extra_conditions
        params = [fts_expr] + extra_params + [limit]
        return run(
            f"""
            SELECT a.artifact_id, a.artifact_type, a.title, a.summary,
                   snippet(artifacts_fts, 2, '', '', '...', {snippet_tokens}) AS content_snippet,
                   c.name AS customer_name, c.region, c.crm_stage
            FROM artifacts_fts f
            JOIN artifacts a ON a.artifact_id = f.artifact_id
            LEFT JOIN customers c ON c.customer_id = a.customer_id
            WHERE {" AND ".join(conditions)}
            LIMIT ?
            """,
            params=tuple(params),
        )

    # Try progressively relaxed queries until we have enough results.
    rows: list = []
    for fts_expr in _fts_candidates(query):
        rows = _run_fts(fts_expr)
        if len(rows) >= 3:
            break
    if not rows:
        # Last-resort: use whatever the final candidate returned (may be empty)
        rows = _run_fts(_fts_candidates(query)[-1])

    if not group_by_customer:
        return json.dumps([dict(r) for r in rows], indent=2)

    # Group by customer, carry artifact IDs for batch read()
    grouped: dict[str, dict] = defaultdict(lambda: {
        "region": "",
        "crm_stage": "",
        "artifact_count": 0,
        "artifact_ids": [],
        "artifact_titles": [],
        "top_snippet": "",
    })
    for row in rows:
        name = row["customer_name"] or "(no customer)"
        g = grouped[name]
        g["region"] = row["region"] or ""
        g["crm_stage"] = row["crm_stage"] or ""
        g["artifact_count"] += 1
        g["artifact_ids"].append(row["artifact_id"])
        g["artifact_titles"].append(row["title"])
        if not g["top_snippet"] and row["content_snippet"]:
            g["top_snippet"] = row["content_snippet"]

    return json.dumps(
        {
            "customers_matched": len(grouped),
            "by_customer": {k: v for k, v in sorted(grouped.items())},
        },
        indent=2,
    )


# ── 2. read ───────────────────────────────────────────────────────────────────

@tool
def read(artifact_ids: list[str]) -> str:
    """Fetch the full content of one or more artifacts by ID.

    Pass multiple IDs in a single call — this is the efficient path for
    cross-account synthesis (e.g. reading root-cause docs for three accounts
    at once). IDs come from search() results or lookup(entity="artifacts").

    Example:
      read(["art_abc123"])
      read(["art_nordfryst_doc", "art_nordchemica_doc", "art_sentinelops_doc"])
    """
    if not artifact_ids:
        return json.dumps({"error": "No artifact_ids provided."})

    placeholders = ", ".join("?" * len(artifact_ids))
    rows = run(
        f"""
        SELECT a.artifact_id, a.artifact_type, a.title, a.content_text,
               c.name AS customer_name
        FROM artifacts a
        LEFT JOIN customers c ON c.customer_id = a.customer_id
        WHERE a.artifact_id IN ({placeholders})
        """,
        params=tuple(artifact_ids),
    )

    if not rows:
        return json.dumps({"error": f"No artifacts found for ids: {artifact_ids}"})

    return json.dumps([dict(r) for r in rows], indent=2)


# ── 3. lookup ─────────────────────────────────────────────────────────────────

@tool
def lookup(
    entity: str,
    name: str = "",
    region: str = "",
    crm_stage: str = "",
    industry: str = "",
    department: str = "",
    limit: int = 100,
) -> str:
    """Structured row lookup from the customers, competitors, employees, or artifacts tables.

    Use this for structural questions that don't require FTS:
      - Enumerate accounts by region, CRM stage, or industry
      - Get a competitor's full profile (pricing position, strengths, weaknesses)
      - Look up an employee's role or department
      - List all artifact metadata (id, type, title, summary) for a specific customer

    entity values and what 'name' means for each:
      "customers"   — name filters by customer name; also accepts region, crm_stage, industry
      "competitors" — name filters by competitor name (e.g. "NoiseGuard"); returns profile
      "employees"   — name filters by employee name; also accepts department, region
      "artifacts"   — name filters by customer name; returns artifact metadata (no body text —
                      use read() to get content after finding the IDs you want)

    All filters are partial-match (case-insensitive LIKE). Omit filters you don't need.

    Examples:
      lookup("customers", region="Nordics", crm_stage="escalation recovery")
      lookup("competitors", name="NoiseGuard")
      lookup("employees", name="Isabella Rossi")
      lookup("artifacts", name="NordFryst")
    """
    entity = entity.lower().strip()

    if entity == "customers":
        conditions, params = [], []
        if name:
            conditions.append("name LIKE ?")
            params.append(f"%{name}%")
        if region:
            conditions.append("region LIKE ?")
            params.append(f"%{region}%")
        if crm_stage:
            conditions.append("crm_stage LIKE ?")
            params.append(f"%{crm_stage}%")
        if industry:
            conditions.append("industry LIKE ?")
            params.append(f"%{industry}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = run(
            f"""
            SELECT customer_id, name, industry, subindustry, region, country,
                   crm_stage, account_health
            FROM customers
            {where}
            ORDER BY name
            LIMIT ?
            """,
            params=tuple(params),
        )

    elif entity == "competitors":
        conditions, params = [], []
        if name:
            conditions.append("name LIKE ?")
            params.append(f"%{name}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = run(
            f"""
            SELECT competitor_id, name, segment, pricing_position,
                   description, strengths_json, weaknesses_json
            FROM competitors
            {where}
            ORDER BY name
            LIMIT ?
            """,
            params=tuple(params),
        )

    elif entity == "employees":
        conditions, params = [], []
        if name:
            conditions.append("full_name LIKE ?")
            params.append(f"%{name}%")
        if department:
            conditions.append("department LIKE ?")
            params.append(f"%{department}%")
        if region:
            conditions.append("region LIKE ?")
            params.append(f"%{region}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = run(
            f"""
            SELECT employee_id, full_name, title, department, region
            FROM employees
            {where}
            ORDER BY full_name
            LIMIT ?
            """,
            params=tuple(params),
        )

    elif entity == "artifacts":
        conditions, params = [], []
        if name:
            conditions.append("c.name LIKE ?")
            params.append(f"%{name}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)
        rows = run(
            f"""
            SELECT a.artifact_id, a.artifact_type, a.title, a.summary,
                   a.created_at, c.name AS customer_name
            FROM artifacts a
            LEFT JOIN customers c ON c.customer_id = a.customer_id
            {where}
            ORDER BY a.created_at
            LIMIT ?
            """,
            params=tuple(params),
        )

    else:
        return json.dumps({
            "error": f"Unknown entity '{entity}'. Use: customers, competitors, employees, artifacts."
        })

    return json.dumps([dict(r) for r in rows], indent=2)


# ── 4. sql_query ──────────────────────────────────────────────────────────────

@tool
def sql_query(sql: str) -> str:
    """Execute a read-only SQL SELECT against the database.

    Use this when lookup() or search() aren't expressive enough — e.g. you need
    exact column values (not LIKE), multi-table joins, aggregations, or you want
    to filter by a column not exposed by lookup().

    Key tables and columns:
      customers    — customer_id, name, industry, subindustry, region, country,
                     crm_stage, account_health
                     region values: 'ANZ', 'Canada', 'Nordics', 'North America West'
      artifacts    — artifact_id, customer_id, scenario_id, artifact_type, title,
                     summary, content_text, created_at
                     artifact_type values: customer_call, internal_document,
                     internal_communication, support_ticket, competitor_research
      scenarios    — scenario_id, customer_id, industry, title, description, start_date
      implementations — implementation_id, customer_id, scenario_id, status,
                        contract_value, deployment_model
      competitors  — competitor_id, name, segment, pricing_position,
                     description, strengths_json, weaknesses_json
      employees    — employee_id, full_name, title, department, region
      products     — name, category, pricing_model

    Only SELECT statements are allowed.

    Examples:
      sql_query("SELECT name, crm_stage FROM customers WHERE region = 'North America West' ORDER BY name")
      sql_query("SELECT a.title, a.summary FROM artifacts a JOIN customers c ON c.customer_id = a.customer_id WHERE c.name = 'BlueHarbor Logistics'")
      sql_query("SELECT COUNT(*) as n, region FROM customers GROUP BY region")
    """
    if not sql.strip().upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT statements are permitted."})
    rows = run(sql)
    return json.dumps([dict(r) for r in rows], indent=2)
