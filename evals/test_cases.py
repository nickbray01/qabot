"""Ground truth test cases for the search agent eval harness.

Each EvalCase contains:
- question: the natural language query sent to the agent
- expected_answer: the canonical correct answer
- expected_customers: customer names that must appear in tool calls / results
  (used to verify relevant sources were retrieved)
- key_facts: specific phrases the agent answer must convey
- difficulty: "easy" | "hard"
- tags: topic labels for filtering / grouping in reports
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    question: str
    expected_answer: str
    expected_customers: list[str]
    key_facts: list[str]
    difficulty: str
    tags: list[str] = field(default_factory=list)


TEST_CASES: list[EvalCase] = [
    # ── Easy queries ────────────────────────────────────────────────────────────
    EvalCase(
        id="q1_blueharbor_taxonomy",
        question=(
            "which customer's issue started after the 2026-02-20 taxonomy rollout, "
            "and what proof plan did we propose to get them comfortable with renewal?"
        ),
        expected_answer=(
            "That was BlueHarbor Logistics. Northstar proposed a 7-10 business day proof-of-fix: "
            "update index weighting, add a taxonomy mapping layer, and run an A/B test on the top 20 "
            "saved searches, with success defined as top-5 correct hit rate of at least 80 percent on "
            "prioritized queries."
        ),
        expected_customers=["BlueHarbor Logistics"],
        key_facts=[
            "BlueHarbor",
            "proof",
            "A/B test",
            "80",
            "taxonomy",
            "index weighting",
            "taxonomy mapping",
        ],
        difficulty="easy",
        tags=["taxonomy", "renewal", "search"],
    ),
    EvalCase(
        id="q2_verdant_bay_patch",
        question=(
            "for Verdant Bay, what's the approved live patch window, and exactly how do we roll back "
            "if the validation checks fail?"
        ),
        expected_answer=(
            "The approved live patch window is 2026-03-24 from 02:00 to 04:00 local time. "
            "If validation fails, the playbook says to run `orchestrator rollback --target ruleset=<prior_sha>`, "
            "which restores the prior ruleset and replays the invalidation hook."
        ),
        expected_customers=["Verdant Bay"],
        key_facts=[
            "2026-03-24",
            "02:00",
            "04:00",
            "orchestrator rollback",
            "prior_sha",
            "invalidation hook",
        ],
        difficulty="easy",
        tags=["patch", "rollback", "ops"],
    ),
    EvalCase(
        id="q3_mapleharvest_quebec",
        question=(
            "in the MapleHarvest Quebec pilot, what temporary field mappings are we planning in the "
            "router transform, and what is the March 23 workshop supposed to produce?"
        ),
        expected_answer=(
            "The temporary transform maps `txn_id` to `transaction_id` and `total_amount` to "
            "`amount_cents`, coerces string values to integers, and preserves `store_id` and `register_id`. "
            "The 2026-03-23 workshop is supposed to agree the canonical schema, define alias mappings and "
            "producer migration milestones, and produce a signed schema document to upload to `SI-SCHEMA-REG`."
        ),
        expected_customers=["MapleHarvest"],
        key_facts=[
            "txn_id",
            "transaction_id",
            "total_amount",
            "amount_cents",
            "store_id",
            "register_id",
            "SI-SCHEMA-REG",
            "canonical schema",
            "signed schema",
        ],
        difficulty="easy",
        tags=["schema", "mapping", "pilot"],
    ),
    EvalCase(
        id="q4_aureum_scim",
        question=(
            "what SCIM fields were conflicting at Aureum, and what fast fix did Jin propose "
            "so we don't have to wait on Okta change control?"
        ),
        expected_answer=(
            "Aureum was sending both `department` and `businessUnit` variants. Jin's fast fix was a "
            "hot-reloadable Signal Ingest preprocessing rule to normalize those attributes into one canonical "
            "field, plus SCIM tracing so the team can see where approval latency is happening."
        ),
        expected_customers=["Aureum"],
        key_facts=[
            "department",
            "businessUnit",
            "Jin",
            "hot-reloadable",
            "Signal Ingest",
            "SCIM",
            "preprocessing rule",
        ],
        difficulty="easy",
        tags=["scim", "okta", "identity"],
    ),
    # ── Hard queries ────────────────────────────────────────────────────────────
    EvalCase(
        id="q5_blueharbor_defect_risk",
        question=(
            "which customer looks most likely to defect to a cheaper tactical competitor if we miss "
            "the next promised milestone, and what exactly is that milestone?"
        ),
        expected_answer=(
            "BlueHarbor Logistics. It is the clearest cheaper tactical competitor risk because NoiseGuard "
            "is explicitly framed as a low-cost, tactical dedupe layer that can buy time if Northstar misses. "
            "The next promised milestone is the 7-10 business day proof-of-fix for search relevance: BlueHarbor "
            "sends schema export and 14 days of query logs by 2026-03-19, Northstar starts the A/B test on "
            "2026-03-22, and success means top-5 correct hit rate of at least 80 percent for the top 20 saved "
            "searches with no suppression regression."
        ),
        expected_customers=["BlueHarbor Logistics"],
        key_facts=[
            "BlueHarbor",
            "NoiseGuard",
            "2026-03-19",
            "2026-03-22",
            "80",
            "suppression regression",
        ],
        difficulty="hard",
        tags=["competitive", "risk", "milestone"],
    ),
    EvalCase(
        id="q6_na_west_taxonomy_vs_duplicate",
        question=(
            "among the North America West Event Nexus accounts, which ones are really dealing with "
            "taxonomy/search semantics problems versus duplicate-action problems?"
        ),
        expected_answer=(
            "The taxonomy/search semantics group is Arcadia Cloudworks, BlueHarbor Logistics, CedarWind "
            "Renewables, HelioFab Systems, Pacific Health Network, and Pioneer Freight Solutions. Those "
            "accounts all have search relevance degradation after taxonomy changes. The duplicate-action group "
            "is Helix Assemblies Inc., LedgerBright Analytics, LedgerPeak Software, MedLogix Distribution, "
            "Peregrine Logistics Group, and Pioneer Grid Retail LLC. Those accounts are dealing with "
            "post-acquisition deduplication drift, duplicate incident generation, or repeated playbook "
            "executions across bridged systems."
        ),
        expected_customers=[
            "Arcadia Cloudworks",
            "BlueHarbor Logistics",
            "CedarWind Renewables",
            "HelioFab Systems",
            "Pacific Health Network",
            "Pioneer Freight Solutions",
            "Helix Assemblies",
            "LedgerBright Analytics",
            "LedgerPeak Software",
            "MedLogix Distribution",
            "Peregrine Logistics Group",
            "Pioneer Grid Retail",
        ],
        key_facts=[
            "taxonomy",
            "duplicate",
            "Arcadia Cloudworks",
            "BlueHarbor",
            "Helix Assemblies",
            "LedgerBright",
            "search relevance degradation",
        ],
        difficulty="hard",
        tags=["taxonomy", "duplicate", "multi-account", "NA West"],
    ),
    EvalCase(
        id="q7_canada_approval_bypass",
        question=(
            "do we have a recurring Canada approval-bypass pattern across accounts, or is MapleBridge "
            "basically a one-off? Give me the customer names and the shared failure pattern in plain English."
        ),
        expected_answer=(
            "It is definitely a recurring pattern, not a MapleBridge one-off. The clearest accounts are "
            "MapleBridge Insurance, City of Verdant Bay, Maple Regional Transit Authority, MapleBay Marketplace, "
            "MapleFork Franchise Systems, MaplePath Career Institute, and MapleWest Bank. In plain English, "
            "after migration from older workflow systems, Northstar ends up with some mix of bad precedence "
            "metadata, stale caches, field alias mismatches, or delayed schema propagation, so global or "
            "country-default rules win when province, city, or Canada-specific approval rules should win. The "
            "result is approvals getting bypassed, denied, stuck, or routed to the wrong approver, with audit "
            "trails becoming incomplete."
        ),
        expected_customers=[
            "MapleBridge Insurance",
            "Verdant Bay",
            "Maple Regional Transit",
            "MapleBay Marketplace",
            "MapleFork Franchise",
            "MaplePath Career",
            "MapleWest Bank",
        ],
        key_facts=[
            "recurring",
            "MapleBridge",
            "approval",
            "bypass",
            "Canada",
            "precedence",
            "stale cach",
            "audit trail",
        ],
        difficulty="hard",
        tags=["approval", "bypass", "Canada", "pattern"],
    ),
]
