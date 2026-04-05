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
    # ── Easy queries (Nordics / ANZ) ────────────────────────
    EvalCase(
        id="q8_nordfryst_renewal_terms",
        question=(
            "what commercial concession did Northstar offer NordFryst at renewal, "
            "and what are the success targets for the 90-day pilot?"
        ),
        expected_answer=(
            "Northstar offered NordFryst an 8% renewal discount upfront, with an additional "
            "conditional 4% rebate if success metrics are met within 90 days. The two success "
            "metrics are: 60% reduction in correlated alert volume and MTTA for critical "
            "refrigeration incidents reduced to under 10 minutes."
        ),
        expected_customers=["NordFryst AB"],
        key_facts=[
            "NordFryst",
            "8%",
            "4%",
            "conditional",
            "60%",
            "MTTA",
            "10",
            "90",
        ],
        difficulty="easy",
        tags=["renewal", "commercial", "Nordics", "alert noise"],
    ),
    EvalCase(
        id="q9_nordchemica_suppression_bundle",
        question=(
            "what exactly is in the suppression tuning bundle Northstar is delivering to "
            "NordChemica on 2026-03-22, and what is the rollback mechanism if something goes wrong?"
        ),
        expected_answer=(
            "The bundle includes: EN-DEDUPE settings with a 5-minute sliding window grouped by "
            "service+host+event_type; Rules Evaluation Engine changes that add per-host rate limits "
            "(heartbeat-missing-v1 capped at 1 alert per minute) and convert temp-thresholds-v1 "
            "from instantaneous to aggregated counts over 5 minutes; and SI-ETL-FILTER transforms "
            "to strip or hash session_id fields for PLC sensors where not required. The rollback "
            "mechanism is a feature-flag in the rules engine that reverts to the prior configuration "
            "within 15 minutes."
        ),
        expected_customers=["NordChemica AB"],
        key_facts=[
            "NordChemica",
            "EN-DEDUPE",
            "5-minute sliding window",
            "heartbeat-missing-v1",
            "temp-thresholds-v1",
            "SI-ETL-FILTER",
            "session_id",
            "15 minutes",
            "feature-flag",
        ],
        difficulty="easy",
        tags=["alert noise", "suppression", "Nordics", "ops"],
    ),
    EvalCase(
        id="q10_drs_hysteresis_pilot",
        question=(
            "for the Department of Regional Services, what scoring hysteresis rule did "
            "Isabella propose to fix the confidence threshold flip-flopping, and what is "
            "the pilot's target override rate?"
        ),
        expected_answer=(
            "Isabella Rossi proposed requiring a score above 0.62 for 3 consecutive correlated "
            "events within a 10-minute window before auto-routing — this prevents flip-flopping "
            "around the original threshold of 0.6. The 4-week pilot starts 2026-03-22 across "
            "10 exception classes and targets a manual override rate of no more than 15 percent."
        ),
        expected_customers=["Department of Regional Services (DRS)"],
        key_facts=[
            "0.62",
            "3 consecutive",
            "10-minute",
            "15%",
            "2026-03-22",
            "Isabella",
            "hysteresis",
            "0.6",
        ],
        difficulty="easy",
        tags=["scoring", "pilot", "ANZ", "compliance"],
    ),
    EvalCase(
        id="q11_northpoint_provisioning_failures",
        question=(
            "what triggered the role provisioning degradation at Northpoint Apparel, and "
            "what are the two main failure modes in the failed mapping events?"
        ),
        expected_answer=(
            "The trigger was an HR-led org restructure effective 2026-02-15, after which median "
            "provisioning time jumped from 3.5 hours to roughly 18 hours, impacting access at 120 "
            "stores. The two main failure modes are: (1) the role attribute is missing entirely in "
            "ServiceNow 'store' catalog webhooks, accounting for about 30 percent of failures; and "
            "(2) the role is present under the legacy key `job_code` for the corporate catalog "
            "instead of the expected attribute."
        ),
        expected_customers=["Northpoint Apparel Pty Ltd"],
        key_facts=[
            "Northpoint Apparel",
            "2026-02-15",
            "3.5",
            "18",
            "120 stores",
            "job_code",
            "ServiceNow",
            "store catalog",
        ],
        difficulty="easy",
        tags=["SCIM", "provisioning", "ANZ", "identity"],
    ),
    EvalCase(
        id="q12_laurentia_schema_mitigation",
        question=(
            "for Province of Laurentia, what was the immediate mitigation for the rejected "
            "events from the Laurentia-East regional launch, and what renewal offer is on the table?"
        ),
        expected_answer=(
            "The immediate mitigation is a permissive SI-ETL-FILTER transform that allows unknown "
            "fields through while replicating them to a quarantined topic called `laurentia.unknown` "
            "with 90-day retention, and sets OR-AUDIT-LOGS flags on each bypassed event. The renewal "
            "offer is 160 complimentary engineering hours (approximately 2 engineer-weeks each from "
            "Ingest and Orchestrator) if Laurentia signs a 12-month renewal by 2026-04-15."
        ),
        expected_customers=["Province of Laurentia — Department of Public Works"],
        key_facts=[
            "Laurentia",
            "SI-ETL-FILTER",
            "laurentia.unknown",
            "90",
            "OR-AUDIT-LOGS",
            "160",
            "2026-04-15",
            "12-month",
        ],
        difficulty="easy",
        tags=["schema", "ingest", "Canada", "renewal", "commercial"],
    ),
    # ── Gap coverage ────────────────────────────────────────────────────────────
    EvalCase(
        id="q13_nordchemica_commercial_concession",
        question=(
            "what commercial concession did Northstar offer NordChemica during the "
            "procurement negotiation, and what specific success metric would trigger "
            "the credit?"
        ),
        expected_answer=(
            "Northstar offered NordChemica a conditional 6% credit, approved by Ava Tran. "
            "The credit is tied to SIQ metrics: the trigger is whether NordChemica's "
            "30-day target is met — specifically, actionable alerts must fall to "
            "500 or fewer per day within 30 days of the suppression bundle going live, "
            "with MTTA below 8 minutes and no missed P1 incidents."
        ),
        expected_customers=["NordChemica AB"],
        key_facts=[
            "NordChemica",
            "6%",
            "conditional",
            "500",
            "30",
            "MTTA",
            "8",
        ],
        difficulty="easy",
        tags=["commercial", "pricing", "Nordics", "alert noise"],
    ),
    EvalCase(
        id="q14_sentinelops_phase0_config",
        question=(
            "according to the SentinelOps design review, what are the specific Phase 0 "
            "config-level changes to fix the enrichment bottleneck, and what median "
            "snapshot latency does Phase 0 target?"
        ),
        expected_answer=(
            "Phase 0 has three changes: (1) deploy an SI-ETL-FILTER dedupe rule at ingest "
            "that computes event_hash and drops strict duplicates within a 120-second window; "
            "(2) add write-gating for enrichment results using an idempotency check on "
            "event_hash plus enrichment_version to prevent duplicate derived writes; "
            "(3) tune enrichment parallelism and pod resource requests for immediate throughput "
            "improvement. Phase 0 targets median snapshot latency of 6 to 12 minutes."
        ),
        expected_customers=["SentinelOps AB"],
        key_facts=[
            "SentinelOps",
            "SI-ETL-FILTER",
            "event_hash",
            "120",
            "idempotency",
            "enrichment_version",
            "6",
            "12",
        ],
        difficulty="easy",
        tags=["technical", "config", "Nordics", "ingest"],
    ),
    EvalCase(
        id="q15_noiseguard_competitor_profile",
        question=(
            "what is NoiseGuard's pricing position, and what are its key strengths "
            "and weaknesses as a competitor — not in terms of deal risk, just as a "
            "product profile?"
        ),
        expected_answer=(
            "NoiseGuard sits in the low-to-mid pricing segment. Its strengths are fast, "
            "easy deployment (can be operational in hours for basic suppression rules), "
            "good for tactical noise reduction, and a simple UI with low operational overhead "
            "for basic dedupe use cases. Its weaknesses are that it is not a full observability "
            "stack, offers limited automations and minimal runbook or orchestration features, "
            "is not schema-aware so it cannot reconcile taxonomy mismatches or re-weight search "
            "indexes, and has a smaller integration surface that may require custom plumbing to "
            "feed dedupe decisions back into downstream systems."
        ),
        expected_customers=[],
        key_facts=[
            "NoiseGuard",
            "Low-mid",
            "Easy to deploy",
            "tactical noise reduction",
            "Not a full observability stack",
            "Limited automations",
            "schema",
        ],
        difficulty="easy",
        tags=["competitor", "pricing", "NoiseGuard"],
    ),
    EvalCase(
        id="q16_nordfryst_action_item_owners",
        question=(
            "after the NordFryst renewal negotiation call, who owns the ML trial "
            "provisioning and the commercial amendment drafting action items, and "
            "what are their respective due dates?"
        ),
        expected_answer=(
            "Isabella Rossi owns provisioning the ML Anomaly Scoring trial in Signal Insights, "
            "due 2026-03-25. Liam O'Connor owns drafting the commercial amendment — the 8% "
            "renewal discount plus the conditional 4% rebate language — due 2026-03-24."
        ),
        expected_customers=["NordFryst AB"],
        key_facts=[
            "Isabella Rossi",
            "2026-03-25",
            "ML",
            "Liam O'Connor",
            "2026-03-24",
            "amendment",
        ],
        difficulty="easy",
        tags=["employee attribution", "Nordics", "renewal", "commercial"],
    ),
    EvalCase(
        id="q17_alert_noise_pilot_scale_pattern",
        question=(
            "NordFryst, NordChemica, and SentinelOps all escalated around the same "
            "period with similar operational problems. What is the shared root cause "
            "pattern across these three accounts?"
        ),
        expected_answer=(
            "All three share a pilot-threshold-at-scale pattern: configurations tuned "
            "during small pilots were never revalidated before scaling to production. "
            "NordFryst had refrigeration alert thresholds set for a 3-site pilot that were "
            "too sensitive at 300 sites, with intermittent-connectivity replay bursts "
            "amplifying the storm. NordChemica had thresholds from a 20-host pilot that "
            "exploded to ~12,000 alerts per day at ~1,200 hosts, worsened by high-cardinality "
            "syslog and session_id fields. SentinelOps expanded Signal Ingest from 10 to 28 "
            "collectors on 2026-02-10 without adjusting enrichment capacity, causing CPU "
            "saturation and 20-45 minute executive dashboard delays from duplicate events and "
            "EdgeCollector backfill behaviour."
        ),
        expected_customers=["NordFryst AB", "NordChemica AB", "SentinelOps AB"],
        key_facts=[
            "NordFryst",
            "NordChemica",
            "SentinelOps",
            "pilot",
            "threshold",
            "scale",
            "12,000",
            "28 collector",
        ],
        difficulty="hard",
        tags=["alert noise", "multi-account", "Nordics", "pattern"],
    ),
    EvalCase(
        id="q18_cross_region_escalation_recovery",
        question=(
            "which accounts across all regions currently have escalation recovery as "
            "their CRM stage, and what region is each account in?"
        ),
        expected_answer=(
            "There are eight escalation recovery accounts across two regions. In ANZ: "
            "Aureum Payments Pty Ltd (recovering), Catalyst Careers Pty Ltd (healthy), "
            "HarborHome Marketplace Pty Ltd (expanding), and Harvest Table Group (watch list). "
            "In Canada: Aurora University System (healthy), MapleFork Franchise Systems "
            "(watch list), MapleWest Bank (recovering), and Province of Laurentia — "
            "Department of Public Works (at risk). No escalation recovery accounts appear "
            "in the Nordics or NA West regions."
        ),
        expected_customers=[
            "Aureum Payments",
            "Catalyst Careers",
            "HarborHome Marketplace",
            "Harvest Table Group",
            "Aurora University System",
            "MapleFork Franchise",
            "MapleWest Bank",
            "Province of Laurentia",
        ],
        key_facts=[
            "escalation recovery",
            "ANZ",
            "Canada",
            "Aureum",
            "Catalyst Careers",
            "HarborHome",
            "Harvest Table",
            "Aurora University",
            "MapleFork",
            "MapleWest",
            "Laurentia",
        ],
        difficulty="hard",
        tags=["cross-region", "CRM stage", "escalation recovery"],
    ),
    EvalCase(
        id="q19_nordfryst_remediation_timeline",
        question=(
            "what is the complete post-call action item timeline for NordFryst's alert "
            "remediation and renewal plan — who does what, and by when?"
        ),
        expected_answer=(
            "There are five action items. By 2026-03-20: Daniel Kim deploys the "
            "sliding-window dedupe rule and expands the Temporal Correlator window from "
            "2 minutes to 6 minutes during the production maintenance window "
            "(02:00–04:00 CET), and Jin Park enables the per-site temporary rate cap "
            "of 30 percent on replay traffic. By 2026-03-24: Liam O'Connor delivers the "
            "draft commercial amendment covering the 8% renewal discount and conditional "
            "4% rebate. By 2026-03-25: Isabella Rossi provisions the ML Anomaly Scoring "
            "trial in Signal Insights. By 2026-04-30: Olivia Grant runs the compliance "
            "export walkthrough with NordFryst."
        ),
        expected_customers=["NordFryst AB"],
        key_facts=[
            "Daniel Kim",
            "2026-03-20",
            "6m",
            "Jin Park",
            "rate cap",
            "Liam O'Connor",
            "2026-03-24",
            "Isabella Rossi",
            "2026-03-25",
            "Olivia Grant",
            "2026-04-30",
        ],
        difficulty="hard",
        tags=["timeline", "milestone", "Nordics", "employee attribution"],
    ),
]
