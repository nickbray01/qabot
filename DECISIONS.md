# Design Decisions, Tradeoffs & Open Questions

Running notes from exploration and design sessions. To be compiled into the final design doc.

---

## Architecture Decisions

### Tool-Calling Agent over RAG Pipeline — LangGraph ReAct Loop

**Decision:** Use a LangGraph ReAct agent (`StateGraph` with `call_model` → `call_tools` cycle, gpt-4o) as the primary architecture rather than a pure RAG pipeline.

**Rationale:**
- The data requires multi-hop retrieval: find the right customer, then fetch their artifacts, then synthesize across types.
- A single retrieval step (RAG) cannot handle questions like "get all docs for the customer whose issue started after X date" — it has no way to navigate the relational structure.
- SQL + FTS already handle structured filtering and keyword retrieval exactly. Adding embeddings first would be premature.
- Transparent reasoning: tool calls are observable, debuggable, and auditable in a way that embedding similarity is not.

---

### FTS as Primary Retrieval Layer

**Decision:** Use the existing SQLite FTS5 index (`artifacts_fts`) as the first-pass retrieval mechanism rather than building a vector store from scratch.

**Rationale:**
- The index already exists and covers `title`, `summary`, and `content_text`.
- Artifacts use consistent domain vocabulary (product names, component names like `EN-RULES-ENGINE`, `SI-SCHEMA-REG`, customer names). FTS handles this well.
- BM25 ranking is sufficient for keyword-heavy queries.
- Avoids standing up embedding infrastructure before validating that FTS has meaningful gaps.

---

### Content Passed in Full, Not Chunked (for now)

**Decision:** Pass full `content_text` of retrieved artifacts to the LLM rather than chunking.

**Rationale:**
- Individual artifacts are ~300–1,500 tokens. A typical query touches 3–5 artifacts, staying well within model context limits.
- Chunking risks splitting the exact context needed for synthesis (e.g. the proof plan is spread across a call transcript and an internal doc).
- Revisit if artifact count scales significantly or multi-customer retrieval regularly hits >20 artifacts.


## Tradeoffs

### FTS vs Vector Search

| | FTS5 | Vector / Semantic Search |
|---|---|---|
| Matches | Exact tokens + stems | Meaning, even with no word overlap |
| Speed | Very fast, no model calls | Requires embedding at query time |
| Infrastructure | Already in DB | Needs vector store + embedding model |
| Fails when | User uses different vocabulary than docs | (rarely fails on vocabulary) |
| Best for | Domain-specific queries, product/customer names | Vague natural language, cross-concept queries |

**Conclusion:** FTS covers the majority of queries against this corpus. Vector search adds value at the edges — conceptual/exploratory questions where the user's words don't match document vocabulary.

---

### Flat Artifact List vs Grouped-by-Customer Results

**Problem discovered:** A tool returning a flat list of artifact hits causes agents to anchor on the first result and stop early. The Canada approval-bypass pattern question exposed this: FTS returned 4 affected customers in one query, but a naive agent would only read the top hits (MapleBridge) and conclude it was a one-off.

**Tradeoff:** Grouped tools are more useful for pattern questions but less general. A flat list is a simple, reusable primitive.

**Decision:** Build both — a low-level `search_artifacts(query)` and a higher-level `find_pattern_across_customers(query)` that groups by customer and returns a count before the agent reads individual docs. Both are now implemented in `search_agent/tools.py`. The v1 eval confirms this was the right call: q7 (Canada approval-bypass) succeeded only because the agent used `find_pattern_across_customers`.


---

### Immediate HTTP 200 Before Agent Work (Slackbot)

**Decision:** Return HTTP 200 immediately and hand the event off to an `asyncio.create_task()` background task before running the agent.

**Rationale:**
- Slack requires a response within 3 seconds or it retries the request, causing duplicate responses.
- The agent (LLM calls + SQL/vector retrieval) takes several seconds minimum.
- Separating the ack from the work is the only safe pattern here. `asyncio.create_task()` keeps everything in the same event loop without needing a separate worker queue.

---

### Thread-Based Replies (Slackbot)

**Decision:** All bot replies use `thread_ts` so they appear in threads, not in the channel directly.

**Rationale:**
- Multi-turn conversations in the main channel would flood it.
- Threading keeps each Q&A session self-contained and readable.
- `thread_ts` is also used as the conversation history key, so history is naturally scoped per session.

---

### In-Memory Conversation History (Slackbot)

**Decision:** Store conversation history in a `defaultdict` keyed by `thread_ts`, capped at 20 turns.

**Rationale:**
- Sufficient for development and single-instance deployment.
- Zero infrastructure overhead — no Redis or separate DB needed to get multi-turn working.
- The cap prevents unbounded memory growth on long threads.

**When to revisit:** If the server restarts frequently (history is lost), if multi-instance deployment is needed, or if threads regularly exceed 20 turns.

---

### HMAC-SHA256 Signature Verification with Replay Guard (Slackbot)

**Decision:** Validate every incoming request using Slack's signing secret via `hmac.compare_digest` (timing-safe), and reject requests with a timestamp older than 5 minutes.

**Rationale:**
- Prevents arbitrary POST requests from triggering the agent.
- `hmac.compare_digest` avoids timing side-channel attacks that a naive `==` comparison would expose.
- The 5-minute replay window is Slack's own recommendation.

---

## Open Questions

1. **How vague will production queries be?**
   Determines whether FTS alone is sufficient or whether a hybrid retrieval layer is necessary from the start. If users ask "what went wrong with search quality" rather than "search relevance degradation", FTS will miss.

2. **What's the right chunking strategy if we add embeddings?**
   Artifacts have a `summary` field that may be the right unit to embed rather than the full `content_text`. Embedding the summary keeps chunks semantically coherent and avoids the 1,500-token ceiling per artifact. Needs evaluation.

3. **Should the agent be given a schema description or allowed to discover it?**
   A system prompt describing the tables and their relationships (scenario → customer → implementation → artifacts) would let the agent write SQL. Without it, the agent can only use pre-built tools. Tradeoff: more capable vs. more brittle (agent writes bad SQL).
   **Update (v1 eval):** The system prompt now includes the full schema. The agent can and does write SQL — but v1 shows it reaches for `sql_query` too eagerly on questions where the answer is in artifact content, not structured columns. Schema awareness is necessary but not sufficient; tool-selection guidance needs to be stronger.

4. **How do we handle the premature-stopping failure mode reliably?**
   Prompt engineering ("always search broadly before concluding") helps but is fragile. A planning agent that enumerates sub-questions before retrieving is more robust but adds latency. What's the acceptable latency budget?
   **Update (v1 eval):** Confirmed as the primary failure mode on hard questions. q5 and q6 both terminated after a single failed `sql_query` (1 tool call, score 0.00) rather than falling back to FTS. Next iteration: add explicit system-prompt instructions that `sql_query` is for structured filtering only and that any question involving risk, competitive signals, or cross-account patterns must start with `search_artifacts` or `find_pattern_across_customers`.

5. **Does the Slackbot need to handle follow-up turns / conversation history?**
   If yes, the agent needs memory (at minimum: pass prior tool calls and results back into context). If it's stateless one-shot Q&A, this is much simpler.

---

## Plans for Future Work

### Retrieval Layer
- [x] Add a `find_pattern_across_customers(query)` tool that groups FTS results by customer and returns counts before surfacing individual artifacts.
- [ ] Evaluate hybrid retrieval: run FTS and vector search in parallel, merge ranked lists with Reciprocal Rank Fusion (RRF), pass top-5 to LLM. LangChain `EnsembleRetriever` handles this.
- [ ] If adding embeddings: embed `summary` field (not full `content_text`) as the primary unit; store `artifact_id` as metadata for downstream full-text fetch.

### Agent Design
- [x] Write a system prompt that describes the database schema and instructs the agent to search broadly before concluding on pattern/one-off questions.
- [x] Use LangGraph ReAct loop as the agent architecture (`StateGraph`, gpt-4o).
- [ ] **[Next]** Strengthen system prompt: explicitly reserve `sql_query` for structured filtering only; require `search_artifacts` or `find_pattern_across_customers` as the starting point for any risk, competitive, or cross-account question. This is the highest-leverage fix given v1 eval results.
- [ ] Evaluate a planning agent vs. flat ReAct loop for multi-hop questions. Planning agents explicitly enumerate sub-questions before retrieving, which reduces anchoring on the first failed query.

### Tool Library (`search_agent/`)
- [x] `search_artifacts(query, limit)` — FTS, flat list
- [x] `find_pattern_across_customers(query)` — FTS grouped by customer
- [x] `customer_artifacts(name)` — all artifacts for one customer
- [x] `artifact_full_text(artifact_id)` — fetch single artifact content
- [x] `scenario_summary_tool(scenario_id)` — customer + implementation + artifact list in one shot
- [x] `sql_query(query)` — read-only SQL passthrough for structured questions

### Eval (`evals/`)
- [x] Define acceptance criteria: 7 benchmark questions (4 easy, 3 hard) with ground-truth answers, expected source customers, and key facts.
- [x] Build eval harness: instrument LangGraph graph to capture ordered tool calls (name, args, result); LLM judge (gpt-4o-mini) for semantic scoring (threshold 0.7); source-retrieval check by customer name substring match.
- [x] Establish v1 baseline (see Eval Results below).
- [ ] Add key-facts coverage metric: report which specific facts the agent missed per question, not just pass/fail.
- [ ] Track eval history across runs in a versioned results file to catch regressions.

### Infrastructure (`indexing/`)
- [ ] Script to embed artifact summaries and load into a vector store (Chroma for local dev)
- [ ] Keep artifact_id as the join key so vector results can be hydrated with full text from SQLite

### Slackbot (`slackbot/`)
- [ ] Decide: stateless one-shot Q&A vs. conversational with memory
- [ ] Surface tool call trace in thread replies so users can see how the answer was derived

---

## Overfitting Analysis — 2026-04-04

### Is the system prompt overfitted to the test cases?

**Yes, significantly.** The 10 rules in the system prompt map nearly 1:1 to the 7 benchmark questions:

| Rule | Test case it was written to fix |
|---|---|
| Rule 1: Geographic/segment → `list_customers()` first | q6 (NA West taxonomy vs. duplicate) |
| Rule 2: Competitive risk → competitor lookup first, never `sql_query` alone | q5 (BlueHarbor / NoiseGuard) |
| Rule 3: Pattern → `find_pattern_across_customers()` 3+ times with synonyms | q7 (Canada approval bypass) |
| Rule 7: Always use `find_pattern_across_customers()` for "widespread?" questions | q7 phrasing |
| Rule 9: Preferred tool sequences | General patch for q5/q6/q7 failures |

The rules read like a post-mortem fix list for these 7 questions, not a principled retrieval strategy. A better prompt would teach *why* FTS is needed (artifact content is where the specifics live, not structured columns) rather than prescribing tool sequences per question type.

### Coverage gaps in the current eval suite

**Region coverage:** The entire Nordics region (13 customers: SentinelOps, NordFryst, NordChemica, etc.) and ANZ region (12 customers) are completely unrepresented — 25 of 50 customers with zero test coverage.

**Question-type gaps not currently tested:**
- Commercial/pricing questions: discounts, contract values, concession triggers
- Technical deep dives: specific config changes, component-level remediation steps, on non-test customers
- Competitor-specific lookups: pricing position, strengths/weaknesses (not risk-framing)
- Employee attribution: who owns which action item
- Alert noise pattern questions: NordFryst, NordChemica, SentinelOps all share pilot-threshold-at-scale root cause — untested multi-account pattern
- Cross-region comparisons: e.g., "which accounts across all regions have escalation recovery status?"
- Timeline/milestone questions on Nordics/ANZ customers

---

## Tool Design — Composability vs. Reliability

### The 3-primitive hypothesis (tested in v6)

**Hypothesis:** Seven tools can be collapsed to three composable primitives — `search` (FTS + inline snippets), `read` (batch full-text fetch), `lookup` (structured rows) — reducing agent confusion and improving generalisation.

**Result:** Overall accuracy dropped from 74% (v5, 7 tools) to 47% (v6, 3 tools). Six previously-correct easy/hard cases regressed; three previously-failing cases improved.

**What improved:**
- q16 (action item owners): 0.20 → 1.00. The `artifact_type="internal_communication"` filter directly targeted Slack threads — the right source — which no v5 tool had made easy.
- q2, q4, q11: 0.80 → 1.00. Cleaner `search → read` path with fewer redundant hops.
- q7 (Canada approval bypass): Solved in 1 call vs. 4, with the same score.

**What regressed and why:**

| Case | Root cause |
|---|---|
| q1, q13 (single-customer detail) | Removed `customer_artifacts` — a deterministic "list every artifact for this customer" path. Agent now relies on FTS query quality, which fails when query vocabulary doesn't match. `search("conditional credit", customer_name="NordChemica")` works perfectly; the agent just wasn't using the filter. |
| q3, q8, q12 (detail questions) | 400-char snippet creates false confidence. Agent found the right customer and artifact, read the snippet, concluded it had enough, and answered incorrectly. Key facts (field mappings, success metrics) were past the 400-char cutoff. |
| q6 (NA West classification) | New `lookup` dropped the product filter that old `list_customers` had via a JOIN. Agent also lost the explicit 3-step recipe: enumerate accounts → search pattern A → search pattern B → classify. |
| q5 (competitor identification) | Removed `sql_query` and the explicit "look up competitor names first" rule. Without knowing NoiseGuard = low-mid pricing, the agent can't identify "cheaper tactical competitor." `lookup("competitors")` exists but the agent wasn't told when to use it. |
| q17 (multi-account pattern) | FTS5 treats multi-word queries as AND — a query like `"NordFryst NordChemica SentinelOps"` returns zero rows because no single artifact contains all three names. Agent burned 6 calls on cross-customer queries instead of doing 3 per-customer `search(customer_name=...)` calls then one batch `read`. |

**Confirmed design insights:**

1. **Snippets should be retrieval guides, not answer sources.** The agent treats a 400-char snippet as sufficient for answering, which it almost never is for specific dates, config values, or action-item lists. Default should be 600–800 chars, and the prompt must explicitly say: *snippets are for deciding which artifacts to read, not for answering.*

2. **Deterministic per-customer lookup is load-bearing.** `customer_artifacts(name)` seems redundant (it's just `lookup(entity="artifacts", name=...)`) but it was a reliable fallback when the agent knew the customer. Removing it forced reliance on FTS query quality, which degrades for known-customer questions with unusual vocabulary.

3. **FTS AND semantics are a hard constraint for multi-customer questions.** For cross-account synthesis across named customers, the agent must issue per-customer scoped searches, not a single query with all names. This needs to be an explicit instruction.

4. **Tool capability gaps can't be patched by prompt.** The q6 product-filter gap was a genuine missing SQL JOIN — no amount of prompt guidance fixes that. Tools need to cover the retrieval operations the data actually requires.

5. **Eliminating question-type recipes cost more than expected.** The old 10 rules were overfitted, but they encoded real multi-step patterns (enumerate → classify, competitor lookup → search) that the agent doesn't spontaneously reconstruct from first principles. The right solution is shorter, *principled* recipes — not removing them entirely.

**Decision:** Keep the 3-primitive structure (`search`/`read`/`lookup`) but restore: (a) product filter on customer lookup, (b) explicit `customer_name` filter guidance in the prompt, (c) condensed recipes for the 3 hard question types (classification, competitor, cross-account), (d) bump snippet default to 700 chars.

---

## Eval Results

### v1 Baseline — 2026-04-03

**Setup:** LangGraph ReAct agent, gpt-4o, 7 benchmark questions (4 easy / 3 hard). Judge: gpt-4o-mini, correctness threshold ≥ 0.7.

| Metric | Value |
|---|---|
| Overall accuracy | 5/7 (71%) |
| Easy accuracy | 4/4 (100%) |
| Hard accuracy | 1/3 (33%) |
| Sources found | 4/7 (57%) |
| Avg answer score | 0.63 |
| Avg tool calls | 2.4 |

**Per-question breakdown:**

| ID | Diff | Correct | Score | Tools | Sources |
|---|---|---|---|---|---|
| q1_blueharbor_taxonomy | easy | ✓ | 0.80 | 3 | ✓ |
| q2_verdant_bay_patch | easy | ✓ | 1.00 | 3 | ✓ |
| q3_mapleharvest_quebec | easy | ✓ | 0.80 | 6 | ✓ |
| q4_aureum_scim | easy | ✓ | 1.00 | 2 | ✓ |
| q5_blueharbor_defect_risk | hard | ✗ | 0.00 | 1 | ✗ |
| q6_na_west_taxonomy_vs_duplicate | hard | ✗ | 0.00 | 1 | ✗ |
| q7_canada_approval_bypass | hard | ✓ | 0.80 | 1 | ✗ |

**Root-cause analysis of failures:**

- **q5 (defect risk):** Agent issued a single `sql_query` to find at-risk customers via structured columns (`crm_stage`, `contract_value`). The competitive risk signal (NoiseGuard as a cheap tactical alternative) lives entirely in artifact content. The query returned nothing useful; the agent concluded no at-risk customers exist.

- **q6 (NA West taxonomy vs duplicate):** Agent issued a single `sql_query` filtering by region and product. The `customers` table has a `region` column but no field that distinguishes taxonomy accounts from duplicate-action accounts — that classification only exists in artifact content. The query returned empty; the agent gave up.

- **q7 (Canada approval-bypass):** Agent correctly used `find_pattern_across_customers` and recognized the recurring pattern, but only surfaced 2/7 expected customers (MapleBridge Insurance, MapleFork Franchise). The other five were missed, likely because a single FTS query wasn't broad enough to pull all variant spellings and related artifact types.

**Primary failure mode:** On hard questions requiring competitive/risk reasoning or cross-account synthesis, the agent selects `sql_query` first and stops after one empty result rather than falling back to artifact search. This is a tool-selection / system-prompt problem, not a retrieval gap — the information is in the DB, the agent just isn't choosing the right tool to get it.

---

### v5 — 2026-04-04

**Snapshot:** [evals/results/v5_snapshot.txt](evals/results/v5_snapshot.txt)

**Setup:** LangGraph ReAct agent, gpt-4o, 19 benchmark questions (13 easy / 6 hard). 7 original questions + 12 new ones covering Nordics/ANZ, commercial terms, competitor profiles, employee attribution, cross-region, and multi-account patterns.

| Metric | Value |
|---|---|
| Overall accuracy | 14/19 (74%) |
| Easy accuracy | 11/13 (85%) |
| Hard accuracy | 3/6 (50%) |
| Sources found | 17/19 (89%) |
| Avg answer score | 0.72 |
| Avg tool calls | 3.4 |

**Per-question breakdown:**

| ID | Diff | Correct | Score | Tools | Sources |
|---|---|---|---|---|---|
| q1_blueharbor_taxonomy | easy | ✓ | 0.80 | 2 | ✓ |
| q2_verdant_bay_patch | easy | ✓ | 0.80 | 3 | ✓ |
| q3_mapleharvest_quebec | easy | ✓ | 0.80 | 3 | ✓ |
| q4_aureum_scim | easy | ✓ | 0.80 | 3 | ✓ |
| q5_blueharbor_defect_risk | hard | ✓ | 0.80 | 3 | ✓ |
| q6_na_west_taxonomy_vs_duplicate | hard | ✓ | 0.80 | 3 | ✓ |
| q7_canada_approval_bypass | hard | ✓ | 0.80 | 4 | ✗ |
| q8_nordfryst_renewal_terms | easy | ✓ | 1.00 | 4 | ✓ |
| q9_nordchemica_suppression_bundle | easy | ✓ | 1.00 | 4 | ✓ |
| q10_drs_hysteresis_pilot | easy | ✓ | 1.00 | 2 | ✓ |
| q11_northpoint_provisioning_failures | easy | ✓ | 0.80 | 7 | ✓ |
| q12_laurentia_schema_mitigation | easy | ✓ | 0.80 | 3 | ✗ |
| q13_nordchemica_commercial_concession | easy | ✗ | 0.00 | 3 | ✗ |
| q14_sentinelops_phase0_config | easy | ✓ | 1.00 | 2 | ✓ |
| q15_noiseguard_competitor_profile | easy | ✓ | 1.00 | 2 | ✓ |
| q16_nordfryst_action_item_owners | easy | ✗ | 0.20 | 3 | ✓ |
| q17_alert_noise_pilot_scale_pattern | hard | ✗ | 0.00 | 9 | ✓ |
| q18_cross_region_escalation_recovery | hard | ✗ | 0.60 | 1 | ✓ |
| q19_nordfryst_remediation_timeline | hard | ✗ | 0.60 | 3 | ✓ |

**Remaining failure modes:**

- **q13 (NordChemica commercial terms):** Agent never called `artifact_full_text`. The commercial concession details (6% conditional credit) sit in the internal_document and Slack thread, both of which surfaced in search results — but without reading them in full, the agent answered from summaries and got 0.00.
- **q16 (action item owners):** Agent fetched the wrong artifact (the internal_document instead of the Slack thread). The Slack thread has the precise per-person action items with dates; the internal_document has a less specific version.
- **q17 (multi-account root cause):** 9 tool calls, still no synthesis. Agent couldn't find SentinelOps at all. Multi-customer queries fail in FTS5 AND semantics.
- **q18/q19 (cross-region, timeline):** Sources found but answer incomplete — agent stopped after one tool call without reading full artifact content.

---

### v6 — 2026-04-05 (3-tool simplification experiment)

**Snapshot:** [evals/results/v6_snapshot.txt](evals/results/v6_snapshot.txt)

**Setup:** Same eval suite, but tools collapsed to 3 primitives: `search` (FTS + inline 400-char snippets, optional `group_by_customer`), `read` (batch full-text by ID list), `lookup` (structured rows for customers/competitors/employees/artifacts). System prompt shortened from ~800 to ~250 words.

| Metric | Value | vs. v5 |
|---|---|---|
| Overall accuracy | 9/19 (47%) | −27pp |
| Easy accuracy | 8/13 (62%) | −23pp |
| Hard accuracy | 1/6 (17%) | −33pp |
| Sources found | 12/19 (63%) | −26pp |
| Avg answer score | 0.57 | −0.15 |
| Avg tool calls | 2.6 | −0.8 |

**What improved vs. v5:**

- q16 (action item owners): 0.20 → 1.00. `artifact_type="internal_communication"` filter directly surfaced Slack thread.
- q2, q4, q11: 0.80 → 1.00. Cleaner two-hop path.
- q7 (Canada approval bypass): Same score, half the tool calls.

**What regressed and why:** See *Tool Design — Composability vs. Reliability* section above.

**Verdict:** The 3-primitive architecture is structurally correct but the execution had three gaps: (1) 400-char snippet default was too short and trained the agent to stop early, (2) removing `customer_artifacts` broke deterministic per-customer lookup and forced reliance on FTS vocabulary matching, (3) removing question-type recipes lost multi-step patterns (enumerate→classify, competitor lookup→search, per-customer scoped FTS→batch read) that the agent doesn't reconstruct from first principles.

---

### v7 — 2026-04-05 (three bug fixes + sql_query restored)

**Snapshot:** [evals/results/v7_snapshot.txt](evals/results/v7_snapshot.txt)

**Setup:** Same 4-tool set as v5 (search, read, lookup, sql_query), but tools implemented as the 3-primitive composable structure from v6. Three bugs fixed in `search_agent/tools.py`; system prompt expanded to ~350 words with stronger read() mandates and new recipes for classification and competitor-risk questions.

| Metric | Value | vs. v6 |
|---|---|---|
| Overall accuracy | 16/19 (84%) | +37pp |
| Easy accuracy | 12/13 (92%) | +30pp |
| Hard accuracy | 4/6 (67%) | +50pp |
| Sources found | 16/19 (84%) | +21pp |
| Avg answer score | 0.83 | +0.26 |
| Avg tool calls | 2.2 | −0.4 |

**Per-question breakdown:**

| ID | Diff | Correct | Score | Tools | Sources |
|---|---|---|---|---|---|
| q1_blueharbor_taxonomy | easy | ✓ | 0.80 | 2 | ✓ |
| q2_verdant_bay_patch | easy | ✓ | 0.80 | 2 | ✓ |
| q3_mapleharvest_quebec | easy | ✓ | 0.80 | 2 | ✓ |
| q4_aureum_scim | easy | ✓ | 1.00 | 2 | ✓ |
| q5_blueharbor_defect_risk | hard | ✗ | 0.20 | 2 | ✗ |
| q6_na_west_taxonomy_vs_duplicate | hard | ✓ | 0.80 | 3 | ✓ |
| q7_canada_approval_bypass | hard | ✓ | 0.80 | 2 | ✗ |
| q8_nordfryst_renewal_terms | easy | ✓ | 1.00 | 2 | ✓ |
| q9_nordchemica_suppression_bundle | easy | ✓ | 0.80 | 2 | ✓ |
| q10_drs_hysteresis_pilot | easy | ✓ | 1.00 | 2 | ✓ |
| q11_northpoint_provisioning_failures | easy | ✓ | 1.00 | 2 | ✓ |
| q12_laurentia_schema_mitigation | easy | ✓ | 0.80 | 2 | ✗ |
| q13_nordchemica_commercial_concession | easy | ✗ | 0.60 | 5 | ✓ |
| q14_sentinelops_phase0_config | easy | ✓ | 1.00 | 2 | ✓ |
| q15_noiseguard_competitor_profile | easy | ✓ | 0.80 | 1 | ✓ |
| q16_nordfryst_action_item_owners | easy | ✓ | 1.00 | 2 | ✓ |
| q17_alert_noise_pilot_scale_pattern | hard | ✓ | 1.00 | 4 | ✓ |
| q18_cross_region_escalation_recovery | hard | ✗ | 0.80 | 1 | ✓ |
| q19_nordfryst_remediation_timeline | hard | ✓ | 0.80 | 2 | ✓ |

**Bugs fixed:**

1. **`snippet()` alias error** (`search_agent/tools.py`): `snippet(f, 2, ...)` used a table alias where FTS5 requires the real table name. SQLite raised `OperationalError: no such column: f` on every `search()` call. LangGraph's `ToolNode` caught the exception and returned it as a `ToolMessage`, so the agent continued but had no real search results — it hallucinated. Fix: `snippet(f, ...)` → `snippet(artifacts_fts, ...)`. This single bug caused the bulk of the v6 regression (q1, q3, q5, q6, q13, q17 all affected).

2. **FTS AND-mode vocabulary mismatch** (`search_agent/tools.py`): FTS5 requires ALL query tokens to appear in a document. Agent-generated queries frequently include words absent from documents (e.g. "issue" vs "degradation"), silently excluding correct results. Example: `"taxonomy rollout issue"` matched CedarWind (which uses "issue") but not BlueHarbor (which says "degradation"). Fix: `_fts_candidates()` builds a progressive fallback ladder — full query → drop pure numeric tokens (date fragments) → drop trailing tokens one at a time to a 2-token minimum. `search()` tries each in order, stopping at the first with ≥ 3 results.

3. **Region filter mismatch for NA West** (design gap carried from v6): `lookup(region="NA West")` uses `LIKE '%NA West%'`, which does not match the DB value `'North America West'`, returning 0 rows. Fix: restored `sql_query()` (which allows `WHERE region = 'North America West'` exactly) and documented the exact region enum values in both the tool docstring and system prompt.

**What was restored:**

- `sql_query()` tool (removed in v6, present in v4/v5). Necessary for exact column-value filtering and joins that `lookup()` cannot express. Concrete gap: q6 agent used `lookup(region="NA West")` → 0 rows → LLM had nothing to work with.

**System prompt changes vs. v6:**

- Rule 2 strengthened: explicit mandate to call `read()` whenever the question asks for specific field names, commands, exact numbers, owners, or multi-step plans. Snippets are teasers, not answer sources.
- Rule 4 added: cross-account classification recipe — `search(group_by_customer=True)` per category keyword, then `read()` to confirm.
- Rule 6 added: explicit guidance to use `sql_query()` for exact column values.
- `lookup` NOTE added: region values spelled out verbatim (`'ANZ'`, `'Canada'`, `'Nordics'`, `'North America West'`).

**Remaining failures:**

- **q5 (BlueHarbor defect risk, hard):** Search now surfaces BlueHarbor + NoiseGuard correctly. Agent reads a Pioneer Freight artifact first (also has taxonomy + competitor content) and concludes too early. Needs a competitor-risk recipe in the system prompt (like v5 Rule 2): enumerate competitors → search by competitor name → read customer artifact for risk signal.
- **q18 (escalation recovery cross-region, hard):** `lookup(crm_stage="escalation recovery")` returns all 8 correct accounts. Agent then confuses `account_health` (healthy/at risk/watch list) with `crm_stage` when narrating the results. Fix: clarify in the prompt or tool docstring that `crm_stage` and `account_health` are independent fields.

**Design insight confirmed:** The 3-primitive structure (`search`/`read`/`lookup`) + `sql_query` as a safety valve is the right tool surface. The v6 failures were not a structural problem with composability — they were three implementation bugs plus missing prompt recipes. Composability wins on tool-call efficiency (avg 2.2 calls vs. 3.4 in v5) while matching or exceeding v5 accuracy.

