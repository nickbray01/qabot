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


