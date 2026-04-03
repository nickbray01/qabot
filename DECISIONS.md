# Design Decisions, Tradeoffs & Open Questions

Running notes from exploration and design sessions. To be compiled into the final design doc.

---

## Architecture Decisions

### Tool-Calling Agent over RAG Pipeline

**Decision:** Use a tool-calling agent (LangChain `create_tool_calling_agent` / ReAct loop) as the primary architecture rather than a pure RAG pipeline.

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

**Decision pending:** Build both — a low-level `search_artifacts(query)` and a higher-level `find_pattern_across_customers(query)` that groups by customer and returns a count before the agent reads individual docs.


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

4. **How do we handle the premature-stopping failure mode reliably?**
   Prompt engineering ("always search broadly before concluding") helps but is fragile. A planning agent that enumerates sub-questions before retrieving is more robust but adds latency. What's the acceptable latency budget?

5. **Does the Slackbot need to handle follow-up turns / conversation history?**
   If yes, the agent needs memory (at minimum: pass prior tool calls and results back into context). If it's stateless one-shot Q&A, this is much simpler.

---

## Plans for Future Work

### Retrieval Layer
- [ ] Add a `find_pattern_across_customers(query)` tool that groups FTS results by customer and returns counts before surfacing individual artifacts.
- [ ] Evaluate hybrid retrieval: run FTS and vector search in parallel, merge ranked lists with Reciprocal Rank Fusion (RRF), pass top-5 to LLM. LangChain `EnsembleRetriever` handles this.
- [ ] If adding embeddings: embed `summary` field (not full `content_text`) as the primary unit; store `artifact_id` as metadata for downstream full-text fetch.

### Agent Design
- [ ] Write a system prompt that describes the database schema and instructs the agent to search broadly before concluding on pattern/one-off questions.
- [ ] Evaluate a planning agent (e.g. LangGraph) vs. a flat ReAct loop for multi-hop questions. Planning agents explicitly enumerate sub-questions before retrieving, which reduces anchoring.
- [ ] Define acceptance criteria for agent eval: what question types must it handle correctly?

### Tool Library (`search_agent/`)
- [ ] `search_artifacts(query, limit)` — FTS, flat list (already in `database_utils`)
- [ ] `find_pattern_across_customers(query)` — FTS grouped by customer
- [ ] `customer_artifacts(name)` — all artifacts for one customer (already in `database_utils`)
- [ ] `artifact_full_text(artifact_id)` — fetch single artifact content
- [ ] `scenario_summary(scenario_id)` — customer + implementation + artifact list in one shot (already in `database_utils`)
- [ ] `sql_query(query)` — read-only SQL passthrough for structured questions

### Infrastructure (`indexing/`)
- [ ] Script to embed artifact summaries and load into a vector store (Chroma for local dev)
- [ ] Keep artifact_id as the join key so vector results can be hydrated with full text from SQLite

### Slackbot (`slackbot/`)
- [ ] Decide: stateless one-shot Q&A vs. conversational with memory
- [ ] Surface tool call trace in thread replies so users can see how the answer was derived


