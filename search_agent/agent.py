"""LangGraph ReAct agent for the synthetic startup database."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from .tools import lookup, read, search, sql_query

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an analyst assistant for a B2B software company with access to a
database of customer scenarios, implementations, and artifacts (call transcripts, meeting
notes, support tickets, internal documents, Slack threads, competitor research).

You have four tools:

search(query, ...)
  Full-text search over all artifact content. Returns matching artifacts with a content
  snippet. Use this as your first move for almost every content question. Filters:
  customer_name, region, crm_stage, artifact_type. Set group_by_customer=True for
  pattern or classification questions.

read(artifact_ids)
  Returns FULL content for one or more artifacts by ID. Pass multiple IDs in a single call.
  ALWAYS call read() when the question asks for specific technical values: field names,
  commands, exact percentages, exact dates, action-item owners, step-by-step plans.
  Snippets are teasers — never try to answer a detail question from a snippet alone.

lookup(entity, ...)
  Structured row lookup — no FTS. Use for enumerating customers by region/crm_stage,
  getting competitor profiles, looking up employees, or listing artifact metadata.
  entity values: "customers", "competitors", "employees", "artifacts"
  NOTE: region values in the DB are exactly: 'ANZ', 'Canada', 'Nordics', 'North America West'

sql_query(sql)
  Execute a read-only SELECT. Use when lookup() filters aren't enough — e.g. exact region
  match, multi-table joins, or aggregations.
  Tables: customers (customer_id, name, industry, region, crm_stage, account_health),
          artifacts (artifact_id, customer_id, artifact_type, title, summary, content_text),
          competitors, employees, implementations, scenarios, products.

Decision rules:
1. Start with search() for content questions; start with lookup() or sql_query() for
   enumeration (customer lists, competitor profiles, employee records).
2. After search(), if the question asks for specific field names, commands, exact numbers,
   owners, or multi-step plans: call read() on the top relevant artifact IDs. Do not
   try to answer from snippets when specific values are required.
3. For "is this widespread?" or "which accounts share X?", use search(group_by_customer=True)
   then read() the top artifact IDs to verify and enrich each account's situation.
4. For cross-account classification (e.g. "which are taxonomy problems vs duplicate problems"),
   use search(group_by_customer=True) with targeted keywords for each category, then
   read() representative artifacts to confirm the classification.
5. For exact action items, owners, or dates: use artifact_type="internal_communication"
   to target Slack threads, which contain the most precise action-item lists.
6. Use sql_query() when you need exact column values (e.g. region = 'North America West'),
   aggregations, or joins not covered by lookup().
7. If search returns no results, rephrase with different/simpler keywords before concluding
   the data doesn't exist — empty results mean vocabulary mismatch.
"""

# ── Tools and model ───────────────────────────────────────────────────────────

_tools = [
    search,
    read,
    lookup,
    sql_query,
]

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_llm_with_tools = _llm.bind_tools(_tools)


# ── Graph nodes ───────────────────────────────────────────────────────────────

def call_model(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ── Build graph ───────────────────────────────────────────────────────────────

_tool_node = ToolNode(_tools)

_builder = StateGraph(MessagesState)
_builder.add_node("call_model", call_model)
_builder.add_node("call_tools", _tool_node)

_builder.add_edge(START, "call_model")
_builder.add_conditional_edges("call_model", tools_condition, {"tools": "call_tools", END: END})
_builder.add_edge("call_tools", "call_model")

graph = _builder.compile()


# ── Public interface ──────────────────────────────────────────────────────────

async def run_agent(
    user_text: str,
    history: list,
    thread_ts: str,
    on_tool_call: Callable[[str, dict], Awaitable[None]] | None = None,
) -> str:
    """Run the agent and return its final text response.

    Args:
        user_text: The current user message (already stripped of @mention).
        history: Full conversation history including the current message,
                 as a list of {"role": "user"|"assistant", "content": str} dicts.
        thread_ts: Slack thread timestamp (unused by agent; available for future tracing).
        on_tool_call: Optional async callback fired just before each tool executes.
                      Receives (tool_name, args) so the caller can post a status update.
    """
    messages = []
    for msg in history[:-1]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_text))

    last_ai_message: AIMessage | None = None
    async for chunk in graph.astream({"messages": messages}):
        if "call_model" in chunk:
            ai_msg = chunk["call_model"]["messages"][-1]
            last_ai_message = ai_msg
            if on_tool_call and ai_msg.tool_calls:
                for tc in ai_msg.tool_calls:
                    await on_tool_call(tc["name"], tc["args"])

    return last_ai_message.content  # type: ignore[union-attr]
