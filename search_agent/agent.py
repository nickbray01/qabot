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

from .tools import (
    artifact_full_text,
    customer_artifacts,
    find_pattern_across_customers,
    scenario_summary_tool,
    search_artifacts,
    sql_query,
)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a QA analyst assistant for a B2B software company.
You have access to a SQLite database of customer scenarios, implementations, and artifacts
(meeting notes, support tickets, internal documents, PRDs, call transcripts, etc.).

Database schema:
- scenarios: scenario_id, customer_id, industry, title, description, start_date
- customers: customer_id, scenario_id, name, industry, region, crm_stage
- implementations: implementation_id, customer_id, scenario_id, status, contract_value, deployment_model
- artifacts: artifact_id, customer_id, scenario_id, artifact_type, title, summary, content_text, created_at
- products: name, category, pricing_model
- company_profile: name, category, headquarters, founding_year
- competitors: name, segment, pricing_position
- employees: department (and others)

FTS index: artifacts_fts covers title, summary, content_text on the artifacts table.
Domain vocabulary includes product/component codes like EN-RULES-ENGINE, SI-SCHEMA-REG, etc.

Search strategy:
1. Start with search_artifacts() or find_pattern_across_customers() for most questions.
2. For "is this widespread or a one-off?" questions, always use find_pattern_across_customers()
   to count affected customers BEFORE drawing conclusions.
3. Use artifact_full_text() only for artifacts directly relevant to the answer —
   do not fetch every result, only the most relevant ones.
4. Use sql_query() for structured filtering (dates, counts, contract values, joins).
5. Never conclude after reading only one result — broaden the search first.
6. Be concise but complete. Quote or paraphrase artifact content when it supports the answer.
"""

# ── Tools and model ───────────────────────────────────────────────────────────

_tools = [
    search_artifacts,
    find_pattern_across_customers,
    customer_artifacts,
    scenario_summary_tool,
    artifact_full_text,
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
