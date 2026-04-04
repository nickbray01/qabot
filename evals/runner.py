"""Agent runner with full tool-call instrumentation.

Runs the LangGraph agent and captures:
- The final answer
- Every tool call (name + args + result), in order
- Total tool-call count

This bypasses the high-level `run_agent` helper so we can observe both the
tool inputs (from AIMessage.tool_calls) and tool outputs (from ToolMessages).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from search_agent.agent import graph


@dataclass
class ToolCall:
    name: str
    args: dict
    result: str  # raw string content returned by the tool


MAX_TOOL_CALLS = 10


@dataclass
class AgentTrace:
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    truncated: bool = False  # True when MAX_TOOL_CALLS was hit

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    def all_tool_text(self) -> str:
        """Concatenate all tool call names, args, and results into one blob for source-checking."""
        parts = []
        for tc in self.tool_calls:
            parts.append(tc.name)
            parts.append(str(tc.args))
            parts.append(tc.result)
        return " ".join(parts)


async def run_agent_traced(question: str) -> AgentTrace:
    """Run the agent on *question* and return a fully instrumented AgentTrace.

    The question is sent as a fresh single-turn conversation (no history).
    """
    messages = [HumanMessage(content=question)]

    # tool_call_id → {name, args} for pending (not-yet-answered) tool calls
    pending: dict[str, dict] = {}
    completed: list[ToolCall] = []
    final_answer = ""

    truncated = False
    async for chunk in graph.astream({"messages": messages}):
        # ── Model turn: may contain tool invocations or a final answer ──────
        if "call_model" in chunk:
            ai_msg = chunk["call_model"]["messages"][-1]
            for tc in ai_msg.tool_calls:
                pending[tc["id"]] = {"name": tc["name"], "args": tc["args"]}
            if not ai_msg.tool_calls and ai_msg.content:
                final_answer = ai_msg.content

        # ── Tool turn: each ToolMessage matches a pending tool call by id ───
        if "call_tools" in chunk:
            for tool_msg in chunk["call_tools"]["messages"]:
                tcid = tool_msg.tool_call_id
                info = pending.pop(tcid, None)
                if info is not None:
                    completed.append(
                        ToolCall(
                            name=info["name"],
                            args=info["args"],
                            result=tool_msg.content if isinstance(tool_msg.content, str)
                                   else str(tool_msg.content),
                        )
                    )
            if len(completed) >= MAX_TOOL_CALLS:
                truncated = True
                break

    return AgentTrace(answer=final_answer, tool_calls=completed, truncated=truncated)


def run_agent_traced_sync(question: str) -> AgentTrace:
    """Synchronous wrapper around run_agent_traced."""
    return asyncio.run(run_agent_traced(question))
