"""LLM-as-judge for semantic answer equivalence.

Uses gpt-4o-mini (fast + cheap) to decide whether the agent's answer captures
the same meaning as the expected ground-truth answer.

Returns a structured JudgeVerdict with:
- score:      float 0.0 – 1.0  (how complete / accurate the answer is)
- correct:    bool              (True if score >= CORRECTNESS_THRESHOLD)
- reasoning:  str               (brief explanation from the judge)
"""
from __future__ import annotations

import json
import re

from langchain_openai import ChatOpenAI

CORRECTNESS_THRESHOLD = 0.7  # score at or above this is treated as "correct"

_JUDGE_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """\
You are a strict but fair grader evaluating whether an AI assistant's answer to a business question
contains all the key information from a reference (ground-truth) answer.

Scoring rubric:
  1.0  – Answer captures all key facts from the reference; may use different wording.
  0.8  – Answer captures most key facts; one or two minor omissions.
  0.6  – Answer captures roughly half the key facts; clearly incomplete.
  0.4  – Answer has the right general topic but is missing most specifics.
  0.2  – Answer is mostly wrong or irrelevant.
  0.0  – Answer is completely wrong, refuses to answer, or is empty.

Rules:
- Slight paraphrasing is fine; exact wording is not required.
- Extra information beyond the reference is not penalised.
- If the agent answer is empty or says "I don't know", score is 0.0.
- Respond ONLY with a JSON object: {"score": <float>, "correct": <bool>, "reasoning": "<1-2 sentences>"}
"""

_USER_TEMPLATE = """\
Question: {question}

Reference answer:
{expected}

Agent answer:
{actual}
"""

_llm = ChatOpenAI(model=_JUDGE_MODEL, temperature=0)


class JudgeVerdict:
    __slots__ = ("score", "correct", "reasoning")

    def __init__(self, score: float, correct: bool, reasoning: str) -> None:
        self.score = score
        self.correct = correct
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return f"JudgeVerdict(score={self.score:.2f}, correct={self.correct}, reasoning={self.reasoning!r})"


def judge_answer(question: str, expected: str, actual: str) -> JudgeVerdict:
    """Call the LLM judge and return a JudgeVerdict."""
    user_msg = _USER_TEMPLATE.format(
        question=question,
        expected=expected,
        actual=actual if actual.strip() else "(empty)",
    )
    response = _llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        correct = bool(data.get("correct", score >= CORRECTNESS_THRESHOLD))
        reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: treat parse failure as a zero score
        score = 0.0
        correct = False
        reasoning = f"Judge response could not be parsed: {raw[:200]}"

    # Clamp score
    score = max(0.0, min(1.0, score))
    return JudgeVerdict(score=score, correct=correct, reasoning=reasoning)


def check_sources(
    expected_customers: list[str],
    all_tool_text: str,
) -> tuple[bool, dict[str, bool]]:
    """Check whether each expected customer appears in the agent's tool call text.

    Args:
        expected_customers: List of customer name substrings to look for.
        all_tool_text: Concatenated tool names + args + results from the trace.

    Returns:
        (sources_found, detail_dict) where sources_found is True when the majority
        of expected customers were surfaced (>= 50% for multi-customer cases,
        100% for single-customer cases).
    """
    lower_text = all_tool_text.lower()
    detail: dict[str, bool] = {}
    for customer in expected_customers:
        detail[customer] = customer.lower() in lower_text

    found_count = sum(detail.values())
    total = len(expected_customers)

    if total == 0:
        return True, detail

    threshold = 1.0 if total == 1 else 0.5
    sources_found = (found_count / total) >= threshold
    return sources_found, detail
