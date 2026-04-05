"""Manual review tool for a single eval case.

Runs the agent on one test case, then prints a side-by-side view of:
  - The question
  - Agent answer vs. expected answer
  - Key-fact coverage (which facts were hit / missed)
  - Source coverage (which expected customers appeared in tool calls)
  - Each tool call in order with its args and full result

Usage:
    python evals/eval_eval.py q1
    python evals/eval_eval.py q8_nordfryst_renewal_terms
    python evals/eval_eval.py q6 --no-tool-results
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.judge import CORRECTNESS_THRESHOLD, check_sources, judge_answer
from evals.runner import run_agent_traced_sync
from evals.test_cases import TEST_CASES

# ── Terminal colours ──────────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _green(s: str)  -> str: return f"{_GREEN}{s}{_RESET}"
def _red(s: str)    -> str: return f"{_RED}{s}{_RESET}"
def _yellow(s: str) -> str: return f"{_YELLOW}{s}{_RESET}"
def _cyan(s: str)   -> str: return f"{_CYAN}{s}{_RESET}"
def _bold(s: str)   -> str: return f"{_BOLD}{s}{_RESET}"
def _dim(s: str)    -> str: return f"{_DIM}{s}{_RESET}"

_W = 80  # wrap width for answers

def _sep(title: str = "") -> None:
    if title:
        pad = (_W - len(title) - 2) // 2
        print(f"\n{'─' * pad} {_bold(title)} {'─' * pad}")
    else:
        print("─" * _W)


def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_W, initial_indent=prefix, subsequent_indent=prefix)


# ── Case lookup ───────────────────────────────────────────────────────────────

def _find_case(id_prefix: str):
    matches = [c for c in TEST_CASES if c.id.startswith(id_prefix)]
    if not matches:
        ids = [c.id for c in TEST_CASES]
        print(_red(f"No test case matching '{id_prefix}'."))
        print("Available IDs:")
        for i in ids:
            print(f"  {i}")
        sys.exit(1)
    if len(matches) > 1:
        print(_yellow(f"Ambiguous prefix '{id_prefix}' matches:"))
        for m in matches:
            print(f"  {m.id}")
        sys.exit(1)
    return matches[0]


# ── Main review ───────────────────────────────────────────────────────────────

def review(id_prefix: str, show_tool_results: bool = True) -> None:
    case = _find_case(id_prefix)

    _sep(f"EVAL REVIEW  ·  {case.id}")
    print(f"  Difficulty : {case.difficulty}   Tags: {', '.join(case.tags)}")
    print(f"\n  {_bold('Question')}")
    print(_wrap(case.question))

    # ── Run agent ─────────────────────────────────────────────────────────────
    print(f"\n  {_dim('Running agent...')}", flush=True)
    import time
    t0 = time.time()
    trace = run_agent_traced_sync(case.question)
    elapsed = time.time() - t0
    print(f"  {_dim(f'Done in {elapsed:.1f}s  ·  {trace.total_tool_calls} tool calls')}")

    # ── Answers ───────────────────────────────────────────────────────────────
    _sep("ANSWERS")
    print(f"  {_bold('Agent')}")
    print(_wrap(trace.answer or "(empty)"))
    print()
    print(f"  {_bold('Expected')}")
    print(_wrap(case.expected_answer))

    # ── Judge scoring ─────────────────────────────────────────────────────────
    _sep("JUDGE")
    verdict = judge_answer(case.question, case.expected_answer, trace.answer)
    score_colour = _green if verdict.correct else (_yellow if verdict.score >= 0.4 else _red)
    print(f"  Score   : {score_colour(f'{verdict.score:.2f}')}  "
          f"({'PASS' if verdict.correct else 'FAIL'}  threshold={CORRECTNESS_THRESHOLD})")
    print(f"  Verdict : {_wrap(verdict.reasoning, indent=0).strip()}")

    # ── Key-fact coverage ─────────────────────────────────────────────────────
    _sep("KEY FACTS")
    answer_lower = (trace.answer or "").lower()
    hits, misses = [], []
    for fact in case.key_facts:
        (hits if fact.lower() in answer_lower else misses).append(fact)

    for f in hits:
        print(f"  {_green('✓')}  {f}")
    for f in misses:
        print(f"  {_red('✗')}  {f}")
    print(f"\n  {len(hits)}/{len(case.key_facts)} facts present in agent answer")

    # ── Source coverage ───────────────────────────────────────────────────────
    _sep("SOURCES")
    sources_found, detail = check_sources(case.expected_customers, trace.all_tool_text())
    for customer, found in detail.items():
        marker = _green("✓") if found else _red("✗")
        print(f"  {marker}  {customer}")
    overall = _green("PASS") if sources_found else _red("FAIL")
    print(f"\n  Source check: {overall}")

    # ── Tool calls ────────────────────────────────────────────────────────────
    _sep("TOOL CALLS")
    if not trace.tool_calls:
        print(_dim("  (no tool calls)"))
    for i, tc in enumerate(trace.tool_calls, 1):
        args_str = ", ".join(f"{k}={v!r}" for k, v in tc.args.items())
        print(f"\n  {_bold(f'{i}. {tc.name}')}({_cyan(args_str)})")
        if show_tool_results:
            # Pretty-print result: indent every line, cap at 40 lines
            lines = tc.result.splitlines()
            cap = 40
            for line in lines[:cap]:
                print(f"     {_dim(line)}")
            if len(lines) > cap:
                print(_dim(f"     ... ({len(lines) - cap} more lines hidden)"))

    if trace.truncated:
        print(_yellow(f"\n  ⚠  Trace truncated at {trace.total_tool_calls} tool calls"))

    _sep()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manually review agent output for a single eval test case."
    )
    parser.add_argument(
        "id",
        help="Test case ID or unique prefix (e.g. 'q8' or 'q8_nordfryst_renewal_terms')",
    )
    parser.add_argument(
        "--no-tool-results",
        action="store_true",
        help="Hide tool call results (show tool names + args only)",
    )
    args = parser.parse_args()
    review(args.id, show_tool_results=not args.no_tool_results)
