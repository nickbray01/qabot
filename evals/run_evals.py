"""Eval harness entry point.

Run from the project root (either style works):

    python evals/run_evals.py                          # all test cases
    python evals/run_evals.py --id q1 q2               # specific case IDs
    python evals/run_evals.py --difficulty easy         # filter by difficulty
    python evals/run_evals.py --output report.json      # also save JSON results

    python -m evals.run_evals                           # module style also works

Produces a human-readable console report and, optionally, a JSON file with
full traces and judge verdicts for downstream analysis.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

# Ensure the project root is on sys.path so this script works when run
# directly (`python evals/run_evals.py`) as well as via `-m evals.run_evals`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from evals.judge import JudgeVerdict, check_sources, judge_answer
from evals.runner import MAX_TOOL_CALLS, AgentTrace, run_agent_traced
from evals.test_cases import TEST_CASES, EvalCase


# ── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    question: str
    difficulty: str
    tags: list[str]
    # Agent output
    agent_answer: str
    total_tool_calls: int
    tool_call_names: list[str]          # ordered list of tool names invoked
    # Correctness
    answer_score: float
    answer_correct: bool
    judge_reasoning: str
    # Source retrieval
    sources_found: bool
    sources_detail: dict[str, bool]     # customer → was it found?
    # Timing
    elapsed_seconds: float
    error: str = ""                      # non-empty if the agent raised


# ── Core evaluation logic ───────────────────────────────────────────────────

async def evaluate_case(case: EvalCase) -> EvalResult:
    """Run one eval case end-to-end and return a structured result."""
    start = time.monotonic()
    error = ""
    trace = AgentTrace(answer="", tool_calls=[])

    try:
        trace = await run_agent_traced(case.question)
        if trace.truncated:
            error = f"Tool call limit ({MAX_TOOL_CALLS}) exceeded — agent did not finish"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - start

    # Judge answer quality
    verdict: JudgeVerdict = judge_answer(
        question=case.question,
        expected=case.expected_answer,
        actual=trace.answer,
    )

    # Check source retrieval
    sources_found, sources_detail = check_sources(
        expected_customers=case.expected_customers,
        all_tool_text=trace.all_tool_text(),
    )

    return EvalResult(
        case_id=case.id,
        question=case.question,
        difficulty=case.difficulty,
        tags=case.tags,
        agent_answer=trace.answer,
        total_tool_calls=trace.total_tool_calls,
        tool_call_names=[tc.name for tc in trace.tool_calls],
        answer_score=verdict.score,
        answer_correct=verdict.correct,
        judge_reasoning=verdict.reasoning,
        sources_found=sources_found,
        sources_detail=sources_detail,
        elapsed_seconds=round(elapsed, 1),
        error=error,
    )


async def run_all(cases: list[EvalCase]) -> list[EvalResult]:
    """Evaluate all cases sequentially (parallel would hit rate limits)."""
    results = []
    total = len(cases)
    for i, case in enumerate(cases, 1):
        print(f"  [{i}/{total}] Running: {case.id} ({case.difficulty}) ...", flush=True)
        result = await evaluate_case(case)
        status = "✓" if result.answer_correct else "✗"
        src = "✓" if result.sources_found else "✗"
        print(
            f"         answer={status}  score={result.answer_score:.2f}  "
            f"tools={result.total_tool_calls}  sources={src}  "
            f"({result.elapsed_seconds}s)",
            flush=True,
        )
        if result.error:
            print(f"         ERROR: {result.error}", flush=True)
        results.append(result)
    return results


# ── Report formatting ────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> str:
    """Wrap *text* to *width*, returning the first line only with ellipsis."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def print_report(results: list[EvalResult]) -> None:
    """Print a human-readable eval report to stdout."""
    if not results:
        print("No results to report.")
        return

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Summary stats ──
    n = len(results)
    n_correct = sum(r.answer_correct for r in results)
    n_sources = sum(r.sources_found for r in results)
    avg_tools = sum(r.total_tool_calls for r in results) / n if n else 0
    avg_score = sum(r.answer_score for r in results) / n if n else 0

    easy = [r for r in results if r.difficulty == "easy"]
    hard = [r for r in results if r.difficulty == "hard"]

    print()
    print("=" * 80)
    print("  EVAL REPORT  —", now)
    print("=" * 80)
    print()
    print(f"  Total questions : {n}")
    print(f"  Correct answers : {n_correct}/{n} ({100*n_correct/n:.0f}%)")
    print(f"  Sources found   : {n_sources}/{n} ({100*n_sources/n:.0f}%)")
    print(f"  Avg answer score: {avg_score:.2f}")
    print(f"  Avg tool calls  : {avg_tools:.1f}")
    if easy:
        ec = sum(r.answer_correct for r in easy)
        print(f"  Easy accuracy   : {ec}/{len(easy)}")
    if hard:
        hc = sum(r.answer_correct for r in hard)
        print(f"  Hard accuracy   : {hc}/{len(hard)}")
    print()

    # ── Per-question table ──
    COL_ID   = 32
    COL_DIFF =  5
    COL_ANS  =  7
    COL_SCO  =  6
    COL_TOO  =  6
    COL_SRC  =  8

    header = (
        f"{'ID':<{COL_ID}} {'Diff':<{COL_DIFF}} {'Correct':<{COL_ANS}} "
        f"{'Score':<{COL_SCO}} {'Tools':<{COL_TOO}} {'Sources':<{COL_SRC}}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        ans_mark = "✓" if r.answer_correct else "✗"
        src_mark = "✓" if r.sources_found else "✗"
        err_mark = " [ERR]" if r.error else ""
        print(
            f"{_wrap(r.case_id, COL_ID):<{COL_ID}} "
            f"{r.difficulty:<{COL_DIFF}} "
            f"{ans_mark:<{COL_ANS}} "
            f"{r.answer_score:<{COL_SCO}.2f} "
            f"{r.total_tool_calls:<{COL_TOO}} "
            f"{src_mark:<{COL_SRC}}"
            f"{err_mark}"
        )

    print()

    # ── Per-question detail ──
    print("=" * 80)
    print("  DETAIL")
    print("=" * 80)
    for r in results:
        print()
        print(f"  [{r.case_id}]  ({r.difficulty})  score={r.answer_score:.2f}  tools={r.total_tool_calls}  {r.elapsed_seconds}s")
        print(f"  Q: {_wrap(r.question, 74)}")
        print(f"  Tools used: {' → '.join(r.tool_call_names) or '(none)'}")

        # Sources detail
        if r.sources_detail:
            found_cust = [c for c, v in r.sources_detail.items() if v]
            missing_cust = [c for c, v in r.sources_detail.items() if not v]
            if found_cust:
                print(f"  Sources ✓: {', '.join(found_cust)}")
            if missing_cust:
                print(f"  Sources ✗: {', '.join(missing_cust)}")

        # Judge reasoning
        print(f"  Judge: {textwrap.fill(r.judge_reasoning, width=74, subsequent_indent='         ')}")

        if r.error:
            print(f"  ERROR: {r.error}")

    print()
    print("=" * 80)


# ── JSON serialisation ───────────────────────────────────────────────────────

def results_to_json(results: list[EvalResult]) -> dict:
    """Convert results to a JSON-serialisable dict."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "correct": sum(r.answer_correct for r in results),
            "sources_found": sum(r.sources_found for r in results),
            "avg_score": round(sum(r.answer_score for r in results) / len(results), 3) if results else 0,
            "avg_tool_calls": round(sum(r.total_tool_calls for r in results) / len(results), 1) if results else 0,
        },
        "results": [asdict(r) for r in results],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the search agent eval harness.")
    p.add_argument(
        "--id",
        nargs="+",
        metavar="ID",
        help="Run only these case IDs (e.g. q1_blueharbor_taxonomy q2_verdant_bay_patch).",
    )
    p.add_argument(
        "--difficulty",
        choices=["easy", "hard"],
        help="Filter to only 'easy' or 'hard' cases.",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Save full JSON results to this file (e.g. report.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Filter cases
    cases = list(TEST_CASES)
    if args.id:
        cases = [c for c in cases if c.id in args.id]
        if not cases:
            sys.exit(f"No cases matched IDs: {args.id}")
    if args.difficulty:
        cases = [c for c in cases if c.difficulty == args.difficulty]
        if not cases:
            sys.exit(f"No cases with difficulty={args.difficulty!r}")

    print(f"\nRunning {len(cases)} eval case(s) …\n")
    results = asyncio.run(run_all(cases))

    print_report(results)

    if args.output:
        data = results_to_json(results)
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"JSON results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
