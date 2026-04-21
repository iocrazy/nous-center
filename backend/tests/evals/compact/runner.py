"""Run compact eval against all fixtures, generate report.

Usage:
    cd backend && CUDA_VISIBLE_DEVICES="" python -m tests.evals.compact.runner

Output: backend/tests/evals/compact/latest_report.json
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.services.context.gzip_compact import GzipCompactContextEngine
from src.services.context.base import ContextOverflowError

from tests.evals.compact.scorer import score_response

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures.jsonl"
BASELINES = HERE / "baselines"
REPORT = HERE / "latest_report.json"

MAX_TOKENS_BUDGET = 1000


async def _simulate_response(compacted_messages: list[dict], test_prompt: str) -> str:
    """In production, call real LLM here. For initial baseline, synthesize a response
    by concatenating the content of compacted messages — this tests whether compress
    preserved the right info."""
    all_text = " ".join(m.get("content", "") for m in compacted_messages if isinstance(m.get("content"), str))
    return f"{all_text} [prompt: {test_prompt}]"


async def main():
    engine = GzipCompactContextEngine()
    await engine.initialize()

    results = []
    with FIXTURES.open() as f:
        for line in f:
            fix = json.loads(line)
            try:
                compacted, truncated = await engine.compress(
                    messages=fix["conversation"],
                    max_tokens=MAX_TOKENS_BUDGET,
                )
                response = await _simulate_response(compacted, fix["test_prompt"])
                score = score_response(
                    response=response,
                    must_contain=fix["must_contain"],
                )
            except ContextOverflowError as e:
                score = {"score": 0, "matched": [], "missing": fix["must_contain"], "error": str(e)}
                truncated = True
            results.append({
                "id": fix["id"],
                "score": score["score"],
                "truncated": truncated,
                "missing": score.get("missing", []),
            })

    avg = sum(r["score"] for r in results) / len(results)
    report = {
        "engine": engine.name,
        "fixtures_count": len(results),
        "avg_score": round(avg, 2),
        "results": results,
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Avg score: {avg:.2f}/10 (fixtures: {len(results)})")
    print(f"Report: {REPORT}")

    # Baseline comparison
    latest_baseline = BASELINES / "gzip_compact_v1.json"
    if latest_baseline.exists():
        baseline = json.loads(latest_baseline.read_text())
        delta = avg - baseline["avg_score"]
        if delta < -2:
            print(f"⚠️  REGRESSION: avg_score dropped {delta:+.2f} from baseline")
            return 1
        print(f"Baseline delta: {delta:+.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
