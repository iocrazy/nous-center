"""Simple keyword-based scorer for compact eval.

LLM-as-judge version is future work; keyword matching is enough for initial baseline.
"""

from __future__ import annotations


def score_response(*, response: str, must_contain: list[str]) -> dict:
    """Score 0-10 based on fraction of must_contain terms present in response."""
    if not must_contain:
        return {"score": 10, "matched": [], "missing": []}
    matched = [term for term in must_contain if term in response]
    missing = [term for term in must_contain if term not in response]
    ratio = len(matched) / len(must_contain)
    return {
        "score": round(ratio * 10),
        "matched": matched,
        "missing": missing,
    }
