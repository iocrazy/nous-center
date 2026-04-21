# Compact Eval Harness

Regression guard for `GzipCompactContextEngine` (or any `ContextEngine` impl).

## Run

```bash
cd backend && CUDA_VISIBLE_DEVICES="" python -m tests.evals.compact.runner
```

（`CUDA_VISIBLE_DEVICES=""` 前缀避免触发 CUDA 初始化，保护宿主 X session。）

Outputs `latest_report.json`. Compares against `baselines/gzip_compact_v1.json`:
- Δavg_score > -2 → OK
- Δavg_score ≤ -2 → ⚠️ REGRESSION (fail the ship decision)

## Update baseline

When compress strategy changes intentionally:

```bash
cp tests/evals/compact/latest_report.json tests/evals/compact/baselines/gzip_compact_v2.json
# 更新 runner.py 里 latest_baseline 指向
```

## Initial baseline

The file `baselines/gzip_compact_v1.json` was committed as a **placeholder** (avg_score=0, fixtures_count=0).
First-time users MUST run the runner locally to generate real baseline values:

```bash
cd backend && CUDA_VISIBLE_DEVICES="" python -m tests.evals.compact.runner
cp tests/evals/compact/latest_report.json tests/evals/compact/baselines/gzip_compact_v1.json
git add backend/tests/evals/compact/baselines/gzip_compact_v1.json
git commit -m "test(evals): populate gzip_compact_v1 baseline with real scores"
```

## Fixtures

10 multi-turn conversations in `fixtures.jsonl`. Each has:
- `id` — unique identifier
- `conversation` — list of `{role, content}` messages (≥6 turns)
- `test_prompt` — downstream query tested after compression
- `must_contain` — keywords the response MUST preserve for a perfect score (10/10)

Scenarios covered: summary-after-multi-turn, cross-turn reference, role consistency, code+debug, multilingual mix, long-doc summary, fact recall, multimodal placeholder, preference reuse, persona injection.

## Scoring

`scorer.py` does keyword-match scoring (0-10): `score = round(matched / total * 10)`.

LLM-as-judge scoring is future work — when needed, swap `scorer.py` keeping the same return shape (`{score, matched, missing}`).
