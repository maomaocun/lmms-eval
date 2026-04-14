# MMBench EN Reasoning Tasks - Standalone Judge Guide

This directory contains the English reasoning variant of MMBench.

## Supported Tasks

| Task | Needs `aggregate` | Notes |
|------|-------------------|-------|
| `mmbench_en_reasoning` | ❌ No | `judge` produces final per-sample scores directly. |
| `mmbench_en_dev_reasoning` | ❌ No | `judge` produces final per-sample scores directly. |

## How Scoring Works

Unlike the base (non-reasoning) MMBench tasks, the reasoning variant's `process_results` function returns scalar metrics (`acc_score`, `format_score`) for each sample. This means:

- `lmms-eval judge --judge-mode auto` will rule-score each sample and, if needed, fall back to the generic binary LLM judge.
- The summary printed at the end of `judge` is already the final result.
- You **do not** need to run `lmms-eval aggregate` for these tasks.

## Usage Example

### 1. Judge (scores directly)

```bash
lmms-eval judge \
  --input_result samples_mmbench_en_dev_reasoning.jsonl \
  --task mmbench_en_dev_reasoning \
  --judge-mode auto \
  --output-dir judged_results/
```

The command will output `rule_acc`, `llm_fallback_acc`, and `total_acc` upon completion.

### 2. (Optional) Inspect results

```bash
cat judged_results/samples_mmbench_en_dev_reasoning.jsonl | head -n 1 | python -m json.tool
```

Each line contains `metrics.acc_score` and `metrics.format_score`.

## Environment Variables

If the generic binary LLM fallback is triggered (`auto` mode), the judge backend respects the standard judge-pipeline env vars:

| Variable | Purpose | Default |
|----------|---------|---------|
| `JUDGE_API_KEY` | API key for fallback judge | `dummy-key-for-local-vllm` |
| `JUDGE_BASE_URL` | Judge endpoint | `https://api.openai.com/v1` |
| `JUDGE_MODEL` | Judge model | `gpt-4o-mini` |
