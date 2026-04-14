# MMBench Reasoning Tasks - Standalone Judge Guide

This directory contains the Chinese and English reasoning variants of MMBench.

## Supported Tasks

| Task | Needs `aggregate` | Notes |
|------|-------------------|-------|
| `mmbench_cn_dev_reasoning` | ❌ No | `judge` produces final per-sample scores directly. |
| `mmbench_cn_test_reasoning` | ❌ No | `judge` produces final per-sample scores directly. |
| `mmbench_en_dev_reasoning` | ❌ No | `judge` produces final per-sample scores directly. |
| `mmbench_en_test_reasoning` | ❌ No | `judge` produces final per-sample scores directly. |

## How Scoring Works

The reasoning variants use `make_reasoning_process_results`, which returns scalar metrics for every sample:

- `acc_score` — correctness of the final answer
- `format_score` — format compliance (e.g., `<think>` / `<answer>` tags)

Because the scores are already produced per-sample, the standalone `lmms-eval judge` command can compute final averages on its own. You **do not** need `lmms-eval aggregate`.

## Usage Example

### Judge directly (dev or test)

```bash
lmms-eval judge \
  --input_result samples_mmbench_cn_dev_reasoning.jsonl \
  --task mmbench_cn_dev_reasoning \
  --judge-mode auto \
  --output-dir judged_results/
```

You will see `rule_acc`, `llm_fallback_acc`, and `total_acc` in the terminal output immediately after judging finishes.

### Inspect per-sample results

```bash
cat judged_results/samples_mmbench_cn_dev_reasoning.jsonl | head -n 1 | python -m json.tool
```

Look for `metrics.acc_score` in each JSON line.

## Environment Variables

When `--judge-mode auto` triggers the generic binary LLM fallback, the following env vars are respected:

| Variable | Purpose | Default |
|----------|---------|---------|
| `JUDGE_API_KEY` | API key for fallback judge | `dummy-key-for-local-vllm` |
| `JUDGE_BASE_URL` | Judge endpoint | `https://api.openai.com/v1` |
| `JUDGE_MODEL` | Judge model | `gpt-4o-mini` |
