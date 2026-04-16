# MMBench Tasks - Standalone Judge & Aggregate Guide

This directory contains the non-reasoning MMBench task variants. All tasks listed below support the `lmms-eval judge` + `lmms-eval aggregate` workflow for separating generation from scoring.

## Supported Tasks

| Task | Split | Needs `aggregate` | Notes |
|------|-------|-------------------|-------|
| `mmbench_en_dev` | dev | ✅ Yes | GPT-based batch scoring happens in aggregation. |
| `mmbench_en_test` | test | ✅ Yes | Generates a submission Excel file (no public ground-truth). |
| `mmbench_cn_dev` | dev | ✅ Yes | Same as EN dev, uses CN-specific prompts. |
| `mmbench_cn_test` | test | ✅ Yes | Generates a submission Excel file. |
| `mmbench_ru_dev` | dev | ✅ Yes | Russian dev variant. |
| `mmbench_ko_dev` | dev | ✅ Yes | Korean dev variant. |
| `mmbench_cn_cc` | test | ✅ Yes | Chinese cultural-concepts variant. |
| `mmbench_en_dev_lite` | lite | ✅ Yes | Lite subset; matched automatically to EN dev aggregator. |
| `mmbench_cn_dev_lite` | lite | ✅ Yes | Lite subset; matched automatically to CN dev aggregator. |

## Why `aggregate` is Required

MMBench's `process_results` function does **not** produce per-sample scalar scores. Instead, it repackages each sample into `gpt_eval_score` and `submission` dicts. The actual scoring logic (rule-based extraction + GPT API fallback) is batch-based and lives inside the aggregation functions.

## Environment Variables

The standalone pipeline aligns with both the traditional MMBench env vars and the judge-pipeline env vars:

| Variable | Purpose | Fallback chain |
|----------|---------|----------------|
| `OPENAI_API_KEY` | API key for GPT judge | → `JUDGE_API_KEY` → `"YOUR_API_KEY"` |
| `OPENAI_API_URL` | Full chat-completions endpoint | → `JUDGE_BASE_URL`* → OpenAI default |
| `MODEL_VERSION` | Judge model name | → `JUDGE_MODEL` → `gpt-4o-2024-11-20` |

\* If `JUDGE_BASE_URL` contains multiple backends separated by `;`, the first one is used and `/chat/completions` is appended automatically.

## Usage Example

### 1. Generate (already done during normal evaluation)

```bash
lmms-eval --model qwen2_vl --tasks mmbench_en_dev --batch_size 1
```

This produces `samples_mmbench_en_dev.jsonl` in your output directory.

### 2. Judge (reformat data)

```bash
lmms-eval judge \
  --input_result samples_mmbench_en_dev.jsonl \
  --task mmbench_en_dev \
  \
  --output-dir judged_results/
```

Rule-based judging is sufficient here because `process_results` only reformats data; the real scoring happens in the next step.

### 3. Aggregate (actual scoring)

```bash
lmms-eval aggregate \
  --input judged_results/samples_mmbench_en_dev.jsonl \
  --task mmbench_en_dev
```

For **dev** tasks, this prints overall accuracy and category-level accuracies.  
For **test** tasks, this generates a submission Excel file in `./submissions/`.

## Batch Workflow (Multiple Tasks)

```bash
export JUDGE_RESULT_DIR=/path/to/eval_results
export JUDGE_OUTPUT_DIR=/path/to/judged_results

lmms-eval judge \
  --input_result "${JUDGE_RESULT_DIR}"/samples_mmbench_*.jsonl \
  --task auto-detect \
  \
  --output-dir "${JUDGE_OUTPUT_DIR}"

for task in mmbench_en_dev mmbench_cn_dev mmbench_ru_dev mmbench_ko_dev; do
  lmms-eval aggregate \
    --input "${JUDGE_OUTPUT_DIR}"/samples_${task}.jsonl \
    --task ${task}
done
```

## Troubleshooting

- **No accuracy shown after `judge`**: This is expected. Wait for the `aggregate` step.
- **API key errors during `aggregate`**: Ensure `JUDGE_API_KEY` (or `OPENAI_API_KEY`) is exported before running `aggregate`.
- **Lite variants not aggregating**: The lite tasks (`mmbench_en_dev_lite`, `mmbench_cn_dev_lite`) share aggregation logic with their full counterparts and are matched automatically by name.
