# SFE Judge / Generate Decoupled Adaptation

## Overview

`sfe-en` and `sfe-zh` have been adapted to support the standalone judge pipeline (`lmms-eval judge` + `lmms-eval aggregate`) used by the rest of the framework. This separates **generation** from **judging/aggregation**.

---

## What Changed

### 1. Generation Phase (`lmms-eval`)
- **No LLM calls during generation.**
- `sfe_process_results` computes all rule-based metrics locally:
  - `open_ended`: Rouge-L, BERTScore, BLEU, METEOR
  - `BBox` (E011/E012): IoU, execute_success_rate
  - `mcq` / `exact_match`: returns `exact_match=0.0` and `needs_llm_judge=True`
- Each sample now carries a structured `sfe_info` dict (containing `id`, `field`, `question_type`, `answer`, `parsed_pred`, and all per-metric arrays) so that aggregation can run offline later.

### 2. Judge Phase (`lmms-eval judge`)
- Works in `auto` mode (recommended), `rule`, or `llm`.
- For `open_ended` and `BBox` tasks, rule-based scores are reused directly.
- For `mcq` / `exact_match`, the `JudgeRunner` detects `needs_llm_judge=True` and calls a **0–10 score judge** (`evaluate_score`) using the task-specific SFE prompt. The returned score is normalized to `0.0–1.0` and written back into `exact_match`.
- When the original `doc` was dropped from the JSONL (standard tracker behavior), `sfe_process_results` automatically rebuilds the minimal doc from `__sample_context__` so judging can still rerun.

### 3. Aggregate Phase (`lmms-eval aggregate`)
- `Aggregator` recognizes tasks matching `sfe` via `SPECIAL_AGGREGATIONS` and calls `sfe_standalone_aggregate`.
- Outputs all supported metrics:
  - `exact_match`
  - `rouge_score`
  - `bert_score`
  - `bleu_score`
  - `meteor_score`
  - `llm_score`
  - `execute_succ_rate`
  - `iou_score`
  - `acc@0.1` ~ `acc@0.9`

---

## Typical Usage

```bash
# 1. Generation (no LLM judge)
lmms-eval \
  --model qwen3_vl \
  --tasks sfe-en,sfe-zh \
  --log_samples \
  --output_path ./eval_result

# 2. Standalone judge
lmms-eval judge \
  -i "eval_result/*_samples_sfe-*.jsonl" \
  \
  --judge-model gpt-4o-mini \
  --parallel 8 \
  --output-dir judged_results/

# 3. Aggregate
lmms-eval aggregate \
  -i judged_results/xxx_samples_sfe-en.jsonl \
  -t sfe-en
```

---

## Implementation Details

### Files Modified
- **`utils.py`**:
  - `sfe_process_results`: removed inline LLM call, added `sfe_info` and `needs_llm_judge`, compatible with dropped docs.
  - `sfe_standalone_aggregate`: new aggregation entrypoint for the standalone pipeline.
- **`lmms_eval/llm_judge/base.py`**:
  - Added `evaluate_score` / `evaluate_score_async` for range-based scoring (0–10).
- **`lmms_eval/llm_judge/standalone.py`**:
  - `_extract_existing_metrics`: whitelists SFE fields.
  - `_needs_llm_judge`: reacts to `needs_llm_judge=True`.
  - `_apply_llm_judge`: SFE branch uses `evaluate_score(0, 10)` with the original SFE prompt.
  - `compute_summary`: SFE-aware summary (avoids binary 0/1 fallback math).
- **`lmms_eval/llm_judge/aggregator.py`**:
  - Registered `sfe` in `SPECIAL_AGGREGATIONS` pointing to `sfe_standalone_aggregate` with `data_key="sfe_info"`.

---

## Notes
- Both `sfe-en` and `sfe-zh` share the same `utils.py`, so both are adapted simultaneously.
- The original inline LLM client (`get_chat_response` with hard-coded `MODEL_VERSION`) is **no longer used** during normal evaluation; you control the judge model via `--judge-model` and `--judge-base-url` in the judge step.
- If you only want to rerun rule metrics, pass `--judge-model none` or rely on the built-in rule-only behavior. `mcq` / `exact_match` will keep `exact_match=0` and will not be LLM-scored.
