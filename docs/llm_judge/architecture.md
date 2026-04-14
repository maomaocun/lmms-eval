# LLM Judge Architecture

This document describes the actual implementation of the standalone judge pipeline in `lmms-eval`. It covers how `lmms-eval judge` and `lmms-eval aggregate` work under the hood.

---

## 1. Overview

The standalone judge system separates **generation** from **judging**:

1. **Generation phase**: `lmms-eval` runs inference and saves per-sample results to a JSONL file.
2. **Judging phase**: `lmms-eval judge` reads the JSONL, re-runs (or applies) scoring logic, and optionally uses an LLM as a fallback judge.
3. **Aggregation phase**: `lmms-eval aggregate` computes task-level metrics from the judged JSONL, including special cross-sample logic (e.g., WeMath multi-step analysis).

The core classes are:

- `lmms_eval.llm_judge.standalone.JudgeRunner` — per-sample judging
- `lmms_eval.llm_judge.aggregator.Aggregator` — task-level aggregation

---

## 2. JudgeRunner Pipeline

### 2.1 Entry Point

```python
# lmms_eval/llm_judge/standalone.py
runner = JudgeRunner(judge_mode="auto", judge_model="gpt-4o-mini", parallel=8)
results = runner.judge_file(Path("results.jsonl"), "mathvision_reason_testmini")
```

The framework always runs in `auto` mode: rule-based judging first, then LLM fallback for low-scoring samples.

### 2.2 Per-Sample Flow (`_judge_sample`)

```
┌─────────────────┐
│  JSONL Sample   │
│  (doc, target,  │
│   filtered_resps)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│ Detect doc_was_dropped      │
│ (tracker pops doc before    │
│  saving JSONL)              │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Rebuild doc if needed       │
│ • Use submission dict       │
│ • Inject __sample_context__ │
│ • Reuse pre-computed metrics│
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Rule-based judging          │
│ (process_results_fn)        │
│  → metrics + judge_mode     │
└────────┬────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌─────────────┐
│ score  │  │ score == 0  │  (auto mode)
│ > 0    │  │ or False    │
└──┬─────┘  └──────┬──────┘
   │               │
   ▼               ▼
 keep metrics  LLM fallback
               (_apply_llm_judge)
```

### 2.3 Handling Dropped Documents

The evaluation tracker drops the `doc` field before saving JSONL to reduce file size. `JudgeRunner` handles this in two ways:

1. **`_extract_existing_metrics`** — If `doc` is missing and the sample already contains top-level metric keys (`acc_score`, `accuracy`, `wemath_loose`, etc.), they are reused instead of re-running `process_results` with an incomplete doc.

2. **`__sample_context__`** — If the doc is empty or minimal, `JudgeRunner` rebuilds it from `sample["submission"]` or injects the full sample dict as `doc["__sample_context__"]` so that tasks like `mmmu_val_qwen3_official` can reconstruct what they need at runtime.

### 2.4 LLM Fallback (`_apply_llm_judge`)

When rule-based judging returns `0`/`False`, the runner calls:

```python
judge_result = self._judge_provider.evaluate_binary(
    question=question,
    answer=answer,
    prediction=prediction,
    output_format="0/1",
)
```

This is a **generic binary judge** by default. However, the runner also supports a lightweight task-specific prompt hook:

- **`get_judge_prompt(doc, prediction, target) -> str`** — If the task module defines this function, `JudgeRunner` will call it and pass the resulting string as the `custom_prompt` to the binary judge. This is used by chemistry tasks (MolParse, OpenRxn) to inject domain-specific evaluation instructions.

- **SFE exception** — The SFE task uses a custom 0-10 scoring prompt that is currently hard-coded in `standalone.py` rather than via `get_judge_prompt`. It is triggered when `process_results` sets `needs_llm_judge=True` for `mcq` / `exact_match` questions.

The resulting metrics include:
- `llm_judge_score` — `0` or `1`
- `llm_judge_raw` — raw LLM response
- `llm_judge_model` — model name
- `llm_judge_success` — whether the call succeeded

### 2.5 Summary Computation (`compute_summary`)

After judging all samples, `JudgeRunner.compute_summary()` produces aggregate numbers:

**Case 1 — Flat scores** (MathVision, WeMath, etc.):
```python
{
  "rule_acc": 0.5234,
  "llm_fallback_acc": 0.2312,
  "total_acc": 0.7546
}
```
- `rule_acc` = average of `acc_score` across all samples
- `llm_fallback_acc` = `(1 - rule_acc) * avg(llm_judge_score)`
- `total_acc` = `rule_acc + llm_fallback_acc`

**Case 2 — Nested official metrics** (MMMU / MMMU-Pro official):
```python
{
  "rule_acc": 0.5123,
  "llm_fallback_acc": 0.2456,
  "total_acc": 0.7579
}
```
- Computed from nested dicts containing `hit`, `extraction_method`, and `extraction_success`
- Distinguishes rule-based hits from GPT-judge fallback hits

---

## 3. Aggregator Architecture

The `Aggregator` class converts per-sample judged results into final task metrics.

### 3.1 Two Paths

1. **Special aggregation** — For tasks with cross-sample logic (WeMath multi-step, MMMU official split-based aggregation). Registered in `Aggregator.SPECIAL_AGGREGATIONS`.
2. **Generic aggregation** — Simple averaging of numeric metrics for all other tasks.

### 3.2 Special Aggregation Registry

```python
# lmms_eval/llm_judge/aggregator.py

SPECIAL_AGGREGATIONS = {
    "wemath": {
        "module": "lmms_eval.tasks.wemath.reasoning.utils",
        "loose_func": "wemath_aggregate_results_loose",
        "strict_func": "wemath_aggregate_results_strict",
        "data_key": "wemath_loose",
    },
    "mathvision": {
        "module": "lmms_eval.tasks.mathvision.utils",
        "accuracy_func": "mathvision_aggregate_results_eval",
        "data_key": "mathvision_standard_eval",
        "score_key": "scores",
        "exclude_patterns": ["mathvision_reason"],
    },
    "mathvision_testmini_qwen3": {
        "module": "lmms_eval.tasks.mathvision.utils_qwen3",
        "accuracy_func": "mathvision_aggregate_results_qwen3",
        "data_key": "mathvision_qwen3_eval",
        "score_key": "scores",
    },
    "mmmu_val_qwen3_official": {
        "module": "lmms_eval.tasks.mmmu.utils_qwen3_official",
        "accuracy_func": "mmmu_qwen3_official_aggregate_accuracy",
        "data_key": "mmmu_acc_official",
    },
    "mmmu_pro": {
        "module": "lmms_eval.tasks.mmmu_pro_qwen3_official.utils_qwen3_official",
        "accuracy_func": "mmmu_pro_qwen3_official_aggregate_accuracy",
        "data_key": "mmmu_pro_acc_official",
    },
    "mmbench_en_dev": {
        "module": "lmms_eval.tasks.mmbench.en_utils",
        "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
        "data_key": "gpt_eval_score",
    },
    "mmbench_en_test": {
        "module": "lmms_eval.tasks.mmbench.en_utils",
        "accuracy_func": "mmbench_aggregate_test_results_standalone",
        "data_key": "submission",
    },
}
```

To add a new task with complex aggregation, register it here.

### 3.3 Data Extraction

For special tasks, the aggregator extracts task-specific data from each sample:
- First tries the top-level key (e.g., `sample["wemath_loose"]`)
- Then tries nested inside `sample["metrics"]`
- Passes the collected list to the task's aggregation function

### 3.4 Generic Aggregation

For tasks without special handling:
- Iterates over `sample["metrics"]`
- Averages all numeric/boolean values
- Skips nested dicts and lists

---

## 4. How Task-Specific Judging Actually Works Today

Tasks handle judge-specific behavior in one of three ways:

### 4.1 Inside `process_results`

Tasks like `mmmu_val_qwen3_official` implement their full two-stage pipeline (rule-based extraction + GPT judge) inside `process_results`. When `lmms-eval judge` runs in `auto` or `rule` mode, it simply calls this function.

### 4.2 Generic Binary Fallback (with optional `get_judge_prompt`)

For standard reasoning tasks (MathVision, MathVerse, etc.), `process_results` does rule-based extraction. If it returns `0`, the generic binary LLM judge evaluates the sample.

If the task module exports a `get_judge_prompt(doc, prediction, target)` function, `JudgeRunner` will use its return value as the custom prompt for the binary judge. This allows tasks to customize the judge prompt without modifying the framework. MolParse and OpenRxn use this hook to provide chemistry-specific evaluation criteria.

### 4.3 Batch-Scoring Tasks (MMBench)

`mmbench_en_dev` is a special case: its `process_results` function does **not** score samples individually. Instead, it repackages each sample into a `gpt_eval_score` dict. The actual scoring (rule-based extraction + GPT API fallback) happens inside the aggregation function `mmbench_aggregate_dev_results_eval`, which needs the full batch to handle MMBench's rolling-record logic.

Since the fix in `lmms_eval/cli/judge_cmd.py`, `lmms-eval judge` **automatically invokes the `Aggregator`** for tasks registered in `SPECIAL_AGGREGATIONS` (including all MMBench splits). This means:

- Running `lmms-eval judge -i results.jsonl -t mmbench_en_dev` alone will produce the correct accuracy.
- A separate `lmms-eval aggregate` step is no longer required for MMBench, although the command can still be used standalone if desired.
- `mmbench_en_test` follows the same pattern but only generates a submission file (no accuracy, since test answers are not public).

---

## 5. Note on Historical Design

An earlier design document proposed a full `standalone_judge` hook system with dynamic discovery and a `make_standalone_judge` factory. That full system was never implemented. However, a lightweight task-specific prompt hook (`get_judge_prompt`) **was** later added to `standalone.py` and is actively used by chemistry tasks. The architecture described in this document reflects the code that actually runs.

---

## 6. Adding a New Task

1. **If the task already has a `process_results` function** that handles rule-based scoring and optional LLM judging, no changes to the judge framework are needed. Just run:
   ```bash
   lmms-eval judge -i results.jsonl -t your_task
   ```

2. **If the task needs special aggregation** (cross-sample logic), register it in `lmms_eval/llm_judge/aggregator.py`:
   ```python
   SPECIAL_AGGREGATIONS["your_task"] = {
       "module": "lmms_eval.tasks.your_task.utils",
       "accuracy_func": "your_aggregate_function",
       "data_key": "your_data_key",
   }
   ```

3. **If the task relies on the generic binary LLM fallback**, ensure `process_results` returns a low/False score when the answer is wrong so that `auto` mode triggers the fallback correctly.

4. **(Optional) If the task needs a custom binary judge prompt**, export `get_judge_prompt(doc, prediction, target)` from the task module. `JudgeRunner` will detect it and pass the returned string as the custom prompt to the LLM judge.
