# Aggregate Command Usage

The `lmms-eval aggregate` command provides task-specific aggregation of judged results, handling complex aggregation logic that cannot be done per-sample (e.g., WeMath's multi-step analysis).

## Overview

When using `lmms-eval judge`, per-sample results are generated. However, some tasks require aggregating multiple samples together to compute final metrics:

- **WeMath**: Computes `Score (Loose)` and `Score (Strict)` based on multi-step question relationships
- **MathVision**: Computes overall accuracy from per-sample scores
- **MMMU Qwen3 Official**: Computes split-based accuracy from nested official metrics
- **Generic tasks**: Simple averaging of numeric metrics

## Basic Usage

### 1. Judge First

```bash
lmms-eval judge \
  --input_result samples_wemath_testmini_reasoning.jsonl \
  --task wemath_testmini_reasoning \
  --output-dir judged_results/
```

### 2. Then Aggregate

```bash
lmms-eval aggregate \
  --input judged_results/samples_wemath_testmini_reasoning.jsonl \
  --task wemath_testmini_reasoning \
  --output final_results.json
```

## Supported Tasks

### WeMath (`wemath_testmini_reasoning`)

```bash
lmms-eval aggregate \
  --input judged_wemath.jsonl \
  --task wemath_testmini_reasoning
```

**Output metrics:**
- `Score (Loose)`: Loose aggregation score
- `Score (Strict)`: Strict aggregation score

These scores reflect the model's ability to handle multi-step mathematical reasoning, including:
- **CompleteMastery**: Both decomposed and combined questions correct
- **RoteMemorization**: Combined correct but decomposed wrong
- **InadequateGeneralization**: Decomposed correct but combined wrong
- **InsufficientKnowledge**: Both wrong

### MathVision (`mathvision_test`)

```bash
lmms-eval aggregate \
  --input judged_mathvision.jsonl \
  --task mathvision_test
```

**Output metrics:**
- `accuracy`: Overall accuracy percentage

### MathVision Qwen3 (`mathvision_testmini_qwen3`)

```bash
lmms-eval aggregate \
  --input judged_mathvision_qwen3.jsonl \
  --task mathvision_testmini_qwen3
```

**Output metrics:**
- `accuracy`: Overall accuracy percentage

**Note:** MathVision Reasoning tasks (`mathvision_reason_test_qwen3`, `mathvision_reason_testmini_qwen3`) use generic aggregation (averaging `acc_score` and `format_score`) and do not require special aggregation.

### MMMU Qwen3 Official (`mmmu_val_qwen3_official`)

```bash
lmms-eval aggregate \
  --input judged_mmmu.jsonl \
  --task mmmu_val_qwen3_official
```

**Output metrics:**
- `accuracy`: Split-based accuracy computed from nested official metrics (`mmmu_acc_official`)

### MMMU-Pro Qwen3 Official (`mmmu_pro_qwen3_official`)

```bash
lmms-eval aggregate \
  --input judged_mmmu_pro.jsonl \
  --task mmmu_pro_qwen3_official
```

**Output metrics:**
- `accuracy`: Overall accuracy computed from nested official metrics (`mmmu_pro_acc_official`)

**Note:** This applies to all MMMU-Pro Qwen3 official variants (`mmmu_pro_standard_qwen3_official`, `mmmu_pro_vision_qwen3_official`, and their reasoning counterparts).

### MMBench EN Dev (`mmbench_en_dev`)

```bash
lmms-eval aggregate \
  --input judged_mmbench.jsonl \
  --task mmbench_en_dev
```

**Output metrics:**
- `accuracy`: Overall accuracy percentage (computed via GPT-based batch evaluation)

**Note:** `lmms-eval judge` now automatically invokes the MMBench batch-scoring aggregation, so running `lmms-eval aggregate` separately is optional. Use the aggregate command only if you need standalone re-aggregation of previously judged results.

### MMBench EN Test (`mmbench_en_test`)

```bash
lmms-eval aggregate \
  --input judged_mmbench.jsonl \
  --task mmbench_en_test
```

**Output:**
- Generates a submission Excel file (`mmbench_en_test_results.xlsx`). No accuracy is reported because the test set answers are not public.

### MMBench Other Languages (`mmbench_cn_dev`, `mmbench_cn_test`, `mmbench_ru_dev`, `mmbench_ko_dev`, `mmbench_cn_cc`)

These variants follow the same patterns as the English splits and are registered for special aggregation.
`lmms-eval judge` handles their aggregation automatically; the separate `lmms-eval aggregate` command is optional.

### SFE (`sfe`)

```bash
lmms-eval aggregate \
  --input judged_sfe.jsonl \
  --task sfe
```

**Output metrics:**
- `exact_match`: Overall exact-match accuracy
- `rouge_score`, `bert_score`, `bleu_score`, `meteor_score`: Text-generation metrics
- `execute_succ_rate`, `iou_score`: Execution / bbox metrics
- `acc@0.1` … `acc@0.9`: IoU accuracy thresholds

SFE uses deferred LLM judging (0-10 scale) for `mcq` and `exact_match` questions during the standalone judge phase.

### Generic Tasks

For tasks without special aggregation requirements, the command performs simple averaging of numeric metrics.

## Command Options

```
lmms-eval aggregate [OPTIONS]

Options:
  --input, -i INPUT       Path to judged JSONL file (required)
  --task, -t TASK         Task name for aggregation (required)
  --metric, -m METRIC     Specific metric to aggregate (optional)
  --output, -o OUTPUT     Output JSON file path (optional)
  --verbose, -v           Enable verbose logging
  -h, --help              Show help message
```

## Output Summary

In addition to task-specific aggregation, `lmms-eval judge` prints a summary that breaks down how much of the final score came from rule-based judging versus LLM fallback:

```json
{
  "rule_acc": 0.5234,
  "llm_fallback_acc": 0.2312,
  "total_acc": 0.7546
}
```

- `rule_acc`: Accuracy from rule-based judging
- `llm_fallback_acc`: Additional accuracy from LLM fallback on failed rule-based samples
- `total_acc`: Combined accuracy (`rule_acc + llm_fallback_acc`)

This breakdown is produced for both flat-score tasks (e.g., MathVision, WeMath) and nested-metric tasks (e.g., MMMU official).

## Shell Script Integration

The `start_vllm_judge_and_run.sh` script automatically runs aggregation after judging:

```bash
export JUDGE_RESULT_DIR=/path/to/results
export JUDGE_OUTPUT_DIR=/path/to/judged
bash examples/judge_process/start_vllm_judge_and_run.sh
```

To skip aggregation:
```bash
export JUDGE_SKIP_AGGREGATE=true
bash examples/judge_process/start_vllm_judge_and_run.sh
```

## Implementation Details

### Special Aggregation Tasks

Tasks that require special aggregation are registered in `lmms_eval/llm_judge/aggregator.py`:

```python
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
    "mmbench_cn_dev": {
        "module": "lmms_eval.tasks.mmbench.cn_utils",
        "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
        "data_key": "gpt_eval_score",
    },
    "mmbench_cn_test": {
        "module": "lmms_eval.tasks.mmbench.cn_utils",
        "accuracy_func": "mmbench_aggregate_test_results_standalone",
        "data_key": "submission",
    },
    "mmbench_ru_dev": {
        "module": "lmms_eval.tasks.mmbench.ru_utils",
        "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
        "data_key": "gpt_eval_score",
    },
    "mmbench_ko_dev": {
        "module": "lmms_eval.tasks.mmbench.ko_utils",
        "accuracy_func": "mmbench_aggregate_dev_results_eval_standalone",
        "data_key": "gpt_eval_score",
    },
    "mmbench_cn_cc": {
        "module": "lmms_eval.tasks.mmbench.cc_utils",
        "accuracy_func": "mmbench_cn_cc_aggregate_dev_results_eval_standalone",
        "data_key": "gpt_eval_score",
    },
    "sfe": {
        "module": "lmms_eval.tasks.sfe.utils",
        "accuracy_func": "sfe_standalone_aggregate",
        "data_key": "sfe_info",
    },
}
```

To add support for a new task with complex aggregation, register it here.

### Data Structure

The aggregate command expects judged JSONL files with the following structure:

```json
{
  "doc_id": 0,
  "metrics": {...},
  "wemath_loose": {
    "ID": "...",
    "key": "2steps_1",
    "acc_score": 1.0,
    ...
  },
  ...
}
```

The command extracts task-specific data (e.g., `wemath_loose`) and passes it to the appropriate aggregation function.
