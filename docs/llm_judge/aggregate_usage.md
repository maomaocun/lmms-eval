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

### MMMU Qwen3 Official (`mmmu_val_qwen3_official`)

```bash
lmms-eval aggregate \
  --input judged_mmmu.jsonl \
  --task mmmu_val_qwen3_official
```

**Output metrics:**
- `accuracy`: Split-based accuracy computed from nested official metrics (`mmmu_acc_official`)

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
    },
    "mmmu_val_qwen3_official": {
        "module": "lmms_eval.tasks.mmmu.utils_qwen3_official",
        "accuracy_func": "mmmu_qwen3_official_aggregate_accuracy",
        "data_key": "mmmu_acc_official",
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
