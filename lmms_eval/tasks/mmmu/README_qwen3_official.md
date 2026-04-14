# MMMU Qwen3 Official Implementation

This directory contains the official Qwen3-VL MMMU evaluation implementation, replicating the logic from the [Qwen3-VL official repository](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu).

## Overview

The `mmmu_val_qwen3_official` task provides an evaluation pipeline that exactly matches the official Qwen3-VL repository's approach:

1. **Two-stage answer extraction**: Rule-based → GPT Judge fallback
2. **Official prompts**: Uses the exact same judge prompts as the official repo
3. **Split-based aggregation**: Reports accuracy by split (dev/validation)

## Files

- `mmmu_val_qwen3_official.yaml` - Task configuration
- `utils_qwen3_official.py` - Official evaluation logic implementation
- `README_qwen3_official.md` - This documentation

## Key Differences from Standard `mmmu_val`

| Aspect | `mmmu_val` | `mmmu_val_qwen3_official` |
|--------|-----------|---------------------------|
| Answer Extraction | Rule-based + random fallback | Rule-based + GPT Judge fallback |
| MCQ Parsing | `parse_mmmu_multi_choice_response()` | Official `can_infer_option()` + `can_infer_text()` |
| Open-ended | Rule-based substring matching | GPT Judge with official prompt |
| Fallback on failure | Random guess | GPT Judge → Random guess |
| Aggregation | By domain category | By split (dev/validation) |

## Usage

### 0. Configure API Keys (Required)

This implementation is **fully compatible** with the Qwen3-VL official repository's API configuration.

**Option A: Use the official setup script (Recommended)**
```bash
source /mnt/cpfs/yangyicun/Qwen3-VL/evaluation/setup_api_keys.sh
```

**Option B: Manual configuration**
```bash
export API_TYPE="compatible"  # or "dash", "openai", "kimi", "mit"
export OPENAI_API_KEY="your-key"
export OPENAI_API_URL="https://yunwu.ai/v1/chat/completions"
export MODEL_VERSION="gpt-4o-mini"
```

**Test your API configuration:**
```bash
bash examples/judge_process/test_qwen3_official_api.sh
```

### 1. Basic Evaluation

```bash
# Source API keys first
source /mnt/cpfs/yangyicun/Qwen3-VL/evaluation/setup_api_keys.sh

# Run evaluation with official logic
lmms-eval \
    --model vllm-backend \
    --model_args "base_url=http://localhost:8000/v1,model=/path/to/Qwen3-VL-8B-Instruct" \
    --tasks mmmu_val_qwen3_official \
    --batch_size 1 \
    --output_path /path/to/results
```

### 2. Using the Shell Script

Update your environment variables and run:

```bash
export TASKS="mmmu_val_qwen3_official"
export MODEL="/path/to/Qwen3-VL-8B-Instruct"
export OUTPUT_PATH="/path/to/results"

bash examples/qwen3-vl/vllm_qwen3_vl_aligned.sh
```

### 3. Using Standalone Judge (for already-generated results)

If you have already generated results and want to apply the official evaluation:

```python
from lmms_eval.tasks.mmmu.utils_qwen3_official import run_official_judge_on_file

# Run official judge on results file
result = run_official_judge_on_file(
    input_file="/path/to/results.jsonl",
    output_file="/path/to/judged_results.jsonl",
    judge_model="gpt-4o-mini",
    max_workers=4
)

print(f"Overall accuracy: {result['overall_accuracy']:.4f}")
```

## Environment Variables

This implementation is **fully compatible** with the Qwen3-VL official repository's API configuration:

### Quick Setup (Recommended)

```bash
# Use the official setup script
source /mnt/cpfs/yangyicun/Qwen3-VL/evaluation/setup_api_keys.sh
```

### Manual Configuration

```bash
# Option 1: yunwu.ai (OpenAI-compatible) - Default
export API_TYPE="compatible"
export OPENAI_API_KEY="your-yunwu-key"
export OPENAI_API_URL="https://yunwu.ai/v1/chat/completions"

# Option 2: DashScope (Alibaba Cloud)
export API_TYPE="dash"
export CHATGPT_DASHSCOPE_API_KEY="your-dashscope-key"
export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# Option 3: OpenAI
export API_TYPE="openai"
export OPENAI_API_KEY="your-openai-key"
export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"

# Option 4: Kimi
export API_TYPE="kimi"
export KIMI_API_KEY="your-kimi-key"
export KIMI_API_BASE="https://api.kimi.com/coding/v1/chat/completions"

# Option 5: MIT Spider
export API_TYPE="mit"
export MIT_SPIDER_TOKEN="your-mit-token"
export MIT_SPIDER_URL="your-mit-url"

# Judge model (default: gpt-4o-mini)
export MODEL_VERSION="gpt-4o-mini"  # or "gpt-4o", etc.
```

### API Type Mapping

| API_TYPE | Environment Variables | Description |
|----------|----------------------|-------------|
| `compatible` | `OPENAI_API_KEY` + `OPENAI_API_URL` | OpenAI-compatible APIs (yunwu.ai, etc.) |
| `dash` | `CHATGPT_DASHSCOPE_API_KEY` + `DASHSCOPE_API_BASE` | Alibaba Cloud DashScope |
| `openai` | `OPENAI_API_KEY` + `OPENAI_API_BASE` | Official OpenAI API |
| `kimi` | `KIMI_API_KEY` + `KIMI_API_BASE` | Moonshot Kimi API |
| `mit` | `MIT_SPIDER_TOKEN` + `MIT_SPIDER_URL` | MIT Spider API |

## Output Format

The evaluation produces results in the following format:

```json
{
  "id": "validation_Accounting_1",
  "split": "validation",
  "question_type": "multiple-choice",
  "question": "...",
  "answer": "A",
  "parsed_pred": "A",
  "hit": 1,
  "extraction_method": "rule",
  "extraction_success": true,
  "extraction_log": "Rule extract success with rule result: A prediction: The answer is A"
}
```

## Aggregation Results

The final aggregation report shows:

```
==================================================
Official MMMU Evaluation Results:
==================================================
Accuracy for dev split: 0.6234 (123/197)
Accuracy for validation split: 0.5847 (407/696)
Overall accuracy: 0.5944 (530/893)
==================================================
```

## Implementation Details

### Rule-Based Extraction (`can_infer`)

1. **`can_infer_option()`**: Extracts option letters (A, B, C, D) from the response
   - Handles punctuation cleaning
   - Checks for single unique option letter
   - Returns 'Z' for refusal to answer

2. **`can_infer_text()`**: Matches option text content
   - Case-insensitive matching
   - Guards against very long responses
   - Returns option letter if exactly one matches

### GPT Judge Fallback

When rule-based extraction fails, the system calls a GPT judge with the official prompt:

```
You are an AI assistant who will help me to match an answer with several options 
of a single-choice question. You are provided with a question, several options, 
and an answer, and you need to find which option is most similar to the answer. 
If the meaning of all options are significantly different from the answer, output Z. 
Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z.
```

### Why This Matters

The official implementation consistently scores **~4 points higher** than the standard `mmmu_val` implementation because:

1. GPT Judge recovers correct answers that rule-based parsing misses
2. More sophisticated option matching logic
3. Better handling of varied response formats (especially from thinking models)

## References

- [Qwen3-VL Official Repository](https://github.com/QwenLM/Qwen3-VL)
- [Official MMMU Evaluation Code](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)
- [MMMU Paper](https://arxiv.org/abs/2311.16502)

## Citation

If you use this evaluation implementation, please cite both the MMMU paper and Qwen3-VL:

```bibtex
@article{yue2023mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI},
  author={Yue, Xiang and Ni, Yuansheng and Zhang, Kai and Zheng, Tianyu and Liu, Ruoqi and Zhang, Ge and Stevens, Samuel and Jiang, Dongfu and Ren, Weiming and Sun, Yuxuan and others},
  journal={arXiv preprint arXiv:2311.16502},
  year={2023}
}

@article{qwen3vl2025,
  title={Qwen3-VL Technical Report},
  author={Qwen Team},
  journal={arXiv preprint},
  year={2025}
}
```
