# Judge Process Examples

This directory contains examples demonstrating how to use the `lmms-eval judge` command to separate generation and judging phases.

## Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `quick_start_local_judge.sh` | One-command local judge | `bash quick_start_local_judge.sh results.jsonl` |
| `run_judge_with_local_vllm.sh` | Full-featured with options | `bash run_judge_with_local_vllm.sh -i results.jsonl` |
| `run_all_examples.sh` | Run all Python examples | `bash run_all_examples.sh` |

## Quick Start

### 0. Local vLLM/SGLang Setup (Optional)

Use a local LLM server instead of cloud APIs for faster, private judging:

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192

# Or start SGLang server
python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct

# Configure environment
export JUDGE_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=dummy  # Local servers often don't validate keys
export JUDGE_MODEL=Qwen2.5-VL-7B-Instruct

# Run judging with local LLM
lmms-eval judge --input_result results.jsonl --judge-mode llm
```

See `06_local_vllm_judge.py` for detailed examples.

### Quick Start: One-Command Local Judge

Use the provided script to start vLLM and run judge in one command:

```bash
# Quick start (auto-detects model and task)
bash quick_start_local_judge.sh results/samples_mathvision.jsonl

# Specify model
bash quick_start_local_judge.sh results/samples.jsonl Qwen/Qwen2.5-VL-7B-Instruct
```

### Full-Featured Script

For more control, use the full-featured script:

```bash
# Show all options
bash run_judge_with_local_vllm.sh --help

# Basic usage
bash run_judge_with_local_vllm.sh -i results/samples.jsonl

# Specify task and output
bash run_judge_with_local_vllm.sh \
  -i results/samples.jsonl \
  -t mathvision_reason_testmini \
  -d judged_results/

# High-throughput batch processing
bash run_judge_with_local_vllm.sh \
  -i "results/*.jsonl" \
  --parallel 16 \
  --gpu-mem 0.95

# Use different model
bash run_judge_with_local_vllm.sh \
  -i results/samples.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct

# Keep vLLM server running after judging
bash run_judge_with_local_vllm.sh \
  -i results/samples.jsonl \
  --keep-server
```

### 1. Rule-based Re-judging

Re-judge existing results with rule-based scoring:

```bash
python 01_rule_based_judging.py
```

### 2. LLM-as-Judge

Use an LLM (e.g., GPT-4o) to judge model outputs:

```bash
export JUDGE_API_KEY=sk-your-key
python 02_llm_judge.py
```

### 3. Batch Processing

Process multiple result files at once:

```bash
python 03_batch_processing.py
```

### 4. Custom Judge Logic

Implement custom judging logic:

```bash
python 04_custom_judge.py
```

## Common Use Cases

### Use Case 1: Experiment with Different Judges

You can run generation once, then experiment with different judging criteria:

```bash
# Step 1: Generate (run once)
lmms-eval --model qwen2_5_vl --tasks mathvision_reason_testmini --log_samples

# Step 2: Try different judging approaches
lmms-eval judge --input_result results/samples_mathvision.jsonl --judge-mode rule
lmms-eval judge --input_result results/samples_mathvision.jsonl --judge-mode llm --judge-model gpt-4o
lmms-eval judge --input_result results/samples_mathvision.jsonl --judge-mode llm --judge-model gpt-4o-mini
```

### Use Case 2: Compare Model Outputs

Judge outputs from multiple models using the same criteria:

```bash
# Judge model A outputs
lmms-eval judge --input_result results/model_a_samples.jsonl -d judged/model_a/

# Judge model B outputs with same criteria
lmms-eval judge --input_result results/model_b_samples.jsonl -d judged/model_b/

# Compare results
python 05_compare_models.py
```

### Use Case 3: Local LLM-as-Judge (Privacy & Speed)

Use a locally deployed model instead of cloud APIs for data privacy and faster inference:

```bash
# Step 1: Start local vLLM server
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# Step 2: Configure environment (in another terminal)
export JUDGE_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=dummy  # Local servers often don't validate
export JUDGE_MODEL=Qwen2.5-VL-7B-Instruct
export JUDGE_MODE=llm

# Step 3: Run judging with local LLM
lmms-eval judge --input_result results/samples.jsonl -d judged_local/

# Step 4: Compare with cloud judge results
lmms-eval judge --input_result results/samples.jsonl \
  --judge-mode llm \
  --judge-model gpt-4o-mini \
  -d judged_cloud/
```

**Why use local judge?**
- **Privacy**: Data never leaves your infrastructure
- **Speed**: ~10-50ms latency vs ~100-500ms for cloud APIs
- **Cost**: No per-token fees after initial hardware cost
- **Customization**: Use fine-tuned models for domain-specific judging

**Supported Local Servers:**
| Server | Command | Default Port |
|--------|---------|--------------|
| vLLM | `vllm serve <model>` | 8000 |
| SGLang | `python -m sglang.launch_server --model-path <model>` | 30000 |
| LM Studio | Start via GUI | 1234 |
| TGI | `text-generation-launcher --model-id <model>` | 8080 |

### Use Case 4: Human-in-the-Loop

Export results for human review:

```bash
# Add judge metadata
lmms-eval judge --input_result results/samples.jsonl -o results/for_review.jsonl --judge-mode rule

# Human reviews and marks correct/incorrect
# Then re-import...
```

## Environment Configuration

### Quick Setup (Recommended)

Create a local configuration file (not committed to git):

```bash
# Copy the example template
cp .env.judge.example .env.judge

# Edit with your API key
nano .env.judge
```

The `run_all_examples.sh` script automatically sources `.env.judge` if it exists.

### Configuration File (.env.judge)

#### Option A: Cloud LLM (OpenAI, Azure)

```bash
# Required for LLM judge
export JUDGE_API_KEY="sk-your-openai-api-key"

# Optional settings
export JUDGE_MODEL="gpt-4o-mini"      # Judge model name
export JUDGE_BASE_URL="https://api.openai.com/v1"
export JUDGE_MODE="auto"              # rule | llm | auto
export JUDGE_MAX_CONCURRENT="8"       # Parallel workers
```

#### Option B: Local vLLM

```bash
# Local vLLM server
export JUDGE_BASE_URL="http://localhost:8000/v1"
export JUDGE_API_KEY="dummy"          # Local servers often don't validate
export JUDGE_MODEL="Qwen2.5-VL-7B-Instruct"
export JUDGE_MODE="llm"
export JUDGE_MAX_CONCURRENT="16"      # Can use higher concurrency locally
```

#### Option C: Local SGLang

```bash
# SGLang server (default port 30000)
export JUDGE_BASE_URL="http://localhost:30000/v1"
export JUDGE_API_KEY="dummy"
export JUDGE_MODEL="Meta-Llama-3-8B-Instruct"
export JUDGE_MODE="llm"
```

#### Option D: Remote vLLM (Another Machine)

```bash
# vLLM running on another GPU server
export JUDGE_BASE_URL="http://192.168.1.100:8000/v1"
export JUDGE_API_KEY="your-secret-key"  # If you set authentication
export JUDGE_MODEL="Qwen2.5-VL-72B-Instruct"
export JUDGE_MODE="llm"
```

### Manual Environment Setup

```bash
# Required for LLM judge
export JUDGE_API_KEY=sk-your-api-key
export JUDGE_MODEL=gpt-4o-mini
export JUDGE_BASE_URL=https://api.openai.com/v1  # Optional

# Optional configurations
export JUDGE_MODE=auto
export JUDGE_MAX_CONCURRENT=8
```

## Output Format

The judged JSONL files contain additional fields:

```json
{
  "doc_id": 0,
  "doc": {...},
  "filtered_resps": "...",
  "metrics": {
    "acc_score": 1.0,
    "format_score": 1.0,
    "llm_judge_score": 1,        // Added for LLM judge
    "llm_judge_raw": "..."       // Raw LLM response
  },
  "judge_mode": "rule"           // "rule", "llm", "llm_fallback", or "error"
}
```

## Troubleshooting

### Issue: "Task not found"

Solution: Specify task explicitly with `-t` or ensure filename matches pattern `*_samples_{task}.jsonl`

### Issue: "JUDGE_API_KEY not set"

Solution: Set environment variable or pass `--judge-api-key` flag

### Issue: Import errors

Solution: Ensure you're using the correct Python environment:

```bash
/mnt/cpfs/yangyicun/lmms-eval/.venv/bin/python your_script.py
```

### Issue: "Local LLM server not accessible"

**Symptoms**: Judge fails with connection error to localhost

**Solutions**:

1. **Check if server is running**:
   ```bash
   curl http://localhost:8000/v1/models
   ```

2. **Verify port and URL**:
   ```bash
   # vLLM default
   export JUDGE_BASE_URL=http://localhost:8000/v1
   
   # SGLang default
   export JUDGE_BASE_URL=http://localhost:30000/v1
   ```

3. **Check server logs** for errors

4. **For GPU memory issues**, reduce max model len:
   ```bash
   vllm serve <model> --max-model-len 4096
   ```

### Issue: "API key required" for local server

**Solution**: Set a dummy key (most local servers don't validate):
```bash
export JUDGE_API_KEY=dummy
```

## Advanced Usage

### High-Throughput Local Judging

For judging thousands of samples locally with maximum speed:

```bash
# 1. Start vLLM with optimized settings
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching

# 2. Configure high concurrency
export JUDGE_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=dummy
export JUDGE_MAX_CONCURRENT=32  # Adjust based on GPU memory

# 3. Run judging
lmms-eval judge \
  --input_result "results/*.jsonl" \
  --judge-mode llm \
  --parallel 32 \
  -d judged/
```

### Comparing Local vs Cloud Judge

```bash
# Create comparison script
#!/bin/bash
set -e

RESULTS_FILE="results/samples_task.jsonl"

# Local judging
echo "Judging with local vLLM..."
export JUDGE_BASE_URL=http://localhost:8000/v1
export JUDGE_API_KEY=dummy
lmms-eval judge -i $RESULTS_FILE -d judged/local/

# Cloud judging
echo "Judging with GPT-4o-mini..."
export JUDGE_BASE_URL=https://api.openai.com/v1
export JUDGE_API_KEY=$OPENAI_API_KEY
export JUDGE_MODEL=gpt-4o-mini
lmms-eval judge -i $RESULTS_FILE -d judged/cloud/

# Compare
python << 'EOF'
import json

local = [json.loads(l) for l in open("judged/local/samples_task.jsonl")]
cloud = [json.loads(l) for l in open("judged/cloud/samples_task.jsonl")]

local_acc = sum(s["metrics"]["acc_score"] for s in local) / len(local)
cloud_acc = sum(s["metrics"]["acc_score"] for s in cloud) / len(cloud)

print(f"Local accuracy: {local_acc:.2%}")
print(f"Cloud accuracy: {cloud_acc:.2%}")
print(f"Agreement: {sum(1 for l,c in zip(local,cloud) if l['metrics']['acc_score']==c['metrics']['acc_score'])/len(local):.2%}")
EOF
```
