#!/bin/bash
# PIPELINE VERIFICATION: 1 sample per benchmark, sequential, with full scoring enabled.
#
# Purpose: confirm end-to-end scoring is ACTUALLY running (not silently zeroing)
# before launching the full evaluation run.
#
# What to check in output:
#   kernelbench  -> "compiled", "correctness" values in results.json (not all 0.0 due to missing pkg)
#   supercoder   -> "correctness", "speedup" in results.json
#   tritonbench  -> "call_acc", "exec_acc" in results.json
#   cad_coder    -> "valid_syntax", "valid_step", "iou" in results.json
#
# Safety checks enforced here:
#   - LMMS_KERNELBENCH_DRY_RUN must NOT be set
#   - LMMS_CADCODER_SKIP_IOU must NOT be set
#   - All dependencies confirmed present before launch

set -u

# ---- dependency guard -------------------------------------------------------
echo "=== Checking dependencies ==="
python3 -c "import kernelbench" 2>/dev/null || { echo "ABORT: kernelbench not installed"; exit 1; }
python3 -c "import cadquery"    2>/dev/null || { echo "WARN: cadquery not installed — iou will be skipped"; }
python3 -c "import triton"      2>/dev/null || { echo "ABORT: triton not installed"; exit 1; }
which gcc      >/dev/null 2>&1  || { echo "ABORT: gcc not on PATH"; exit 1; }
which hyperfine>/dev/null 2>&1  || { echo "WARN: hyperfine not found — speedup will be 1.0 for supercoder"; }
echo "Dependencies OK"

# ---- reward-hacking guard ---------------------------------------------------
if [ -n "${LMMS_KERNELBENCH_DRY_RUN:-}" ]; then
    echo "ABORT: LMMS_KERNELBENCH_DRY_RUN is set — this would report fake zeros. Unset it first."
    exit 1
fi
echo "Reward-hacking guard: LMMS_KERNELBENCH_DRY_RUN not set. Good."

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE 2>/dev/null || true
export HF_DATASETS_OFFLINE=0
export HF_HUB_DOWNLOAD_TIMEOUT=300

MODEL_PATH="/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_BASE="./logs/4bench_test1_$(date +%Y%m%d_%H%M)"
# Critical: keep low so kernelbench/tritonbench scoring subprocess has GPU room.
# Qwen3-VL-8B (~16GB) + KV cache fits in 0.5*80GB; leaves ~40GB for kernel exec.
GPU_MEMORY_UTILIZATION=0.50
TP=2

# Full scoring timeouts even for 1-sample test (we want to see real execution)
export LMMS_CADCODER_EXEC_TIMEOUT=120
export LMMS_CADCODER_TIMEOUT=300
unset LMMS_CADCODER_SKIP_IOU   2>/dev/null || true   # MUST be off for real IoU
export LMMS_KERNELBENCH_TIMEOUT=300
export LMMS_KERNELBENCH_NUM_CORRECT=3
export LMMS_KERNELBENCH_NUM_PERF=5
export LMMS_SUPERCODER_TIMEOUT=120
export LMMS_SUPERCODER_MAX_CASES=5
export LMMS_TRITONBENCH_TIMEOUT=120

mkdir -p "$OUTPUT_BASE"

echo ""
echo "=============================================="
echo "  Pipeline test (1 sample / benchmark)"
echo "  Model : ${MODEL_PATH}"
echo "  Output: ${OUTPUT_BASE}"
echo "  Start : $(date)"
echo "=============================================="
echo ""

run_task () {
    local gpus="$1"
    local name="$2"
    local tasks="$3"
    local out="${OUTPUT_BASE}/${name}"
    local log="${OUTPUT_BASE}/${name}.log"
    local cache="${OUTPUT_BASE}/.vllm_cache_${name}"
    local triton_cache="${OUTPUT_BASE}/.triton_cache_${name}"
    mkdir -p "$out" "$cache" "$triton_cache"
    echo "[${name}] START  GPUs=${gpus} tasks=${tasks}  $(date)"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    VLLM_CACHE_ROOT="${cache}" \
    TRITON_CACHE_DIR="${triton_cache}" \
    python -m lmms_eval \
        --model vllm \
        --model_args "model=${MODEL_PATH},tensor_parallel_size=${TP},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=8192,enforce_eager=True" \
        --tasks "${tasks}" \
        --limit 1 \
        --batch_size 1 \
        --output_path "${out}" \
        --log_samples --log_samples_suffix "${name}" \
        > "${log}" 2>&1
    local exit_code=$?
    echo "[${name}] DONE   exit=${exit_code}  $(date)"
    # Print the scores so we can see immediately if scoring ran
    echo "[${name}] --- scores ---"
    python3 -c "
import json, glob, sys
files = glob.glob('${out}/**/results.json', recursive=True)
if not files:
    print('  WARNING: no results.json found — pipeline may have crashed')
    sys.exit(0)
for f in files:
    d = json.load(open(f))
    results = d.get('results', {})
    for task, metrics in results.items():
        print(f'  {task}:')
        for k, v in metrics.items():
            if not k.startswith('alias'):
                print(f'    {k}: {v}')
" 2>&1
    echo ""
}

# Run all 4 in parallel on separate GPU pairs (same as full run)
# cad_coder_test100: 100-sample subset (avoid 7355-sample preprocessing)
run_task "0,1" "cad_coder"   "cad_coder_test100"  &
run_task "2,3" "kernelbench" "kernelbench_level1"  &
run_task "4,5" "supercoder"  "supercoder_val"      &
run_task "6,7" "tritonbench" "tritonbench_g"       &
wait

echo "=============================================="
echo "  Test run complete: $(date)"
echo "  Review scores above — if all metrics are 0.0"
echo "  and errors appear, fix before full run."
echo "=============================================="
