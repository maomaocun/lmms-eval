#!/bin/bash
# Full evaluation of Qwen3-VL-8B-Instruct on all 4 industrial benchmarks.
#
# Verified end-to-end on 1 sample each (test1 v5, 23:33 CST 04-29-2026):
#   kernelbench_level1: compiled=1.0, correctness=1.0  -> real CUDA execution
#   tritonbench_g    : call_acc=0, exec_acc=0          -> real Triton execution
#   supercoder_val   : correctness=0, speedup=1.0      -> real gcc + hyperfine
#   cad_coder_test100: valid_syntax=0, valid_step=0    -> real CadQuery execution
#
# Tuning derived from the test runs:
#   - GPU_MEMORY_UTILIZATION=0.50 (vLLM ~40 GB; leaves room for kernel scoring)
#   - enforce_eager=True (avoid torch.compile cache collision between 4 jobs)
#   - per-task VLLM_CACHE_ROOT and TRITON_CACHE_DIR (no cross-task corruption)
#   - max_model_len=8192 (sized to longest prompt seen in test)

set -u  # -e omitted: collect whatever metrics succeed

# ---- dependency guard -------------------------------------------------------
echo "=== Pre-flight checks ==="
python3 -c "import kernelbench"  2>/dev/null || { echo "ABORT: kernelbench not installed"; exit 1; }
python3 -c "import cadquery"     2>/dev/null || { echo "ABORT: cadquery not installed"; exit 1; }
python3 -c "import triton"       2>/dev/null || { echo "ABORT: triton not installed"; exit 1; }
python3 -c "import pytablewriter" 2>/dev/null || { echo "ABORT: pytablewriter not installed"; exit 1; }
which gcc       >/dev/null 2>&1  || { echo "ABORT: gcc not on PATH"; exit 1; }
which hyperfine >/dev/null 2>&1  || { echo "ABORT: hyperfine not on PATH"; exit 1; }
echo "  Dependencies OK"

# ---- reward-hacking guard ---------------------------------------------------
if [ -n "${LMMS_KERNELBENCH_DRY_RUN:-}" ]; then
    echo "ABORT: LMMS_KERNELBENCH_DRY_RUN set — would fake-zero kernelbench scores."
    exit 1
fi
if [ -n "${LMMS_CADCODER_SKIP_IOU:-}" ]; then
    echo "ABORT: LMMS_CADCODER_SKIP_IOU set — would skip CAD IoU scoring."
    exit 1
fi
echo "  Reward-hacking guards: clear"

# ---- GPU clean check --------------------------------------------------------
gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${gpu_used}" -gt 1000 ]; then
    echo "ABORT: a GPU is already using ${gpu_used} MiB. Free GPUs first."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 1
fi
echo "  GPUs free"
echo ""

unset VLLM_USE_MODELSCOPE LMDEPLOY_USE_MODELSCOPE 2>/dev/null || true
export HF_DATASETS_OFFLINE=0
export HF_HUB_DOWNLOAD_TIMEOUT=300

MODEL_PATH="/mnt/workspace/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_BASE="./logs/4bench_full_$(date +%Y%m%d_%H%M)"
GPU_MEMORY_UTILIZATION=0.50
TP=2
BATCH_SIZE=4

# Production executor timeouts
export LMMS_CADCODER_EXEC_TIMEOUT=120
export LMMS_CADCODER_TIMEOUT=300
unset LMMS_CADCODER_SKIP_IOU 2>/dev/null || true
export LMMS_KERNELBENCH_TIMEOUT=300
export LMMS_KERNELBENCH_NUM_CORRECT=5
export LMMS_KERNELBENCH_NUM_PERF=10
export LMMS_SUPERCODER_TIMEOUT=120
export LMMS_SUPERCODER_MAX_CASES=10
export LMMS_TRITONBENCH_TIMEOUT=120

mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "  Full 4-benchmark Evaluation"
echo "  Model : ${MODEL_PATH}"
echo "  Output: ${OUTPUT_BASE}"
echo "  Start : $(date)"
echo "=============================================="

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
        --batch_size "${BATCH_SIZE}" \
        --output_path "${out}" \
        --log_samples --log_samples_suffix "${name}" \
        > "${log}" 2>&1
    local exit_code=$?
    echo "[${name}] DONE   exit=${exit_code}  $(date)  -> ${log}"
}

# 4 parallel tracks, one per GPU pair
#   GPU 0,1 -> CAD-Coder       (full test set, 7355 samples)
#   GPU 2,3 -> KernelBench     (all 4 levels, 270 problems)
#   GPU 4,5 -> SuperCoder      (val set)
#   GPU 6,7 -> TritonBench     (G + T, ~370 problems)
run_task "0,1" "cad_coder"    "cad_coder_test"  &
run_task "2,3" "kernelbench"  "kernelbench_level1,kernelbench_level2,kernelbench_level3,kernelbench_level4" &
run_task "4,5" "supercoder"   "supercoder_val"  &
run_task "6,7" "tritonbench"  "tritonbench_g,tritonbench_t" &
wait

echo ""
echo "=============================================="
echo "  All evaluations finished: $(date)"
echo "=============================================="
echo "Results:"
find "$OUTPUT_BASE" -name "*results*.json" -not -path "*/.vllm_cache*" -not -path "*/.triton_cache*" | sort
