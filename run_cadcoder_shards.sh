#!/bin/bash
# CAD-Coder full test set, sharded 4-way across all 8 H800 GPUs.
# Replaces the previous single-job run (35h ETA) with 4 parallel slices (~9-12h).
#
# Each shard:
#   * vLLM TP=2 on its own GPU pair
#   * own VLLM_CACHE_ROOT and TRITON_CACHE_DIR (no cross-shard collision)
#   * own --offset / --limit window into the 7355-sample test split
#   * own output directory (results merged at the end)
#
# Total: 7355 samples split into 4 ranges:
#   shard_a: 0-1838    (1839)
#   shard_b: 1839-3677 (1839)
#   shard_c: 3678-5516 (1839)
#   shard_d: 5517-7354 (1838)

set -u

# ---- dependency guard -------------------------------------------------------
echo "=== Pre-flight checks ==="
python3 -c "import cadquery"      2>/dev/null || { echo "ABORT: cadquery not installed"; exit 1; }
python3 -c "import pytablewriter" 2>/dev/null || { echo "ABORT: pytablewriter not installed"; exit 1; }
echo "  Dependencies OK"

if [ -n "${LMMS_CADCODER_SKIP_IOU:-}" ]; then
    echo "ABORT: LMMS_CADCODER_SKIP_IOU set — would skip CAD IoU scoring."
    exit 1
fi
if [ -n "${LMMS_CADCODER_DRY_RUN:-}" ]; then
    echo "ABORT: LMMS_CADCODER_DRY_RUN set — would fake-zero CAD scoring."
    exit 1
fi
echo "  Reward-hacking guards: clear"

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
OUTPUT_BASE="./logs/cadcoder_shards_$(date +%Y%m%d_%H%M)"
GPU_MEMORY_UTILIZATION=0.50
TP=2
BATCH_SIZE=4

export LMMS_CADCODER_EXEC_TIMEOUT=120
export LMMS_CADCODER_TIMEOUT=300
unset LMMS_CADCODER_SKIP_IOU 2>/dev/null || true

mkdir -p "$OUTPUT_BASE"

echo "=============================================="
echo "  CAD-Coder 4-shard full evaluation"
echo "  Model : ${MODEL_PATH}"
echo "  Output: ${OUTPUT_BASE}"
echo "  Start : $(date)"
echo "=============================================="

run_shard () {
    local gpus="$1"
    local name="$2"
    local offset="$3"
    local limit="$4"
    local out="${OUTPUT_BASE}/${name}"
    local log="${OUTPUT_BASE}/${name}.log"
    local cache="${OUTPUT_BASE}/.vllm_cache_${name}"
    local triton_cache="${OUTPUT_BASE}/.triton_cache_${name}"
    mkdir -p "$out" "$cache" "$triton_cache"
    echo "[${name}] START GPUs=${gpus} offset=${offset} limit=${limit}  $(date)"
    CUDA_VISIBLE_DEVICES="${gpus}" \
    VLLM_CACHE_ROOT="${cache}" \
    TRITON_CACHE_DIR="${triton_cache}" \
    python -m lmms_eval \
        --model vllm \
        --model_args "model=${MODEL_PATH},tensor_parallel_size=${TP},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},max_model_len=8192,enforce_eager=True" \
        --tasks cad_coder_test \
        --offset "${offset}" \
        --limit "${limit}" \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${out}" \
        --log_samples --log_samples_suffix "${name}" \
        > "${log}" 2>&1
    local exit_code=$?
    echo "[${name}] DONE exit=${exit_code}  $(date)  -> ${log}"
}

# 4 parallel shards, one per GPU pair
run_shard "0,1" "shard_a"    0 1839 &
run_shard "2,3" "shard_b" 1839 1839 &
run_shard "4,5" "shard_c" 3678 1839 &
run_shard "6,7" "shard_d" 5517 1838 &
wait

echo ""
echo "=============================================="
echo "  All 4 shards finished: $(date)"
echo "=============================================="
echo "Results files:"
find "$OUTPUT_BASE" -name "*results*.json" -not -path "*/.vllm_cache*" -not -path "*/.triton_cache*" | sort
echo ""
echo "Run merge_cadcoder_shards.py on these to get the full-dataset metric."
