#!/bin/bash
# vllm_qwen3_vl_async_multi_gpu.sh
#
# 用法：bash <script> [config.json]
#
# 功能：
#   1. 在每台机器上启动若干 vLLM 主模型后端（每个后端独占 TP 张 GPU）
#   2. 启动 lmms-eval 评测（accelerate launch，多机多进程）

set -euo pipefail
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# ══════════════════════════════════════════════════════════════════════════════
# §0  读取 JSON 配置
# ══════════════════════════════════════════════════════════════════════════════
CONFIG="${1:-$(dirname "$0")/config.json}"
CMD_MODEL_PATH="${2:-}"
[[ ! -f "${CONFIG}" ]] && { echo "[ERROR] Config not found: ${CONFIG}"; exit 1; }
command -v jq &>/dev/null  || { echo "[ERROR] jq is required but not installed."; exit 1; }

cfg()     { jq -r "$1"       "${CONFIG}"; }
cfg_int() { jq -r "$1 // 0" "${CONFIG}"; }   # 字段不存在时返回 0

# ── 环境 ──────────────────────────────────────────────────────────────────────
export HF_HOME=$(cfg '.env.hf_home')
export HF_TOKEN=$(cfg '.env.hf_token')
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
unset  HF_DATASETS_OFFLINE                        # 防止离线模式拦截数据集下载

# 虚拟环境路径 (使用 .venv 替代 conda)
VENV_PATH=$(cfg '.env.venv_path')

# ── 日志 ──────────────────────────────────────────────────────────────────────
LOG_BASE=$(cfg '.log.dir')

# ── 分布式 ─────���──────────────────────────────────────────────────────────────
MASTER_ADDR=$(cfg     '.distributed.master_addr')
MASTER_PORT=$(cfg_int '.distributed.master_port')
WORLD_SIZE=$(cfg_int  '.distributed.world_size')
RANK=$(cfg_int        '.distributed.rank')

# ── 主模型 ────────────────────────────────────────────────────────────────────
MODEL_FROM_JSON=$(cfg                   '.model.path')
MODEL="${CMD_MODEL_PATH:-$MODEL_FROM_JSON}"
MODEL_TP=$(cfg_int            '.model.tp')
MODEL_MAX_MODEL_LEN=$(cfg_int '.model.max_model_len')
MODEL_GPU_MEM_UTIL=$(cfg      '.model.gpu_memory_utilization')
MODEL_MAX_NUM_SEQS=$(cfg_int  '.model.max_num_seqs')
MODEL_BASE_PORT=$(cfg_int     '.model.base_port')

# ── 评测 ──────────────────────────────────────────────────────────────────────
TASKS=$(cfg            '.eval.tasks')
OUTPUT_PATH_BASE=$(cfg '.eval.output_path')
MODEL_NAME=$(basename "${MODEL}")
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_PATH="${OUTPUT_PATH_BASE}/${MODEL_NAME}/${TIMESTAMP}"
CONCURRENCY=$(cfg_int '.eval.concurrency // 128')
VERBOSITY=$(cfg        '.eval.verbosity')
MAX_NEW_TOKENS=$(cfg_int '.eval.max_new_tokens')
MAX_PIXELS=$(cfg_int     '.eval.max_pixels')
GEN_KWARGS=$(cfg         '.eval.gen_kwargs // "max_new_tokens=32768"')

# ── Debug 模式 ────────────────────────────────────────────────────────────────
DEBUG=$(cfg '.eval.debug // false')
[[ "${DEBUG}" == "null" || -z "${DEBUG}" ]] && DEBUG="false"

# ══════════════════════════════════════════════════════════════════════════════
# §1  计算当前机器的角色与 GPU 分配
# ══════════════════════════════════════════════════════════════════════════════
LOCAL_GPU_NUM=$(nvidia-smi -L | wc -l)
NPROC_PER_NODE=${LOCAL_GPU_NUM}
NUM_MACHINES=$(( (WORLD_SIZE + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))
MACHINE_RANK=$(( RANK / NPROC_PER_NODE ))

MAIN_GPU_NUM=${LOCAL_GPU_NUM}
NUM_BACKENDS=$(( MAIN_GPU_NUM / MODEL_TP ))

# ══════════════════════════════════════════════════════════════════════════════
# §2  日志目录 & 启动摘要
# ══════════════════════════════════════════════════════════════════════════════
LOG_DIR="${LOG_BASE}/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOG_DIR}"

echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Config          : ${CONFIG}"
echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Rank            : ${RANK}/${WORLD_SIZE}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Local GPUs      : ${LOCAL_GPU_NUM}  main=${MAIN_GPU_NUM} (TP=${MODEL_TP}, backends=${NUM_BACKENDS})"
echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Log dir         : ${LOG_DIR}"
if [[ "${DEBUG}" == "true" ]]; then
    echo "[WARN][Machine ${MACHINE_RANK}/${NUM_MACHINES}] DEBUG mode    : ENABLED (vLLM backends will NOT be killed on exit)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# §3  激活虚拟环境 (.venv)
# ══════════════════════════════════════════════════════════════════════════════
if [[ -z "${VENV_PATH}" || "${VENV_PATH}" == "null" ]]; then
    # 默认使用脚本所在目录的 .venv
    VENV_PATH="$(dirname "$0")/.venv"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[ERROR] Virtual environment not found: ${VENV_PATH}"
    echo "[ERROR] Please ensure .venv exists or set correct venv_path in config.json"
    exit 1
fi

echo "[INFO][Machine ${MACHINE_RANK}] Activating virtual environment: ${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

# ══════════════════════════════════════════════════════════════════════════════
# §4  进程管理（cleanup on exit / signal）
# ══════════════════════════════════════════════════════════════════════════════
PIDS=()
cleanup() {
    trap - EXIT INT TERM
    
    # DEBUG 模式下不 kill 后端，方便调试
    if [[ "${DEBUG}" == "true" ]]; then
        echo "[INFO][Machine ${MACHINE_RANK}] DEBUG mode enabled, skipping vLLM cleanup."
        echo "[INFO][Machine ${MACHINE_RANK}] PIDs to keep running: ${PIDS[*]}"
        echo "[INFO][Machine ${MACHINE_RANK}] To manually stop: kill ${PIDS[*]}"
        return
    fi
    
    [[ ${#PIDS[@]} -eq 0 ]] && return
    echo "[INFO][Machine ${MACHINE_RANK}] Stopping vLLM instances (PIDs: ${PIDS[*]})..."
    for pid in "${PIDS[@]}"; do
        kill -9 $(pgrep -P "${pid}" 2>/dev/null) 2>/dev/null || true
        kill -9 "${pid}" 2>/dev/null || true
    done
    echo "[INFO][Machine ${MACHINE_RANK}] Done."
}
trap cleanup EXIT INT TERM

# ══════════════════════════════════════════════════════════════════════════════
# §5  启动主模型 vLLM 后端（每台机器均执行）
# ═══════════════════════════════════════════════════════════════════════════��══
BACKEND_URLS=""
for (( i=0; i<NUM_BACKENDS; i++ )); do
    PORT=$(( MODEL_BASE_PORT + i ))

    START_GPU=$(( i * MODEL_TP ))
    GPUS=""
    for (( g=START_GPU; g<START_GPU+MODEL_TP; g++ )); do
        GPUS="${GPUS}${g},"
    done
    GPUS=${GPUS%,}

    MODEL_LOG="${LOG_DIR}/vllm_model_rank${RANK}_port${PORT}.log"
    echo "[INFO][Machine ${MACHINE_RANK}] Starting model vLLM  GPUs=${GPUS}  port=${PORT}..."

    CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
        --model                  "${MODEL}" \
        --tensor-parallel-size   "${MODEL_TP}" \
        --max-model-len          "${MODEL_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${MODEL_GPU_MEM_UTIL}" \
        --max-num-seqs           "${MODEL_MAX_NUM_SEQS}" \
        --port                   "${PORT}" \
        --mm-encoder-tp-mode data \
        --trust-remote-code \
        --enable-prefix-caching \
        > "${MODEL_LOG}" 2>&1 &
    PIDS+=($!)
    BACKEND_URLS="${BACKEND_URLS}http://localhost:${PORT}/v1;"
done
BACKEND_URLS=${BACKEND_URLS%;}

# ══════════════════════════════════════════════════════════════════════════════
# §6  等待所有后端就绪
# ══════════════════════════════════════════════════════════════════════════════
check_http() { curl -s -o /dev/null -w "%{http_code}" "$1/models" 2>/dev/null; }

echo "[INFO][Machine ${MACHINE_RANK}] Waiting for all backends to be ready (timeout 30min)..."
IFS=';' read -ra URL_ARRAY <<< "${BACKEND_URLS}"
for url in "${URL_ARRAY[@]}"; do
    retries=0
    while [[ "$(check_http "${url}")" != "200" ]]; do
        sleep 5
        retries=$(( retries + 1 ))
        if (( retries >= 360 )); then
            echo "[ERROR] Timeout waiting for ${url}"
            exit 1
        fi
    done
    echo "[INFO][Machine ${MACHINE_RANK}] Ready: ${url}"
done

# ══════════════════════════════════════════════════════════════════════════════
# §7  启动 lmms-eval 评测
# ═══════════════════════════════════════════════���══════════════════════════════
mkdir -p "${OUTPUT_PATH}"
EVAL_LOG="${LOG_DIR}/lmms_eval_rank${RANK}.log"
echo "[INFO][Machine ${MACHINE_RANK}] Launching lmms-eval  tasks=${TASKS}  output=${OUTPUT_PATH}  log= ${EVAL_LOG}"

accelerate launch \
    --num_processes     "${NPROC_PER_NODE}" \
    --num_machines      "${NUM_MACHINES}" \
    --machine_rank      "${MACHINE_RANK}" \
    --main_process_ip   "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --mixed_precision   "no" \
    --dynamo_backend    "no" \
    -m lmms_eval \
    --model       openai \
    --model_args  "model_version=${MODEL},base_url=${BACKEND_URLS},api_key=EMPTY,num_concurrent=${CONCURRENCY},adaptive_concurrency=False,adaptive_max_concurrency=${CONCURRENCY},is_qwen3_vl=True,max_pixels=${MAX_PIXELS}" \
    --gen_kwargs  "${GEN_KWARGS}" \
    --tasks       "${TASKS}" \
    --batch_size  1 \
    --output_path "${OUTPUT_PATH}" \
    --verbosity   "${VERBOSITY}" \
    --log_samples \
    > "${EVAL_LOG}" 2>&1

echo "[INFO][Machine ${MACHINE_RANK}] Evaluation completed successfully."
