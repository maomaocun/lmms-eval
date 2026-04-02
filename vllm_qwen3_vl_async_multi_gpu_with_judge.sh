#!/bin/bash
# vllm_qwen3_vl_async_multi_gpu_with_judge.sh
#
# 用法：bash <script> [config.json]
#
# 功能：
#   1. 在每台机器上启动若干 vLLM 主模型后端（每个后端独占 TP 张 GPU）
#   2. 在指定的 judge 机器（judge.machine_rank）上启动 judge.count 个 judge 后端
#      - 每个 judge 实例占用 judge.tp 张 GPU，依次分配到末尾 GPU
#      - 各实例端口：judge.port, judge.port+1, ..., judge.port+count-1
#      - 所有 judge URL 拼成分号分隔的 OPENAI_API_URL，供 lmms-eval 多后端 round-robin
#      - 其他机器直接将 OPENAI_API_URL 指向 judge 机器（judge.host:port...）
#      - judge.host 留空时自动推断为 master_addr（仅当 judge.machine_rank==0 时有效）
#   3. 启动 lmms-eval 评测（accelerate launch，多机多进程）

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

# ── Judge 模型（path 为空则不启用）────────────────────────────────────────────
JUDGE_MODEL=$(cfg             '.judge.path')
JUDGE_TP=$(cfg_int            '.judge.tp')
JUDGE_PORT=$(cfg_int          '.judge.port')
JUDGE_COUNT=$(cfg_int         '.judge.count')      # 同机启动的 judge 实例数；默认 1
JUDGE_MAX_MODEL_LEN=$(cfg_int '.judge.max_model_len')
JUDGE_GPU_MEM_UTIL=$(cfg      '.judge.gpu_memory_utilization')
JUDGE_MAX_NUM_SEQS=$(cfg_int  '.judge.max_num_seqs')
# judge.machine_rank：哪台机器（machine_rank）负责运行 judge；默认 0
JUDGE_MACHINE_RANK=$(cfg_int  '.judge.machine_rank')
# judge.host：judge 服务对其他机器可达的 IP/hostname
# 留空时：若 judge.machine_rank==0 则自动取 master_addr，否则必须手动填写
JUDGE_HOST=$(cfg              '.judge.host')

[[ "${JUDGE_COUNT}" -le 0 ]] && JUDGE_COUNT=1     # 未配置时默认 1 个实例

# ── 评测 ──────────────────────────────────────────────────────────────────────
TASKS=$(cfg            '.eval.tasks')
OUTPUT_PATH=$(cfg      '.eval.output_path')
CONCURRENCY=$(cfg_int  '.eval.concurrency')
VERBOSITY=$(cfg        '.eval.verbosity')
MAX_NEW_TOKENS=$(cfg_int '.eval.max_new_tokens')
MAX_PIXELS=$(cfg_int     '.eval.max_pixels')

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

# 判断当前机器是否负责运行 judge
JUDGE_IS_LOCAL=false
if [[ -n "${JUDGE_MODEL}" ]] && (( MACHINE_RANK == JUDGE_MACHINE_RANK )); then
    JUDGE_IS_LOCAL=true
fi

# 本机 judge 占用的 GPU 总数 = count × tp
JUDGE_TOTAL_TP=$(( JUDGE_COUNT * JUDGE_TP ))

if ${JUDGE_IS_LOCAL}; then
    if (( JUDGE_TOTAL_TP >= LOCAL_GPU_NUM )); then
        echo "[ERROR] judge.count(${JUDGE_COUNT}) × judge.tp(${JUDGE_TP}) = ${JUDGE_TOTAL_TP} >= local GPU count(${LOCAL_GPU_NUM}), not enough GPUs for main model."
        exit 1
    fi
    # 末尾 JUDGE_TOTAL_TP 张 GPU 留给所有 judge 实例
    JUDGE_START_GPU=$(( LOCAL_GPU_NUM - JUDGE_TOTAL_TP ))
    MAIN_GPU_NUM=$(( LOCAL_GPU_NUM - JUDGE_TOTAL_TP ))
else
    MAIN_GPU_NUM=${LOCAL_GPU_NUM}
fi

NUM_BACKENDS=$(( MAIN_GPU_NUM / MODEL_TP ))

# judge URL 列表（分号分隔，供 OPENAI_API_URL 多后端 round-robin）
JUDGE_URL_LIST=""
if [[ -n "${JUDGE_MODEL}" ]]; then
    if [[ -z "${JUDGE_HOST}" || "${JUDGE_HOST}" == "null" ]]; then
        if (( JUDGE_MACHINE_RANK == 0 )); then
            JUDGE_HOST="${MASTER_ADDR}"
        else
            echo "[ERROR] judge.host must be set when judge.machine_rank != 0."
            exit 1
        fi
    fi
    for (( j=0; j<JUDGE_COUNT; j++ )); do
        PORT=$(( JUDGE_PORT + j ))
        JUDGE_URL_LIST="${JUDGE_URL_LIST}http://${JUDGE_HOST}:${PORT}/v1;"
    done
    JUDGE_URL_LIST="${JUDGE_URL_LIST%;}"
fi

# ══════════════════════════════════════════════════════════════════════════════
# §2  日志目录 & 启动摘要
# ══════════════════════════════════════════════════════════════════════════════
LOG_DIR="${LOG_BASE}/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOG_DIR}"

echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Config          : ${CONFIG}"
echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Rank            : ${RANK}/${WORLD_SIZE}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Local GPUs      : ${LOCAL_GPU_NUM}  main=${MAIN_GPU_NUM} (TP=${MODEL_TP}, backends=${NUM_BACKENDS})"
if [[ -n "${JUDGE_MODEL}" ]]; then
    if ${JUDGE_IS_LOCAL}; then
        echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Judge           : LOCAL  count=${JUDGE_COUNT}  TP=${JUDGE_TP}  GPUs=$(( JUDGE_TOTAL_TP ))  ports=${JUDGE_PORT}..$(( JUDGE_PORT + JUDGE_COUNT - 1 ))"
    else
        echo "[INFO][Machine ${MACHINE_RANK}/${NUM_MACHINES}] Judge           : REMOTE  urls=${JUDGE_URL_LIST}  (on machine ${JUDGE_MACHINE_RANK})"
    fi
fi
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
# §5  启动 Judge vLLM（仅 judge 机器，支持多实例）
# ══════════════════════════════════════════════════════════════════════════════
if [[ -n "${JUDGE_MODEL}" ]]; then
    # 所有机器都需要设置这几个环境变量，供 lmms-eval 内部的 judge client 使用
    export OPENAI_API_URL="${JUDGE_URL_LIST}"
    export MODEL_VERSION="${JUDGE_MODEL}"
    export OPENAI_API_KEY="EMPTY"
    export API_TYPE="openai"
    # 供任务 utils（sfe/MolParse/OpenRxn 等）直接调用 judge 时使用
    FIRST_JUDGE_URL="${JUDGE_URL_LIST%%[;]*}"
    export OPENAI_BASE_URL="${FIRST_JUDGE_URL}"
    export OPENAI_MODEL_NAME="${JUDGE_MODEL}"

    if ${JUDGE_IS_LOCAL}; then
        for (( j=0; j<JUDGE_COUNT; j++ )); do
            PORT=$(( JUDGE_PORT + j ))
            START_GPU=$(( JUDGE_START_GPU + j * JUDGE_TP ))
            GPUS=""
            for (( g=START_GPU; g<START_GPU+JUDGE_TP; g++ )); do
                GPUS="${GPUS}${g},"
            done
            GPUS="${GPUS%,}"

            JUDGE_LOG="${LOG_DIR}/vllm_judge_rank${RANK}_port${PORT}.log"
            echo "[INFO][Machine ${MACHINE_RANK}] Starting judge vLLM [$(( j+1 ))/${JUDGE_COUNT}]  model=$(basename ${JUDGE_MODEL})  GPUs=${GPUS}  port=${PORT}..."

            CUDA_VISIBLE_DEVICES=${GPUS} python -m vllm.entrypoints.openai.api_server \
                --model                  "${JUDGE_MODEL}" \
                --tensor-parallel-size   "${JUDGE_TP}" \
                --max-model-len          "${JUDGE_MAX_MODEL_LEN}" \
                --gpu-memory-utilization "${JUDGE_GPU_MEM_UTIL}" \
                --max-num-seqs           "${JUDGE_MAX_NUM_SEQS}" \
                --port                   "${PORT}" \
                --trust-remote-code \
                --enable-prefix-caching \
                > "${JUDGE_LOG}" 2>&1 &
            PIDS+=($!)
        done
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# §6  启动主模型 vLLM 后端（每台机器均执行）
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
# §7  等待所有后端就绪（主模型 + judge）
# ══════════════════════════════════════════════════════════════════════════════
check_http() { curl -s -o /dev/null -w "%{http_code}" "$1/models" 2>/dev/null; }

# ���模型后端 + 所有 judge 后端（本机或远端，每台机器都等，无副作用）
WAIT_URLS="${BACKEND_URLS}"
[[ -n "${JUDGE_MODEL}" ]] && WAIT_URLS="${WAIT_URLS};${JUDGE_URL_LIST}"

echo "[INFO][Machine ${MACHINE_RANK}] Waiting for all backends to be ready (timeout 30min)..."
IFS=';' read -ra URL_ARRAY <<< "${WAIT_URLS}"
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
# §8  启动 lmms-eval 评测
# ═══════════════════════════════════════════════���══════════════════════════════
EVAL_LOG="${LOG_DIR}/lmms_eval_rank${RANK}.log"
echo "[INFO][Machine ${MACHINE_RANK}] Launching lmms-eval  tasks=${TASKS}  log= ${EVAL_LOG}"

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
    --model_args  "model_version=${MODEL},base_url=${BACKEND_URLS},api_key=EMPTY,num_concurrent=${CONCURRENCY},adaptive_concurrency=False,adaptive_max_concurrency=128,is_qwen3_vl=True,max_pixels=${MAX_PIXELS}" \
    --gen_kwargs  "max_new_tokens=${MAX_NEW_TOKENS}" \
    --tasks       "${TASKS}" \
    --batch_size  1 \
    --output_path "${OUTPUT_PATH}" \
    --verbosity   "${VERBOSITY}" \
    --log_samples \
    > "${EVAL_LOG}" 2>&1

echo "[INFO][Machine ${MACHINE_RANK}] Evaluation completed successfully."
