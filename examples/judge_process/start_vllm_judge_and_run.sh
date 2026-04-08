#!/bin/bash
# Start multiple local vLLM judge backends and run lmms-eval judge + aggregate
#
# Usage:
#   export JUDGE_RESULT_DIR=/mnt/cpfs/yangyicun/eval_result/model__Qwen3-VL-8B-Instruct
#   export JUDGE_OUTPUT_DIR=judged_results/
#   bash start_vllm_judge_and_run.sh
#
# This script will:
# 1. Start multiple vLLM judge backends (default 4 backends, each with tensor_parallel=2)
# 2. Run lmms-eval judge across them (per-sample evaluation)
# 3. Run lmms-eval aggregate for tasks that need special aggregation (e.g., WeMath)
#
# ============================================================================
#                        关于 Judge vs Aggregate
# ============================================================================
#
# Judge (评判):
#   - 对每道题进行评分
#   - 生成 per-sample 的结果（每道题的对错）
#   - 所有任务都需要这一步
#
# Aggregate (聚合):
#   - 把 per-sample 结果汇总成最终指标
#   - 大部分任务使用简单平均 (mean)，在 judge 阶段已完成
#   - 少数任务（如 WeMath）需要特殊的聚合逻辑（分析子问题间的关系）
#
# 哪些任务需要 Aggregate？
#   ✅ wemath_testmini_reasoning  - 需要（特殊的 Loose/Strict 分数计算）
#   ❌ mathvision_*_reasoning     - 不需要（使用简单平均）
#   ❌ mathverse_*_reasoning      - 不需要（使用简单平均）
#   ❌ mathvista_*_reasoning      - 不需要（使用简单平均）
#
# ============================================================================
#
# Environment variables (all optional):
#   JUDGE_RESULT_DIR        Directory containing result JSONL files (required)
#   JUDGE_OUTPUT_DIR        Output directory for judged results
#   JUDGE_NUM_BACKENDS      Number of vLLM judge backends (default: 4)
#   JUDGE_TP                Tensor parallel size per backend (default: 2)
#   JUDGE_BASE_PORT         Starting port for backends (default: 8000)
#   JUDGE_MODEL             Judge model (default: /mnt/cpfs/public_data/public_model/Qwen3-vl/Qwen3-VL-32B-Instruct)
#   JUDGE_GPU_MEM           GPU memory utilization (default: 0.9)
#   JUDGE_MAX_MODEL_LEN     Max model length (default: 8192)
#   JUDGE_PARALLEL          Parallel workers (default: 8)
#   JUDGE_MODE              rule|llm|auto (default: llm)
#   JUDGE_DEBUG             Keep vLLM running on failure (default: true)
#   JUDGE_KEEP_SERVER       Do not stop vLLM after judging (default: false)
#   JUDGE_SKIP_AGGREGATE    Skip aggregation step (default: false)

set -uo pipefail

# ── Tasks to Judge ───────────────────────────────────────────────────────────
# Override by exporting TASKS as a space-separated string before running.
# Example:
#   export TASKS="wemath_testmini_reasoning mathvision_reason_testmini_reasoning"
#
# 注意：以下任务需要 aggregate 步骤：
#   - wemath_testmini_reasoning (特殊的 Loose/Strict 分数计算)
#   - mmmu_val_qwen_judge (按学科分类聚合)
#   - mmmu_val_qwen3_official (按 dev/validation split 聚合，官方 Qwen3-VL 评估逻辑)
# 其他任务使用简单平均，judge 完成后即得到最终分数
_default_tasks=(
    # mathvision_reason_test_reasoning
    # mathvision_reason_testmini_reasoning
    # mathverse_testmini_reasoning
    # mathvista_testmini_cot_reasoning
    # wemath_testmini_reasoning
    # mmmu_val_qwen_judge
    mmmu_val_qwen3_official
    # mmmu_val_qwen3_official  # 官方 Qwen3-VL 评估（使用 GPT judge fallback）
    # mmmu_test
)
if [ -n "${TASKS:-}" ]; then
    read -ra TASKS <<< "$TASKS"
else
    TASKS=("${_default_tasks[@]}")
fi

# Uncomment to auto-discover all *samples_*.jsonl files instead:
# export AUTO_DISCOVER=true

# ── Configuration from Environment ───────────────────────────────────────────
RESULT_DIR="${JUDGE_RESULT_DIR:-/mnt/cpfs/yangyicun/eval_result/model__Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${JUDGE_OUTPUT_DIR:-/mnt/cpfs/yangyicun/judge_result}"
NUM_BACKENDS="${JUDGE_NUM_BACKENDS:-4}"
TP="${JUDGE_TP:-2}"
BASE_PORT="${JUDGE_BASE_PORT:-8000}"
MODEL="${JUDGE_MODEL:-/mnt/cpfs/public_data/public_model/Qwen3-vl/Qwen3-VL-32B-Instruct}"
GPU_MEM="${JUDGE_GPU_MEM:-0.9}"
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
PARALLEL="${JUDGE_PARALLEL:-128}"
JUDGE_MODE="${JUDGE_MODE:-llm}"
DEBUG="${JUDGE_DEBUG:-true}"
KEEP_SERVER="${JUDGE_KEEP_SERVER:-false}"
SKIP_AGGREGATE="${JUDGE_SKIP_AGGREGATE:-false}"

# ── Helper Functions ─────────────────────────────────────────────────────────
log_info()  { echo -e "\033[0;34m[INFO]\033[0m  $1"; }
log_ok()    { echo -e "\033[0;32m[OK]\033[0m    $1"; }
log_warn()  { echo -e "\033[0;33m[WARN]\033[0m  $1"; }
log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

is_vllm_running_on_port() {
    local p="$1"
    curl -s "http://localhost:${p}/health" >/dev/null 2>&1
}

# Check if a task requires special aggregation
# Returns 0 if task needs aggregate, 1 otherwise
needs_special_aggregation() {
    local task="$1"
    # Tasks requiring special/category-level aggregation
    if [[ "$task" == *"wemath"* ]] || [[ "$task" == *"mmmu_val"* ]] || [[ "$task" == *"mmmu_test"* ]] || [[ "$task" == *"mmmu_val_qwen3_official"* ]]; then
        return 0  # true - needs aggregate
    else
        return 1  # false - doesn't need aggregate
    fi
}

# Check if a task uses official Qwen3 evaluation with GPT judge fallback
# These tasks need the judge model to be properly configured
is_official_qwen3_task() {
    local task="$1"
    if [[ "$task" == *"mmmu_val_qwen3_official"* ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# ── Activate Virtual Environment ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)/.venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    log_info "Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
else
    log_warn "Virtual environment not found at ${VENV_PATH}, assuming already activated"
fi

# ── Check Input ──────────────────────────────────────────────────────────────
# Trim leading/trailing whitespace from paths
RESULT_DIR="$(echo "$RESULT_DIR" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
OUTPUT_DIR="$(echo "$OUTPUT_DIR" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"

if [ -z "$RESULT_DIR" ]; then
    log_error "JUDGE_RESULT_DIR is not set. Please export it before running this script."
    exit 1
fi
RESULT_DIR="${RESULT_DIR%/}"

if [ ! -d "$RESULT_DIR" ]; then
    log_error "Result directory not found: '$RESULT_DIR'"
    exit 1
fi

# ── Discover files to judge ─────────────────────────────────────────────────
JUDGE_FILES=()
JUDGE_TASKS=()

if [[ "${AUTO_DISCOVER:-false}" == true ]]; then
    for f in "$RESULT_DIR"/*samples_*.jsonl; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        task=$(echo "$fname" | sed 's/.*samples_//' | sed 's/\.jsonl$//')
        JUDGE_FILES+=("$f")
        JUDGE_TASKS+=("$task")
    done
else
    for task in "${TASKS[@]}"; do
        pattern="${RESULT_DIR}"/*samples_${task}.jsonl
        # shellcheck disable=SC2086
        matches=($pattern)
        if [ ! -f "${matches[0]}" ]; then
            log_warn "No file found for task: $task"
            continue
        fi
        latest=$(ls -t ${pattern} 2>/dev/null | head -n1)
        JUDGE_FILES+=("$latest")
        JUDGE_TASKS+=("$task")
    done
fi

if [ ${#JUDGE_FILES[@]} -eq 0 ]; then
    log_error "No JSONL files found to judge in $RESULT_DIR"
    exit 1
fi

log_info "Found ${#JUDGE_FILES[@]} task(s) to judge:"
for i in "${!JUDGE_FILES[@]}"; do
    task="${JUDGE_TASKS[$i]}"
    if needs_special_aggregation "$task"; then
        echo "  [${task}] -> ${JUDGE_FILES[$i]} (需要 aggregate)"
    else
        echo "  [${task}] -> ${JUDGE_FILES[$i]} (judge 后直接出分)"
    fi
done
echo ""

# ── GPU Assignment ───────────────────────────────────────────────────────────
TOTAL_JUDGE_GPUS=$(( NUM_BACKENDS * TP ))
LOCAL_GPU_NUM=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$LOCAL_GPU_NUM" -lt "$TOTAL_JUDGE_GPUS" ]; then
    log_error "Not enough GPUs: need ${TOTAL_JUDGE_GPUS} for ${NUM_BACKENDS} backends (TP=${TP}), but only ${LOCAL_GPU_NUM} available"
    exit 1
fi
JUDGE_START_GPU=$(( LOCAL_GPU_NUM - TOTAL_JUDGE_GPUS ))
log_info "GPU allocation: total=${LOCAL_GPU_NUM}, judge GPUs=${TOTAL_JUDGE_GPUS} (backends=${NUM_BACKENDS} x TP=${TP}), starting at GPU ${JUDGE_START_GPU}"

# Create timestamped log directory matching existing vllm_logs style
VLLM_LOG_DIR="/mnt/cpfs/yangyicun/vllm_logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$VLLM_LOG_DIR"
log_info "vLLM logs will be saved to: $VLLM_LOG_DIR"

# ── vLLM Backend Management ──────────────────────────────────────────────────
VLLM_PIDS=()

start_vllm_backend() {
    local idx="$1"
    local port="$2"
    local start_gpu="$3"
    local log_file="${VLLM_LOG_DIR}/vllm_judge_${port}.log"

    local gpus=""
    for (( g=start_gpu; g<start_gpu+TP; g++ )); do
        gpus="${gpus}${g},"
    done
    gpus="${gpus%,}"

    log_info "Starting judge backend [$((idx+1))/${NUM_BACKENDS}] on GPUs=${gpus}, port=${port}"
    log_info "  Log: $log_file"

    CUDA_VISIBLE_DEVICES=${gpus} python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --tensor-parallel-size "$TP" \
        --gpu-memory-utilization "$GPU_MEM" \
        --max-model-len "$MAX_MODEL_LEN" \
        --port "$port" \
        --max-num-seqs 512 \
        --dtype bfloat16 \
        --enable-prefix-caching \
        --trust-remote-code \
        > "$log_file" 2>&1 &

    VLLM_PIDS+=($!)
}

cleanup_vllm() {
    # Remove traps to prevent re-entry
    trap - EXIT INT TERM
    if [[ "$DEBUG" == true ]]; then
        log_info "DEBUG=true, leaving vLLM backends running (PIDs: ${VLLM_PIDS[*]:-N/A})"
        return
    fi
    if [[ "$KEEP_SERVER" == true ]]; then
        log_info "JUDGE_KEEP_SERVER=true, leaving vLLM backends running (PIDs: ${VLLM_PIDS[*]:-N/A})"
        return
    fi
    if [ ${#VLLM_PIDS[@]} -eq 0 ]; then
        return
    fi
    log_info "Stopping vLLM judge backends (PIDs: ${VLLM_PIDS[*]})..."
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    log_ok "All vLLM backends stopped"
}
# trap cleanup_vllm EXIT INT TERM

# Check existing backends and start missing ones
NEED_START=()
PORTS=()
for (( i=0; i<NUM_BACKENDS; i++ )); do
    port=$(( BASE_PORT + i ))
    PORTS+=("$port")
    if is_vllm_running_on_port "$port"; then
        log_ok "vLLM is already running on port $port"
    else
        NEED_START+=("$i")
    fi
done

if ! command -v vllm &>/dev/null; then
    log_error "vllm command not found. Please install: pip install vllm"
    exit 1
fi

# Start all missing backends in parallel
for idx in "${NEED_START[@]}"; do
    port=${PORTS[$idx]}
    start_gpu=$(( JUDGE_START_GPU + idx * TP ))
    start_vllm_backend "$idx" "$port" "$start_gpu"
done

if [ ${#NEED_START[@]} -gt 0 ]; then
    log_info "Waiting for all backends to be ready (timeout 30min)..."
    for port in "${PORTS[@]}"; do
        retries=0
        while ! is_vllm_running_on_port "$port"; do
            sleep 5
            retries=$((retries + 1))
            if (( retries >= 360 )); then
                log_error "Timeout waiting for backend on port $port"
                log_info "Last 20 lines of log:"
                tail -n 20 "${VLLM_LOG_DIR}/vllm_judge_${port}.log" || true
                exit 1
            fi
        done
        log_ok "Backend ready on port $port"
    done
fi

# ── Build Judge Base URL List ────────────────────────────────────────────────
JUDGE_BASE_URL=""
for port in "${PORTS[@]}"; do
    JUDGE_BASE_URL="${JUDGE_BASE_URL}http://localhost:${port}/v1;"
done
JUDGE_BASE_URL="${JUDGE_BASE_URL%;}"

log_info "Judge backends: $JUDGE_BASE_URL"

# ── Run Judge ────────────────────────────────────────────────────────────────
export JUDGE_BASE_URL="$JUDGE_BASE_URL"
export JUDGE_API_KEY="dummy"
export JUDGE_MODEL="$MODEL"
export JUDGE_MAX_CONCURRENT="$PARALLEL"
export JUDGE_MODE="$JUDGE_MODE"

TOTAL=${#JUDGE_FILES[@]}
SUCCESS=0
FAILED=0

if [ "$TOTAL" -eq 0 ]; then
    log_error "No files to judge."
    exit 1
fi

# Build comma-separated task list (same as evaluation framework's --tasks)
TASK_ARG=$(IFS=,; echo "${JUDGE_TASKS[*]}")

# Use RESULT_DIR as input when multiple tasks so Python can resolve files
if [ "$TOTAL" -eq 1 ]; then
    INPUT_ARG="${JUDGE_FILES[0]}"
else
    INPUT_ARG="$RESULT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="/tmp/lmms_eval_judge_multi_$(date +%Y%m%d_%H%M%S).log"

echo "=================================================="
log_info "Step 1/2: Judging ${TOTAL} task(s): ${TASK_ARG}"
log_info "Input: ${INPUT_ARG}"
log_info "Output: ${OUTPUT_DIR}"
echo "=================================================="

# For mmmu_val_qwen3_official, use rule mode to trigger official evaluation logic
# This ensures the custom process_results function is called (rule-based + GPT judge fallback)
if [[ "${TASK_ARG}" == *"mmmu_val_qwen3_official"* ]]; then
    JUDGE_MODE="rule"
    log_info "Detected mmmu_val_qwen3_official task, using judge-mode=rule to enable official evaluation logic"
fi

JUDGE_CMD="python -u -m lmms_eval judge --input_result ${INPUT_ARG} --task ${TASK_ARG} --judge-mode ${JUDGE_MODE} --parallel ${PARALLEL} --output-dir ${OUTPUT_DIR}"
log_info "Running: $JUDGE_CMD"
log_info "Judge log file: $LOG_FILE"
echo ""

if eval "$JUDGE_CMD" 2>&1 | tee "$LOG_FILE"; then
    log_ok "All tasks judged successfully"
    SUCCESS=$TOTAL
else
    log_error "Judging failed"
    log_info "Last 20 lines of log:"
    tail -n 20 "$LOG_FILE" || true
    FAILED=1
fi

# ── Aggregate Results ────────────────────────────────────────────────────────
# After judging, run aggregation for tasks that need special aggregation logic.
# 
# 需要 Aggregate 的任务：
#   - wemath_testmini_reasoning: 需要分析子问题间的关系（泛化不足、死记硬背等）
#   - mmmu_val_qwen3_official: 需要按 dev/validation split 聚合（官方实现方式）
#
# 不需要 Aggregate 的任务（使用简单平均，judge 已计算）：
#   - mathvision_*_reasoning: aggregation=mean
#   - mathverse_*_reasoning: aggregation=mean  
#   - mathvista_*_reasoning: aggregation=mean

if [ $FAILED -eq 0 ] && [ "${SKIP_AGGREGATE}" != "true" ]; then
    echo ""
    echo "=================================================="
    log_info "Step 2/2: Aggregating results (only for tasks needing special aggregation)"
    echo "=================================================="
    
    # Track aggregation success/failure per task
    AGG_SUCCESS=0
    AGG_SKIPPED=0
    AGG_FAILED=0
    
    for task in "${JUDGE_TASKS[@]}"; do
        # Check if this task needs special aggregation
        if ! needs_special_aggregation "$task"; then
            log_info "Skipping aggregation for ${task} (使用简单平均，judge 结果已包含最终分数)"
            AGG_SKIPPED=$((AGG_SKIPPED + 1))
            continue
        fi
        
        # Find the judged output file for this task
        judged_pattern="${OUTPUT_DIR}"/*samples_${task}.jsonl
        judged_files=($judged_pattern)
        
        if [ ! -f "${judged_files[0]}" ]; then
            log_warn "No judged file found for task: $task"
            AGG_FAILED=$((AGG_FAILED + 1))
            continue
        fi
        
        # Get the latest judged file
        judged_file=$(ls -t ${judged_pattern} 2>/dev/null | head -n1)
        
        # Output file for aggregation results
        agg_output="${OUTPUT_DIR}/${task}_aggregated_results.json"
        
        log_info "Aggregating: $task"
        log_info "  Input: $judged_file"
        log_info "  Output: $agg_output"
        
        AGG_CMD="python -u -m lmms_eval aggregate --input ${judged_file} --task ${task} --output ${agg_output}"
        
        if eval "$AGG_CMD"; then
            log_ok "Aggregation successful: $task"
            AGG_SUCCESS=$((AGG_SUCCESS + 1))
        else
            log_error "Aggregation failed: $task"
            AGG_FAILED=$((AGG_FAILED + 1))
        fi
        echo ""
    done
    
    echo "=================================================="
    log_info "Aggregation Summary: $AGG_SUCCESS succeeded, $AGG_SKIPPED skipped (不需要), $AGG_FAILED failed"
    echo "=================================================="
fi

echo ""
echo "=================================================="
if [ $FAILED -eq 0 ]; then
    log_info "✅ 全部完成: $SUCCESS 个任务 judged"
    if [ "${SKIP_AGGREGATE}" != "true" ]; then
        log_info "   - $AGG_SUCCESS 个任务需要并完成了 aggregate"
        log_info "   - $AGG_SKIPPED 个任务不需要 aggregate (judge 结果已包含最终分数)"
    fi
else
    log_error "❌ 执行失败: $FAILED 个任务出错"
fi
echo "=================================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi
