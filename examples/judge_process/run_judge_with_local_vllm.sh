#!/bin/bash
# Run judge with local vLLM server
#
# This script will:
# 1. Start a local vLLM server (if not already running)
# 2. Wait for the server to be ready
# 3. Run lmms-eval judge with the local LLM
# 4. Optionally stop the server after judging
#
# Usage:
#   bash run_judge_with_local_vllm.sh --input_result results.jsonl [options]
#
# Examples:
#   # Basic usage
#   bash run_judge_with_local_vllm.sh --input_result results/samples.jsonl
#
#   # Specify task
#   bash run_judge_with_local_vllm.sh -i results.jsonl -t mathvision_reason_testmini
#
#   # Judge multiple tasks from a directory
#   bash run_judge_with_local_vllm.sh -i results/ -t mathvision_test,wemath_testmini_reasoning -d judged/
#
#   # Use specific model
#   bash run_judge_with_local_vllm.sh -i results.jsonl --model Qwen2.5-VL-7B-Instruct
#
#   # High concurrency for batch processing
#   bash run_judge_with_local_vllm.sh -i "results/*.jsonl" --parallel 16

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default settings
DEFAULT_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_PORT=8000
DEFAULT_GPU_MEM=0.9
DEFAULT_MAX_MODEL_LEN=8192
DEFAULT_PARALLEL=8
DEFAULT_JUDGE_MODE="llm"

# Parse arguments
INPUT_RESULT=""
TASK="auto-detect"
OUTPUT_DIR=""
MODEL="$DEFAULT_MODEL"
PORT=$DEFAULT_PORT
GPU_MEM=$DEFAULT_GPU_MEM
MAX_MODEL_LEN=$DEFAULT_MAX_MODEL_LEN
PARALLEL=$DEFAULT_PARALLEL
JUDGE_MODE=$DEFAULT_JUDGE_MODE
KEEP_SERVER=false
DRY_RUN=false

# =============================================================================
# Helper Functions
# =============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") --input_result <path> [OPTIONS]

Start local vLLM server and run lmms-eval judge

Required Arguments:
  -i, --input_result <path>     Path to JSONL result file(s)

Optional Arguments:
  -t, --task <name>             Task name(s). Comma-separated for multiple tasks (default: auto-detect)
                                When multiple tasks are given, --input_result should be a directory.
  -d, --output-dir <dir>        Output directory for judged results
  -m, --model <name>            Model to use (default: $DEFAULT_MODEL)
  -p, --port <number>           vLLM server port (default: $DEFAULT_PORT)
  -j, --parallel <number>       Parallel workers (default: $DEFAULT_PARALLEL)
  --judge-mode <mode>           Judging mode: rule|llm|auto (default: llm)
  --gpu-mem <ratio>             GPU memory utilization (default: $DEFAULT_GPU_MEM)
  --max-model-len <number>      Max model length (default: $DEFAULT_MAX_MODEL_LEN)
  --keep-server                 Keep vLLM server running after judging
  --dry-run                     Show commands without executing
  -h, --help                    Show this help message

Examples:
  # Basic usage
  $(basename "$0") -i results/samples.jsonl

  # Specify single task
  $(basename "$0") -i results.jsonl -t mathvision_reason_testmini -d judged/

  # Judge multiple tasks from a directory
  $(basename "$0") -i results/ -t mathvision_test,wemath_reasoning -d judged/

  # Use different model
  $(basename "$0") -i results.jsonl --model meta-llama/Llama-3.1-8B-Instruct

  # Batch processing with high concurrency
  $(basename "$0") -i "results/*.jsonl" --parallel 16

Environment:
  The script will set these environment variables for the judge:
    JUDGE_BASE_URL=http://localhost:<port>/v1
    JUDGE_API_KEY=dummy
    JUDGE_MODEL=<model_name>
    JUDGE_MAX_CONCURRENT=<parallel>

EOF
}

log_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

log_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

log_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

log_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1"
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input_result)
            INPUT_RESULT="$2"
            shift 2
            ;;
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -j|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --judge-mode)
            JUDGE_MODE="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --keep-server)
            KEEP_SERVER=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_RESULT" ]]; then
    log_error "Missing required argument: --input_result"
    echo "Use -h or --help for usage information"
    exit 1
fi

# =============================================================================
# Check Dependencies
# =============================================================================

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        log_error "Python not found. Please install Python 3.8+."
        exit 1
    fi
    
    # Check vLLM
    if ! command -v vllm &> /dev/null; then
        log_error "vLLM not found. Please install: pip install vllm"
        exit 1
    fi
    
    # Check lmms-eval
    if ! python -c "import lmms_eval" 2>/dev/null; then
        log_error "lmms-eval not found. Please install from ${PROJECT_ROOT}"
        exit 1
    fi
    
    log_success "All dependencies found"
}

# =============================================================================
# vLLM Server Management
# =============================================================================

VLLM_PID=""

is_vllm_running() {
    local url="http://localhost:${PORT}/health"
    curl -s "$url" > /dev/null 2>&1
}

check_existing_vllm() {
    log_info "Checking for existing vLLM server on port ${PORT}..."
    
    if is_vllm_running; then
        log_success "Found existing vLLM server on port ${PORT}"
        return 0
    else
        log_info "No existing vLLM server found"
        return 1
    fi
}

start_vllm() {
    log_info "Starting vLLM server..."
    log_info "  Model: ${MODEL}"
    log_info "  Port: ${PORT}"
    log_info "  GPU Memory: ${GPU_MEM}"
    log_info "  Max Model Length: ${MAX_MODEL_LEN}"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would execute:"
        echo "  vllm serve ${MODEL} \\"
        echo "    --port ${PORT} \\"
        echo "    --gpu-memory-utilization ${GPU_MEM} \\"
        echo "    --max-model-len ${MAX_MODEL_LEN} \\"
        echo "    --dtype bfloat16 \\"
        echo "    --enable-prefix-caching &"
        return
    fi
    
    # Start vLLM in background
    vllm serve "${MODEL}" \
        --port "${PORT}" \
        --gpu-memory-utilization "${GPU_MEM}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --dtype bfloat16 \
        --enable-prefix-caching \
        > /tmp/vllm_server.log 2>&1 &
    
    VLLM_PID=$!
    log_info "vLLM server started with PID: ${VLLM_PID}"
}

wait_for_vllm() {
    log_info "Waiting for vLLM server to be ready..."
    
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if is_vllm_running; then
            log_success "vLLM server is ready!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 1
        
        # Check if process died
        if [[ -n "$VLLM_PID" ]] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo
            log_error "vLLM server process died. Check logs: /tmp/vllm_server.log"
            exit 1
        fi
    done
    
    echo
    log_error "Timeout waiting for vLLM server after ${max_attempts} seconds"
    log_error "Check logs: /tmp/vllm_server.log"
    exit 1
}

stop_vllm() {
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log_info "Stopping vLLM server (PID: ${VLLM_PID})..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        log_success "vLLM server stopped"
    fi
}

# =============================================================================
# Run Judge
# =============================================================================

run_judge() {
    log_info "Configuring environment for local judge..."
    
    export JUDGE_BASE_URL="http://localhost:${PORT}/v1"
    export JUDGE_API_KEY="dummy"
    export JUDGE_MODEL="${MODEL}"
    export JUDGE_MAX_CONCURRENT="${PARALLEL}"
    export JUDGE_MODE="${JUDGE_MODE}"
    
    log_info "Judge configuration:"
    log_info "  JUDGE_BASE_URL: ${JUDGE_BASE_URL}"
    log_info "  JUDGE_MODEL: ${JUDGE_MODEL}"
    log_info "  JUDGE_MODE: ${JUDGE_MODE}"
    log_info "  JUDGE_MAX_CONCURRENT: ${JUDGE_MAX_CONCURRENT}"
    
    # Build judge command
    local judge_cmd="python -m lmms_eval judge"
    judge_cmd="${judge_cmd} --input_result ${INPUT_RESULT}"
    judge_cmd="${judge_cmd} --task ${TASK}"
    judge_cmd="${judge_cmd} --judge-mode ${JUDGE_MODE}"
    judge_cmd="${judge_cmd} --parallel ${PARALLEL}"
    
    if [[ -n "$OUTPUT_DIR" ]]; then
        judge_cmd="${judge_cmd} --output-dir ${OUTPUT_DIR}"
    fi
    
    log_info "Running judge command:"
    echo "  ${judge_cmd}"
    echo
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Command would be executed above"
        return
    fi
    
    # Run judge
    eval "$judge_cmd"
    
    log_success "Judging completed!"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "================================================================================"
    echo "LMMS-Eval Judge with Local vLLM"
    echo "================================================================================"
    echo
    
    # Trap to cleanup vLLM on exit
    trap stop_vllm EXIT
    
    # Check dependencies
    check_dependencies
    
    echo
    echo "Configuration Summary:"
    echo "  Input: ${INPUT_RESULT}"
    echo "  Task: ${TASK}"
    echo "  Model: ${MODEL}"
    echo "  Port: ${PORT}"
    echo "  Parallel: ${PARALLEL}"
    echo "  Judge Mode: ${JUDGE_MODE}"
    [[ -n "$OUTPUT_DIR" ]] && echo "  Output Dir: ${OUTPUT_DIR}"
    [[ "$KEEP_SERVER" == true ]] && echo "  Keep Server: Yes"
    [[ "$DRY_RUN" == true ]] && echo "  Dry Run: Yes"
    echo
    
    # Check if vLLM is already running
    if ! check_existing_vllm; then
        start_vllm
        wait_for_vllm
    fi
    
    echo
    
    # Run judge
    run_judge
    
    echo
    echo "================================================================================"
    log_success "All done!"
    echo "================================================================================"
    
    # Handle --keep-server flag
    if [[ "$KEEP_SERVER" == true ]]; then
        log_info "Keeping vLLM server running (--keep-server was specified)"
        log_info "To stop it later, run: kill ${VLLM_PID}"
        # Don't kill on exit
        trap - EXIT
    fi
}

# Run main
main
