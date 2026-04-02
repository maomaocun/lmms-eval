#!/bin/bash
# Quick start script for local vLLM judge
#
# This is a simplified version for quick testing.
# For full options, use: run_judge_with_local_vllm.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_results.jsonl> [model_name]"
    echo
    echo "Examples:"
    echo "  $0 results/samples.jsonl"
    echo "  $0 results/samples.jsonl Qwen/Qwen2.5-VL-7B-Instruct"
    exit 1
fi

INPUT_FILE="$1"
MODEL="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT=8000

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    # Try wildcard
    if ! ls $INPUT_FILE 1>/dev/null 2>&1; then
        error "File not found: $INPUT_FILE"
        exit 1
    fi
fi

# Check if vLLM is running
log "Checking if vLLM is running on port $PORT..."
if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
    success "vLLM is already running!"
    VLLM_PID=""
else
    log "vLLM not running. Starting..."
    
    # Check if vllm command exists
    if ! command -v vllm &> /dev/null; then
        error "vLLM not found. Please install: pip install vllm"
        exit 1
    fi
    
    # Start vLLM
    log "Starting vLLM with model: $MODEL"
    vllm serve "$MODEL" \
        --port $PORT \
        --gpu-memory-utilization 0.9 \
        --max-model-len 8192 \
        --dtype bfloat16 \
        > /tmp/vllm_quickstart.log 2>&1 &
    
    VLLM_PID=$!
    log "vLLM started with PID: $VLLM_PID"
    
    # Wait for ready
    log "Waiting for vLLM to be ready..."
    for i in {1..60}; do
        if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
            success "vLLM is ready!"
            break
        fi
        echo -n "."
        sleep 1
        
        # Check if process died
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo
            error "vLLM process died. Check logs: /tmp/vllm_quickstart.log"
            exit 1
        fi
    done
    
    # Check if we timed out
    if ! curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
        echo
        error "Timeout waiting for vLLM"
        exit 1
    fi
fi

# Set environment
export JUDGE_BASE_URL="http://localhost:$PORT/v1"
export JUDGE_API_KEY="dummy"
export JUDGE_MODEL="$MODEL"
export JUDGE_MODE="llm"

# Auto-detect task from filename
FILENAME=$(basename "$INPUT_FILE")
if echo "$FILENAME" | grep -q "samples_"; then
    TASK=$(echo "$FILENAME" | sed 's/.*samples_//' | sed 's/\.jsonl$//')
    log "Auto-detected task: $TASK"
else
    TASK="auto-detect"
fi

# Run judge
log "Running judge..."
echo "  Input: $INPUT_FILE"
echo "  Task: $TASK"
echo "  Model: $MODEL"
echo

python -m lmms_eval judge \
    --input_result "$INPUT_FILE" \
    --task "$TASK" \
    --judge-mode llm \
    --parallel 8

success "Judging completed!"

# Cleanup
if [ -n "$VLLM_PID" ]; then
    log "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    success "vLLM stopped"
fi

echo
success "All done! Results saved to judged/ directory"
