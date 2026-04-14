#!/bin/bash
# qwen3_vl_worker.sh
# Worker entrypoint: launches vLLM backends and runs lmms-eval.
# This script is intended to run directly on a single machine (local debug) or
# inside each DLC worker container.
#
# Usage:
#   bash scripts/qwen3_vl_worker.sh [config.json] [optional_model_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/eval_common.sh"

CONFIG="${1:-$(dirname "$0")/config_eval.json}"
CMD_MODEL_PATH="${2:-}"

# ── load config & prepare environment ─────────────────────────────────────────
load_config "${CONFIG}" "${CMD_MODEL_PATH}"
compute_resources
setup_logging
ensure_venv
setup_cleanup_trap

# ── launch vLLM backends, wait, run eval, cleanup via trap ────────────────────
launch_vllm_backends

# 在 vLLM 后端启动/预热期间，后台并行 staging 数据集缓存，充分利用等待时间
stage_datasets &
DATASET_STAGE_PID=$!

wait_for_backends

# 确保数据集拷贝完成后再进入 eval（若目录不存在则 stage_datasets 为空操作，wait 立即返回）
if [[ -n "${DATASET_STAGE_PID:-}" ]]; then
    wait "${DATASET_STAGE_PID}" 2>/dev/null || true
fi

run_lmms_eval
