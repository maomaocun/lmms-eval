#!/bin/bash
# qwen3_vl_submit.sh
# Submitter entrypoint: reads a DLC config + an eval config, then submits a DLC PyTorchJob.
# The worker containers will execute qwen3_vl_worker.sh with the eval config (or a runtime variant).
#
# Usage:
#   bash scripts/qwen3_vl_submit.sh <dlc_config.json> <eval_config.json>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DLC_CONFIG="${1:-$(dirname "$0")/config_dlc.json}"
EVAL_CONFIG="${2:-$(dirname "$0")/config_eval.json}"

for f in "${DLC_CONFIG}" "${EVAL_CONFIG}"; do
    if [[ ! -f "$f" ]]; then
        echo "[ERROR] Config not found: $f"
        echo "[ERROR] Usage: bash $(basename "$0") <dlc_config.json> <eval_config.json>"
        exit 1
    fi
done

if ! command -v jq &>/dev/null; then
    echo "[WARN] jq not found, attempting to install..."
    apt-get update -qq && apt-get install -y -qq jq || { echo "[ERROR] Failed to install jq."; exit 1; }
fi

# ── helpers for reading the two configs ───────────────────────────────────────
dlc_cfg()     { jq -r "$1"       "${DLC_CONFIG}"; }
dlc_cfg_int() { jq -r "$1 // 0" "${DLC_CONFIG}"; }
eval_cfg()     { jq -r "$1"       "${EVAL_CONFIG}"; }
eval_cfg_int() { jq -r "$1 // 0" "${EVAL_CONFIG}"; }

# ── validate DLC binary ───────────────────────────────────────────────────────
DLC_BINARY=$(dlc_cfg '.dlc.binary')
if [[ -z "${DLC_BINARY}" || "${DLC_BINARY}" == "null" ]]; then
    echo "[ERROR] DLC binary not configured in ${DLC_CONFIG} (dlc.binary)"
    exit 1
fi
if [[ ! -x "${DLC_BINARY}" ]]; then
    echo "[ERROR] DLC binary not found or not executable: ${DLC_BINARY}"
    exit 1
fi

# ── resolve job name (needs model info from eval config) ──────────────────────
MODEL=$(eval_cfg '.model.path')
MODEL_TP=$(eval_cfg_int '.model.tp')
LOG_BASE=$(eval_cfg '.log.dir')

JOB_NAME_FROM_CFG=$(dlc_cfg '.dlc.job_name // ""')
if [[ -n "${JOB_NAME_FROM_CFG}" && "${JOB_NAME_FROM_CFG}" != "null" ]]; then
    JOB_NAME="${JOB_NAME_FROM_CFG}"
else
    JOB_NAME="eval_$(basename ${MODEL})_tp${MODEL_TP}_$(date +%m%d_%H%M%S)"
fi

# 统一时间戳由 submitter 生成，保证所有 worker 目录一致
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
FIXED_LOG_DIR="${LOG_BASE}/${JOB_NAME}/${TIMESTAMP}"
mkdir -p "${FIXED_LOG_DIR}"

# ── generate runtime config for workers ───────────────────────────────────────
# Workers should never submit again, and cluster runs are always non-debug.
# 同时将统一时间戳写入 runtime config，worker 会用它作为 output_path
RUNTIME_CONFIG="${FIXED_LOG_DIR}/runtime_config.json"
jq --arg ts "${TIMESTAMP}" '.dlc.submit = false | .eval.debug = false | .eval.timestamp = $ts' "${EVAL_CONFIG}" > "${RUNTIME_CONFIG}"

# ── resolve absolute paths for worker script and runtime config ───────────────
WORKER_SCRIPT="${SCRIPT_DIR}/qwen3_vl_worker.sh"
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
    echo "[ERROR] Worker script not found: ${WORKER_SCRIPT}"
    exit 1
fi

# ── read DLC parameters ───────────────────────────────────────────────────────
WORKERS=$(dlc_cfg_int '.dlc.workers')
WORKER_GPU=$(dlc_cfg_int '.dlc.worker_gpu')
WORKER_CPU=$(dlc_cfg_int '.dlc.worker_cpu')
WORKER_MEMORY=$(dlc_cfg '.dlc.worker_memory')
WORKER_SHARED_MEMORY=$(dlc_cfg '.dlc.worker_shared_memory')
PRIORITY=$(dlc_cfg_int '.dlc.priority')
RUNNING_TIMEOUT=$(dlc_cfg_int '.dlc.running_timeout')
WORKER_IMAGE=$(dlc_cfg '.dlc.worker_image')
DATA_SOURCE_URIS=$(dlc_cfg '.dlc.data_source_uris')
RESOURCE_ID=$(dlc_cfg '.dlc.resource_id')
WORKSPACE_ID=$(dlc_cfg '.dlc.workspace_id')
VPC_ID=$(dlc_cfg '.dlc.vpc_id')
SWITCH_ID=$(dlc_cfg '.dlc.switch_id')
SECURITY_GROUP_ID=$(dlc_cfg '.dlc.security_group_id')
EXTENDED_CIDRS=$(dlc_cfg '.dlc.extended_cidrs')

# ── build DLC command ─────────────────────────────────────────────────────────
COMMAND="export LMMS_EVAL_LOG_DIR=${FIXED_LOG_DIR}; export LMMS_EVAL_STAGE_DATASETS=1; bash ${WORKER_SCRIPT} ${RUNTIME_CONFIG}"

# ── submit ────────────────────────────────────────────────────────────────────
echo "[INFO] Safety override for cluster run: debug=false"
echo "[INFO] Submitting DLC job: ${JOB_NAME}"
"${DLC_BINARY}" submit pytorchjob \
    --name="${JOB_NAME}" \
    --priority="${PRIORITY}" \
    --workers="${WORKERS}" \
    --worker_cpu="${WORKER_CPU}" \
    --worker_gpu="${WORKER_GPU}" \
    --worker_memory="${WORKER_MEMORY}" \
    --worker_shared_memory="${WORKER_SHARED_MEMORY}" \
    --worker_image="${WORKER_IMAGE}" \
    --data_source_uris="${DATA_SOURCE_URIS}" \
    --resource_id="${RESOURCE_ID}" \
    --workspace_id="${WORKSPACE_ID}" \
    --vpc_id="${VPC_ID}" \
    --switch_id="${SWITCH_ID}" \
    --security_group_id="${SECURITY_GROUP_ID}" \
    --extended_cidrs="${EXTENDED_CIDRS}" \
    --command="${COMMAND}"

echo "[INFO] Job submitted successfully."
echo "[INFO] Expected log locations:"
echo "  - vLLM logs: ${FIXED_LOG_DIR}"
echo "  - eval log:  ${FIXED_LOG_DIR}"
echo "[INFO] Unified timestamp: ${TIMESTAMP}"
