#!/bin/bash
# Run all judge process examples
#
# Usage:
#   bash run_all_examples.sh                    # Run with default settings
#   bash run_all_examples.sh --with-llm-judge   # Run with LLM judge (requires env vars)
#
# Environment Configuration:
#   This script will automatically source a local config file if it exists:
#   - .env.judge (in the same directory as this script)
#
#   You can create this file with your API keys:
#
#   cat > .env.judge << 'EOF'
#   # Judge Model Configuration
#   export JUDGE_API_KEY="sk-your-api-key-here"
#   export JUDGE_MODEL="gpt-4o-mini"
#   export JUDGE_BASE_URL="https://api.openai.com/v1"
#
#   # Optional Settings
#   export JUDGE_MODE="auto"
#   export JUDGE_MAX_CONCURRENT="8"
#   EOF
#
#   NOTE: .env.judge is gitignored by default and should NOT be committed!

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

# =============================================================================
# Load Environment Configuration
# =============================================================================

# Source local environment file if it exists
ENV_FILE="${SCRIPT_DIR}/.env.judge"
if [ -f "$ENV_FILE" ]; then
    echo "Loading environment from: $ENV_FILE"
    source "$ENV_FILE"
    echo "✓ Environment loaded"
    echo ""
elif [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "Loading environment from: ${SCRIPT_DIR}/.env"
    source "${SCRIPT_DIR}/.env"
    echo "✓ Environment loaded"
    echo ""
fi

# =============================================================================
# Check Dependencies
# =============================================================================

echo "========================================"
echo "LMMS-Eval Judge Examples"
echo "========================================"
echo ""
echo "Python: ${PYTHON}"
echo ""

# Check Python exists
if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at ${PYTHON}"
    echo "Please ensure the virtual environment is set up:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  source .venv/bin/activate"
    exit 1
fi

# Check if we should run LLM judge examples
RUN_LLM_JUDGE=false
if [ "$1" == "--with-llm-judge" ] || [ -n "$JUDGE_API_KEY" ]; then
    RUN_LLM_JUDGE=true
fi

# Check LLM judge configuration
if [ "$RUN_LLM_JUDGE" = true ]; then
    if [ -z "$JUDGE_API_KEY" ]; then
        echo "⚠ Warning: JUDGE_API_KEY not set"
        echo ""
        echo "To enable LLM judge examples, create ${ENV_FILE} with:"
        echo "  export JUDGE_API_KEY='sk-your-api-key'"
        echo ""
        echo "Or run with the config file:"
        echo "  source ${ENV_FILE} && bash $0"
        echo ""
        echo "Continuing with rule-based examples only..."
        echo ""
        RUN_LLM_JUDGE=false
    else
        echo "✓ LLM judge configured"
        echo "  Model: ${JUDGE_MODEL:-gpt-4o-mini}"
        echo "  Mode: ${JUDGE_MODE:-auto}"
        echo ""
    fi
fi

# =============================================================================
# Run Examples
# =============================================================================

# Change to script directory
cd "${SCRIPT_DIR}"

# Example 1: Rule-based Judging (always runs)
echo "Running Example 1: Rule-based Judging"
echo "----------------------------------------"
"${PYTHON}" 01_rule_based_judging.py
echo ""

# Example 2: LLM-as-Judge (only if configured)
if [ "$RUN_LLM_JUDGE" = true ]; then
    echo "Running Example 2: LLM-as-Judge"
    echo "----------------------------------------"
    "${PYTHON}" 02_llm_judge.py
    echo ""
else
    echo "Skipping Example 2: LLM-as-Judge (configure JUDGE_API_KEY to enable)"
    echo "----------------------------------------"
    echo ""
fi

# Example 3: Batch Processing
echo "Running Example 3: Batch Processing"
echo "----------------------------------------"
"${PYTHON}" 03_batch_processing.py
echo ""

# Example 4: Custom Judge Logic
echo "Running Example 4: Custom Judge Logic"
echo "----------------------------------------"
"${PYTHON}" 04_custom_judge.py
echo ""

# Example 5: Model Comparison
echo "Running Example 5: Model Comparison"
echo "----------------------------------------"
"${PYTHON}" 05_compare_models.py
echo ""

echo "Running Example 6: Local vLLM Judge"
echo "----------------------------------------"
"${PYTHON}" 06_local_vllm_judge.py
echo ""

echo "========================================"
echo "All examples completed!"
echo "========================================"
echo ""

# =============================================================================
# Configuration Help
# =============================================================================

if [ "$RUN_LLM_JUDGE" = false ] && [ -z "$JUDGE_API_KEY" ]; then
    echo "💡 Tip: To run LLM judge examples, create a config file:"
    echo ""
    echo "  cat > ${ENV_FILE} << 'EOF'"
    echo "  # Judge Model API Configuration"
    echo "  export JUDGE_API_KEY='sk-your-openai-api-key'"
    echo "  export JUDGE_MODEL='gpt-4o-mini'"
    echo "  export JUDGE_BASE_URL='https://api.openai.com/v1'"
    echo "  EOF"
    echo ""
    echo "  # Then run again:"
    echo "  bash $0"
    echo ""
    echo "Note: ${ENV_FILE} is automatically loaded if it exists."
    echo "      This file should NOT be committed to git (add to .gitignore)."
    echo ""
fi
