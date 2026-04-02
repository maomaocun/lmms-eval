#!/bin/bash
# Test runner for llm_judge module
#
# Usage:
#   cd /mnt/cpfs/yangyicun/lmms-eval
#   bash test/llm_judge/run_tests.sh
#   bash test/llm_judge/run_tests.sh -v  # verbose mode
#   bash test/llm_judge/run_tests.sh -k test_name  # run specific test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv"

echo "========================================="
echo "Running llm_judge tests"
echo "========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Virtual env:  ${VENV_PATH}"
echo ""

# Check virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi

# Activate virtual environment
source "${VENV_PATH}/bin/activate"

# Verify Python is from venv
PYTHON_VERSION=$(python --version)
echo "Python: ${PYTHON_VERSION}"
echo "Python path: $(which python)"
echo ""

# Install package in editable mode if not already installed
echo "Installing package..."
cd "${PROJECT_ROOT}"
pip install -e . --quiet

# Run tests
echo ""
echo "Running tests..."
echo "========================================="

# Build pytest arguments
PYTEST_ARGS="${SCRIPT_DIR}"

# Pass through any additional arguments
if [ $# -gt 0 ]; then
    PYTEST_ARGS="${PYTEST_ARGS} $@"
fi

# Run pytest
python -m pytest ${PYTEST_ARGS}

# Capture exit code
TEST_EXIT_CODE=$?

echo ""
echo "========================================="
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Tests failed with exit code: ${TEST_EXIT_CODE}"
fi
echo "========================================="

exit $TEST_EXIT_CODE
