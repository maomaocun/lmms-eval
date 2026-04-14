#!/bin/bash

# ============================================================================
# API Keys Setup Script for Qwen3-VL Evaluation
# ============================================================================
# This script sets up API keys for GPT-based evaluation
#
# Usage:
#   source setup_api_keys.sh
#
# Supported API:
#   - yunwu.ai (OpenAI-compatible)
#
# Note: Kimi Code API 有限制，只能在特定 Coding Agents 中使用
#       无法直接用于评测脚本
# ============================================================================

echo "=========================================="
echo "API Keys Setup for Qwen3-VL Evaluation"
echo "=========================================="
echo ""

# ============================================================================
# Option 1: yunwu.ai (OpenAI-compatible) - 当前配置
# ============================================================================
# 我们将 API_TYPE 设为 "openai"，因为代码库中绝大多数 task 的 judge 逻辑
# 只识别 "openai" 和 "azure" 两种类型。yunwu.ai 本身是 OpenAI-compatible 的，
# 使用 "openai" 类型并传入自定义 base_url 即可正常工作。
export API_TYPE="openai"
export OPENAI_API_KEY=""
export OPENAI_API_URL=""

echo "✓ yunwu.ai API key configured"
echo "  API_BASE: $OPENAI_API_URL"
echo ""

echo ""
echo "=========================================="
echo "API Configuration Summary"
echo "=========================================="
echo "  Current API: openai (via yunwu.ai compatible endpoint)"
echo "  API_TYPE: $API_TYPE"
echo "  To use: source setup_api_keys.sh && bash run_all_evaluations.sh"
echo ""
echo "To run evaluation:"
echo "  bash run_all_evaluations.sh"
echo ""
echo "To test API connection:"
echo "  python test_api.py"
echo ""
