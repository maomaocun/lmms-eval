#!/bin/bash

# ============================================================================
# API Keys Setup Script for Qwen3-VL Evaluation
# ============================================================================
# This script sets up API keys for GPT-based evaluation
#
# Usage:
#   source setup_api_keys.sh
#
# Supported APIs:
#   1. yunwu.ai (OpenAI-compatible) - 当前配置
#   2. DashScope API (Alibaba Cloud)
#   3. OpenAI API
#   4. 其他 OpenAI-compatible API
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
export OPENAI_COMPATIBLE_KEY=""
export OPENAI_COMPATIBLE_URL="https://yunwu.ai/v1/chat/completions"

echo "✓ yunwu.ai API key configured"
echo "  API_BASE: $OPENAI_COMPATIBLE_URL"
echo ""

# ============================================================================
# Option 2: DashScope API (Alibaba Cloud)
# ============================================================================
# 如果你有阿里云 DashScope API Key，取消下面两行的注释并填写
# export CHATGPT_DASHSCOPE_API_KEY="your-dashscope-api-key"
# export DASHSCOPE_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

if [ -n "$CHATGPT_DASHSCOPE_API_KEY" ]; then
    echo "✓ DashScope API key configured"
    echo "  API_BASE: $DASHSCOPE_API_BASE"
fi

# ============================================================================
# Option 3: OpenAI API
# ============================================================================
# 如果你有 OpenAI API Key，取消下面两行的注释并填写
# export OPENAI_API_KEY="your-openai-api-key"
# export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"

if [ -n "$OPENAI_API_KEY" ]; then
    echo "✓ OpenAI API key configured"
    echo "  API_BASE: $OPENAI_API_BASE"
fi

echo ""
echo "=========================================="
echo "API Configuration Summary"
echo "=========================================="
echo "  Current API: compatible (yunwu.ai)"
echo "  To use: API_TYPE=compatible bash run_all_evaluations.sh"
echo ""
echo "To run evaluation:"
echo "  bash run_all_evaluations.sh"
echo ""
echo "To test API connection:"
echo "  python test_api.py"
echo ""
