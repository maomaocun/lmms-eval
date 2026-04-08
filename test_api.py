#!/usr/bin/env python3
"""
Test script for API connectivity
Supports: DashScope, OpenAI, OpenAI-compatible APIs
Note: Kimi Code API is not supported for direct API calls (requires specific Coding Agents)
"""

import os
import sys
import requests


def test_api(api_name, api_key, api_base, payload_format="openai", model="gpt-4o-mini"):
    """Test generic API connectivity."""
    if not api_key:
        print(f"❌ {api_name} API key not set")
        return False
    
    print(f"Testing {api_name} API...")
    print(f"  API_BASE: {api_base}")
    print(f"  Model: {model}")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    if payload_format == "dashscope":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'API test successful'"}],
            "max_completion_tokens": 50,
            "temperature": 0
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Say 'API test successful'"}],
            "max_tokens": 50,
            "temperature": 0
        }
    
    try:
        response = requests.post(api_base, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content']
            print(f"✅ {api_name} API test successful")
            print(f"   Response: {content[:50]}...")
            return True
        else:
            print(f"❌ {api_name} API test failed: HTTP {response.status_code}")
            try:
                error_content = response.json()
                print(f"   Error: {error_content.get('error', {}).get('message', response.text)}")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ {api_name} API test failed: {e}")
        return False


def test_dashscope_api():
    """Test DashScope API connectivity."""
    api_key = os.environ.get('CHATGPT_DASHSCOPE_API_KEY', '')
    api_base = os.environ.get('DASHSCOPE_API_BASE', 
                              'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
    return test_api("DashScope", api_key, api_base, payload_format="dashscope")


def test_openai_api():
    """Test OpenAI API connectivity."""
    api_key = os.environ.get('OPENAI_API_KEY', '')
    api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1/chat/completions')
    return test_api("OpenAI", api_key, api_base)


def test_compatible_api():
    """Test OpenAI-compatible API connectivity."""
    api_key = os.environ.get('OPENAI_COMPATIBLE_KEY', '')
    api_base = os.environ.get('OPENAI_COMPATIBLE_URL', '')
    if not api_base:
        print("⚠️ OpenAI-compatible API URL not set")
        return False
    # yunwu.ai 等服务商可能不支持 gpt-3.5-turbo，使用 gpt-4o-mini
    return test_api("OpenAI-Compatible", api_key, api_base, model="gpt-4o-mini")


def test_kimi_api():
    """Test Kimi API connectivity."""
    api_key = os.environ.get('KIMI_API_KEY', '')
    api_base = os.environ.get('KIMI_API_BASE', 'https://api.kimi.com/coding/v1/chat/completions')
    
    if not api_key:
        return None  # Not configured
    
    print(f"Testing Kimi API...")
    print(f"  API_BASE: {api_base}")
    print(f"  ⚠️ Note: Kimi Code API may have access restrictions")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Say 'API test successful'"}],
        "max_tokens": 50,
        "temperature": 0
    }
    
    try:
        response = requests.post(api_base, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content']
            print(f"✅ Kimi API test successful")
            print(f"   Response: {content[:50]}...")
            return True
        else:
            print(f"❌ Kimi API test failed: HTTP {response.status_code}")
            try:
                error_content = response.json()
                error_msg = error_content.get('error', {}).get('message', '')
                print(f"   Error: {error_msg}")
                if 'access_terminated_error' in str(error_content) or 'only available for Coding Agents' in error_msg:
                    print(f"   ℹ️ This API Key is restricted to specific Coding Agents.")
                    print(f"   Please use DashScope, OpenAI, or other compatible APIs instead.")
            except:
                print(f"   Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Kimi API test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("API Connectivity Test for Qwen3-VL Evaluation")
    print("=" * 60)
    print()
    
    # Test all APIs
    results = {}
    
    # Kimi API (may be restricted)
    results['kimi'] = test_kimi_api()
    print()
    
    # DashScope API
    results['dashscope'] = test_dashscope_api()
    print()
    
    # OpenAI API
    results['openai'] = test_openai_api()
    print()
    
    # OpenAI-compatible API
    results['compatible'] = test_compatible_api()
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if results['kimi'] is None:
        print("⚪ Kimi API: Not configured")
    elif results['kimi']:
        print("✅ Kimi API: Ready")
    else:
        print("❌ Kimi API: Failed (restricted to Coding Agents)")
    
    if results['dashscope']:
        print("✅ DashScope API: Ready")
    else:
        print("❌ DashScope API: Not configured or failed")
    
    if results['openai']:
        print("✅ OpenAI API: Ready")
    else:
        print("❌ OpenAI API: Not configured or failed")
    
    if results['compatible'] is None:
        print("⚪ OpenAI-Compatible API: Not configured")
    elif results['compatible']:
        print("✅ OpenAI-Compatible API: Ready")
    else:
        print("❌ OpenAI-Compatible API: Failed")
    
    print()
    
    # Check if any API is ready
    ready_apis = [k for k, v in results.items() if v is True]
    
    if ready_apis:
        print(f"✅ Ready APIs: {', '.join(ready_apis)}")
        print()
        print("You can now run evaluation:")
        if 'dashscope' in ready_apis:
            print("  API_TYPE=dash bash run_all_evaluations.sh")
        if 'openai' in ready_apis:
            print("  API_TYPE=openai bash run_all_evaluations.sh")
        if 'compatible' in ready_apis:
            print("  API_TYPE=compatible bash run_all_evaluations.sh")
        return 0
    else:
        print("❌ No API is ready. Please configure API keys first.")
        print()
        print("To setup API keys, edit and source:")
        print("  vim setup_api_keys.sh")
        print("  source setup_api_keys.sh")
        print()
        print("Recommended APIs for China users:")
        print("  1. DashScope (阿里云): https://dashscope.aliyun.com")
        print("  2. SiliconFlow: https://siliconflow.cn")
        print("  3. OpenAI: https://platform.openai.com")
        return 1


if __name__ == "__main__":
    sys.exit(main())
