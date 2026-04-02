#!/usr/bin/env python
"""
Example 6: Local vLLM/SGLang as Judge

This example shows how to use a locally deployed vLLM or SGLang server
as the judge model. This is useful when:
1. You don't want to send data to cloud APIs
2. You need faster inference (no network latency)
3. You want to use custom fine-tuned models
4. You have GPU resources available locally

Prerequisites:
1. Install vLLM: pip install vllm
2. Start vLLM server:
   vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16 --max-model-len 8192
   
   Or with SGLang:
   python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct

3. Set environment variables (or create .env.judge file)
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from unittest.mock import MagicMock


def check_local_server():
    """Check if local vLLM/SGLang server is running."""
    import urllib.request
    
    base_url = os.getenv("JUDGE_BASE_URL", "http://localhost:8000/v1")
    # Remove '/v1' suffix for health check
    health_url = base_url.replace("/v1", "/health") if base_url.endswith("/v1") else f"{base_url}/health"
    
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def create_test_samples(output_path: Path, num_samples: int = 3):
    """Create test samples for local judging."""
    print(f"Creating {num_samples} test samples...")
    
    samples = [
        {
            "doc_id": 0,
            "doc": {
                "question": "What is the capital of France?",
                "answer": "Paris",
            },
            "filtered_resps": "The capital of France is Paris, a beautiful city known for the Eiffel Tower.",
        },
        {
            "doc_id": 1,
            "doc": {
                "question": "Calculate 15 * 4",
                "answer": "60",
            },
            "filtered_resps": "15 multiplied by 4 equals 60.",
        },
        {
            "doc_id": 2,
            "doc": {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare",
            },
            "filtered_resps": "William Shakespeare wrote Romeo and Juliet in the late 16th century.",
        },
    ]
    
    with open(output_path, "w") as f:
        for sample in samples[:num_samples]:
            f.write(json.dumps(sample) + "\n")
    
    print(f"✓ Created {output_path}\n")
    return samples[:num_samples]


def demo_local_vllm():
    """Demonstrate using local vLLM as judge."""
    print("=" * 70)
    print("Demo: Local vLLM/SGLang as Judge")
    print("=" * 70)
    print()
    
    # Check configuration
    base_url = os.getenv("JUDGE_BASE_URL", "")
    api_key = os.getenv("JUDGE_API_KEY", "")
    model = os.getenv("JUDGE_MODEL", "local-model")
    
    print("Configuration:")
    url_display = base_url if base_url else "Not set (using default)"
    print(f"  JUDGE_BASE_URL: {url_display}")
    print(f"  JUDGE_MODEL: {model}")
    key_status = "Set" if api_key else "Not set (will use dummy)"
    print(f"  JUDGE_API_KEY: {key_status}")
    print()
    
    # Check if it looks like a local server
    is_local = any(x in base_url for x in ["localhost", "127.0.0.1", ":8000", ":30000"])
    
    if not is_local:
        print("⚠ Warning: JUDGE_BASE_URL doesn't look like a local server.")
        print("  For local vLLM, set: export JUDGE_BASE_URL=http://localhost:8000/v1")
        print()
    
    # Check if server is running
    print("Checking if local server is running...")
    if check_local_server():
        print("✓ Local LLM server is running!\n")
    else:
        print("✗ Local LLM server is NOT running or not accessible.")
        print()
        print("To start a local server:")
        print("  # Using vLLM:")
        print("  vllm serve Qwen/Qwen2.5-VL-7B-Instruct --dtype bfloat16")
        print()
        print("  # Using SGLang:")
        print("  python -m sglang.launch_server --model-path Qwen/Qwen2.5-VL-7B-Instruct")
        print()
        print("Then set the environment and run again:")
        print("  export JUDGE_BASE_URL=http://localhost:8000/v1")
        print("  export JUDGE_API_KEY=dummy  # Local servers often don't need real keys")
        print()
        return False
    
    # Run judging
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create samples
        input_file = tmp_path / "samples.jsonl"
        create_test_samples(input_file)
        
        print("Initializing judge runner for local LLM...")
        
        try:
            runner = JudgeRunner(
                judge_mode="llm",
                judge_model=model,
                judge_api_key=api_key or "dummy-key",
                judge_base_url=base_url or "http://localhost:8000/v1",
            )
            
            # Create mock task
            mock_task = MagicMock()
            mock_task.config.process_results = lambda doc, res: {"acc_score": 0.0}
            
            # Judge samples
            with open(input_file) as f:
                samples = [json.loads(line) for line in f]
            
            print(f"Judging {len(samples)} samples with local LLM...\n")
            
            results = []
            for i, sample in enumerate(samples):
                print(f"  Judging sample {i+1}/{len(samples)}...", end=" ")
                result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
                
                score = result["metrics"].get("llm_judge_score", "N/A")
                print(f"score={score}")
                results.append(result)
            
            # Summary
            print("\n" + "-" * 70)
            print("Results Summary")
            print("-" * 70)
            
            correct = sum(1 for r in results if r["metrics"].get("llm_judge_score") == 1)
            print(f"\nJudged {len(results)} samples")
            print(f"Correct: {correct}/{len(results)}")
            
            # Show details
            print("\nDetails:")
            for r in results:
                q = r["doc"]["question"]
                score = r["metrics"].get("llm_judge_score", "N/A")
                print(f"  Q: {q[:50]}...")
                print(f"     Score: {score}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("\nTroubleshooting:")
            print("  1. Verify the server is running: curl http://localhost:8000/v1/models")
            print("  2. Check the base URL matches your server port")
            print("  3. Ensure the model name is correct")
            return False


def demo_configuration_options():
    """Show different configuration options for local LLMs."""
    print("\n" + "=" * 70)
    print("Configuration Options for Local LLM Judge")
    print("=" * 70)
    print()
    
    configs = [
        {
            "name": "vLLM Local",
            "env": {
                "JUDGE_BASE_URL": "http://localhost:8000/v1",
                "JUDGE_MODEL": "Qwen2.5-VL-7B-Instruct",
                "JUDGE_API_KEY": "dummy",
            },
            "description": "Standard vLLM deployment",
        },
        {
            "name": "SGLang Local",
            "env": {
                "JUDGE_BASE_URL": "http://localhost:30000/v1",
                "JUDGE_MODEL": "Qwen2.5-VL-7B-Instruct",
                "JUDGE_API_KEY": "dummy",
            },
            "description": "SGLang server (default port 30000)",
        },
        {
            "name": "vLLM Remote",
            "env": {
                "JUDGE_BASE_URL": "http://192.168.1.100:8000/v1",
                "JUDGE_MODEL": "Meta-Llama-3-8B-Instruct",
                "JUDGE_API_KEY": "dummy",
            },
            "description": "Remote vLLM on another machine",
        },
        {
            "name": "LM Studio",
            "env": {
                "JUDGE_BASE_URL": "http://localhost:1234/v1",
                "JUDGE_MODEL": "local-model",
                "JUDGE_API_KEY": "lm-studio",
            },
            "description": "LM Studio local inference",
        },
    ]
    
    for cfg in configs:
        print(f"\n{cfg['name']}:")
        print(f"  Description: {cfg['description']}")
        print("  Environment:")
        for key, val in cfg['env'].items():
            print(f"    export {key}={val}")


def demo_performance_comparison():
    """Show performance benefits of local LLM."""
    print("\n" + "=" * 70)
    print("Local vs Cloud LLM Judge")
    print("=" * 70)
    print()
    
    comparison = [
        ("Latency", "~1-10ms", "~100-500ms", "No network round-trip"),
        ("Cost", "Free (after hardware)", "Per-token pricing", "No API fees"),
        ("Privacy", "Data stays local", "Sent to cloud", "Sensitive data safe"),
        ("Rate Limits", "None (hardware limited)", "API quotas", "Judge large batches"),
        ("Customization", "Any local model", "Provider's models", "Use fine-tuned models"),
        ("Setup", "Requires GPU server", "Just API key", "One-time setup vs ongoing"),
    ]
    
    print(f"{'Aspect':<20} {'Local vLLM':<20} {'Cloud API':<20} {'Advantage'}")
    print("-" * 70)
    for aspect, local, cloud, advantage in comparison:
        print(f"{aspect:<20} {local:<20} {cloud:<20} {advantage}")


def main():
    print("=" * 70)
    print("Example 6: Local vLLM/SGLang as Judge")
    print("=" * 70)
    print()
    print("This example demonstrates using a local LLM server (vLLM or SGLang)")
    print("as the judge instead of cloud APIs like OpenAI.")
    print()
    
    # Run demos
    success = demo_local_vllm()
    demo_configuration_options()
    demo_performance_comparison()
    
    print("\n" + "=" * 70)
    if success:
        print("Example completed successfully!")
    else:
        print("Example completed (server not running)")
    print("=" * 70)
    print()
    print("Quick Start:")
    print("  1. Start vLLM: vllm serve Qwen/Qwen2.5-VL-7B-Instruct")
    print("  2. Set environment:")
    print("     export JUDGE_BASE_URL=http://localhost:8000/v1")
    print("     export JUDGE_API_KEY=dummy")
    print("  3. Run judge:")
    print("     lmms-eval judge -i results.jsonl --judge-mode llm")
    print()
    print("Note: The model must support the required context length for judging.")
    print("      Recommended: 8K+ context for most judging tasks.")


if __name__ == "__main__":
    main()
