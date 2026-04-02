#!/usr/bin/env python
"""
Example 3: Batch Processing

This example shows how to process multiple JSONL files at once,
useful when you have results from multiple models or tasks.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from unittest.mock import MagicMock


def create_multiple_result_files(base_dir: Path):
    """Create multiple result files simulating different models."""
    print("Creating sample result files from multiple models...\n")
    
    models = ["model_a", "model_b", "model_c"]
    tasks = ["math_vision", "math_verse", "math_vista"]
    
    created_files = []
    
    for model in models:
        for task in tasks:
            # Create filename following convention: {timestamp}_samples_{task}.jsonl
            filename = f"20240328_samples_{task}_{model}.jsonl"
            filepath = base_dir / filename
            
            # Generate sample data with varying accuracy per model
            samples = []
            for i in range(5):
                # Model A: 80% accuracy
                # Model B: 60% accuracy
                # Model C: 40% accuracy
                accuracy_map = {"model_a": 0.8, "model_b": 0.6, "model_c": 0.4}
                is_correct = (i / 5) < accuracy_map[model]
                
                answer = i * 2
                prediction = answer if is_correct else answer + 1
                
                sample = {
                    "doc_id": i,
                    "doc": {
                        "question": f"What is {i} * 2?",
                        "answer": str(answer),
                        "task": task,
                    },
                    "filtered_resps": f"<answer>{prediction}</answer>",
                    "target": str(answer),
                    "model": model,
                }
                samples.append(sample)
            
            # Write to file
            with open(filepath, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")
            
            created_files.append(filepath)
            print(f"  ✓ {filename} ({len(samples)} samples)")
    
    print(f"\nTotal: {len(created_files)} files")
    return created_files


def simple_judge(doc: dict, results: list) -> dict:
    """Simple judging function."""
    import re
    
    prediction = results[0] if results else ""
    answer = doc.get("answer", "")
    
    # Extract from <answer> tags
    match = re.search(r"<answer>(.*?)</answer>", prediction)
    extracted = match.group(1) if match else prediction
    
    correct = str(answer).strip() == str(extracted).strip()
    
    return {
        "acc_score": 1.0 if correct else 0.0,
        "extracted": extracted,
    }


def demo_single_file():
    """Process a single file."""
    print("\n" + "-" * 70)
    print("Demo 1: Single File Processing")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files = create_multiple_result_files(tmp_path)
        
        # Process first file
        input_file = files[0]
        print(f"\nProcessing: {input_file.name}")
        
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = simple_judge
        
        # Override task loading
        with open(input_file) as f:
            samples = [json.loads(line) for line in f]
        
        judged = []
        for sample in samples:
            result = runner._judge_sample(sample, mock_task, simple_judge)
            judged.append(result)
        
        # Results
        correct = sum(1 for j in judged if j["metrics"]["acc_score"] == 1.0)
        print(f"Results: {correct}/{len(judged)} correct ({100*correct/len(judged):.0f}%)")
        
        # Save
        output_file = tmp_path / f"judged_{input_file.name}"
        runner.save_results(judged, output_file)
        print(f"Saved to: {output_file.name}")


def demo_batch_by_model():
    """Process all files for each model."""
    print("\n" + "-" * 70)
    print("Demo 2: Batch Processing by Model")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        all_files = create_multiple_result_files(tmp_path)
        
        # Group by model
        models = {}
        for f in all_files:
            model = f.stem.split("_")[-1]  # Extract model name from filename
            if model not in models:
                models[model] = []
            models[model].append(f)
        
        print(f"\nFound {len(models)} models")
        
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = simple_judge
        
        # Process each model
        results_by_model = {}
        
        for model, files in sorted(models.items()):
            print(f"\nProcessing {model} ({len(files)} files)...")
            
            all_judged = []
            for f in files:
                with open(f) as fp:
                    samples = [json.loads(line) for line in fp]
                
                for sample in samples:
                    judged = runner._judge_sample(sample, mock_task, simple_judge)
                    all_judged.append(judged)
            
            # Calculate aggregate metrics
            correct = sum(1 for j in all_judged if j["metrics"]["acc_score"] == 1.0)
            accuracy = correct / len(all_judged) if all_judged else 0
            
            results_by_model[model] = {
                "total": len(all_judged),
                "correct": correct,
                "accuracy": accuracy,
            }
            
            print(f"  {correct}/{len(all_judged)} correct ({100*accuracy:.1f}%)")
        
        # Summary table
        print("\n" + "-" * 50)
        print(f"{'Model':<15} {'Total':<10} {'Correct':<10} {'Accuracy':<10}")
        print("-" * 50)
        for model, stats in sorted(results_by_model.items()):
            print(f"{model:<15} {stats['total']:<10} {stats['correct']:<10} {stats['accuracy']:.1%}")


def demo_wildcard_processing():
    """Demonstrate wildcard file matching."""
    print("\n" + "-" * 70)
    print("Demo 3: Wildcard File Matching")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        all_files = create_multiple_result_files(tmp_path)
        
        # Simulate wildcard matching
        patterns = [
            "*math_vision*.jsonl",    # Should match 3 files
            "*model_a*.jsonl",        # Should match 3 files
            "*math_vision_model_a*",  # Should match 1 file
        ]
        
        for pattern in patterns:
            print(f"\nPattern: {pattern}")
            matched = list(tmp_path.glob(pattern))
            print(f"  Matched {len(matched)} files:")
            for m in matched:
                print(f"    - {m.name}")
        
        # CLI equivalent
        print("\nCLI equivalent:")
        print("  lmms-eval judge -i '*math_vision*.jsonl' -d judged/")


def demo_parallel_processing():
    """Demonstrate parallel processing setup."""
    print("\n" + "-" * 70)
    print("Demo 4: Parallel Processing Configuration")
    print("-" * 70)
    
    print("\nParallel processing is configured via:")
    print("  1. Command line: --parallel 8")
    print("  2. Environment: export JUDGE_MAX_CONCURRENT=8")
    
    # Show configuration
    configs = [
        {"mode": "single", "parallel": 1},
        {"mode": "moderate", "parallel": 4},
        {"mode": "aggressive", "parallel": 16},
    ]
    
    print("\nRecommended settings:")
    print(f"{'Mode':<15} {'Workers':<10} {'Use Case'}")
    print("-" * 50)
    
    use_cases = {
        "single": "Debugging, API rate limits",
        "moderate": "Standard usage (recommended)",
        "aggressive": "Local judge server, high throughput",
    }
    
    for cfg in configs:
        print(f"{cfg['mode']:<15} {cfg['parallel']:<10} {use_cases[cfg['mode']]}")
    
    # Example initialization
    print("\nExample initialization:")
    print("  runner = JudgeRunner(")
    print("      judge_mode='llm',")
    print("      judge_model='gpt-4o-mini',")
    print("      parallel=8,")
    print("  )")


def main():
    print("=" * 70)
    print("Example 3: Batch Processing")
    print("=" * 70)
    print()
    print("This example demonstrates processing multiple JSONL files,")
    print("useful when you have results from multiple models or tasks.")
    print()
    
    demo_single_file()
    demo_batch_by_model()
    demo_wildcard_processing()
    demo_parallel_processing()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Use wildcards to process multiple files: *.jsonl")
    print("  • Group by model/task for aggregate analysis")
    print("  • Set --parallel for faster processing")
    print("\nCLI examples:")
    print("  # All files")
    print("  lmms-eval judge -i 'results/*.jsonl' -d judged/")
    print()
    print("  # Specific model")
    print("  lmms-eval judge -i 'results/*model_a*.jsonl' -d judged/model_a/")
    print()
    print("  # With parallelism")
    print("  lmms-eval judge -i '*.jsonl' --parallel 8")


if __name__ == "__main__":
    main()
