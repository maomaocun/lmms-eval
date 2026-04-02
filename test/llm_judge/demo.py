#!/usr/bin/env python
"""Demo script showing how to use the judge command.

This script creates sample JSONL data and demonstrates the judge functionality.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def create_sample_jsonl(output_path: Path, num_samples: int = 5):
    """Create sample JSONL data for testing."""
    
    samples = []
    for i in range(num_samples):
        # Create samples with varying correctness
        is_correct = i % 2 == 0
        
        sample = {
            "doc_id": i,
            "doc": {
                "question": f"What is {i}+{i}?",
                "answer": str(i + i),
                "options": [str(i), str(i+i), str(i+i+1), str(i*3)],
                "question_type": "open",
            },
            "filtered_resps": (
                f"<think>The sum of {i} and {i} is {i+i}.</think><answer>{i+i}</answer>"
                if is_correct
                else f"<think>I think the answer is {i+i+1}.</think><answer>{i+i+1}</answer>"
            ),
            "target": str(i + i),
            "input": f"What is {i}+{i}?",
        }
        samples.append(sample)
    
    # Write to file
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {num_samples} samples in {output_path}")
    return samples


def demo_rule_judging():
    """Demonstrate rule-based judging."""
    print("\n" + "=" * 60)
    print("Demo: Rule-based Judging")
    print("=" * 60)
    
    from lmms_eval.llm_judge.standalone import JudgeRunner
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create sample data
        input_file = tmp_path / "samples_math_test.jsonl"
        create_sample_jsonl(input_file, num_samples=5)
        
        # Create mock process_results function
        def mock_process_results(doc, results):
            import re
            
            pred = results[0] if results else ""
            answer = doc.get("answer", "")
            
            # Extract from <answer> tags
            match = re.search(r"<answer>(.*?)</answer>", pred)
            extracted = match.group(1) if match else pred
            
            correct = str(answer).strip() == str(extracted).strip()
            has_format = "<answer>" in pred and "</answer>" in pred
            
            return {
                "acc_score": 1.0 if correct else 0.0,
                "format_score": 1.0 if has_format else 0.0,
            }
        
        # Create runner
        runner = JudgeRunner(judge_mode="rule")
        
        # Create mock task
        from unittest.mock import MagicMock
        mock_task = MagicMock()
        mock_task.config.process_results = mock_process_results
        
        # Judge file
        import json
        with open(input_file) as f:
            samples = [json.loads(line) for line in f]
        
        judged = []
        for sample in samples:
            result = runner._judge_sample(
                sample, mock_task, mock_task.config.process_results
            )
            judged.append(result)
        
        # Show results
        print("\nResults:")
        for r in judged:
            doc_id = r["doc_id"]
            metrics = r["metrics"]
            print(f"  Sample {doc_id}: acc_score={metrics['acc_score']:.0f}, "
                  f"format_score={metrics['format_score']:.0f}")
        
        correct = sum(1 for r in judged if r["metrics"]["acc_score"] == 1.0)
        print(f"\nSummary: {correct}/{len(judged)} correct")


def demo_cli_help():
    """Show CLI help."""
    print("\n" + "=" * 60)
    print("CLI Help")
    print("=" * 60)
    
    from lmms_eval.cli.judge_cmd import add_judge_parser
    import argparse
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_judge_parser(subparsers)
    
    # Print help
    print("\nAvailable options:")
    print("  --input_result, -i: Input JSONL result file(s)")
    print("  --task, -t        : Task name (or auto-detect)")
    print("  --output, -o      : Output file path")
    print("  --output-dir, -d  : Output directory")
    print("  --judge-mode      : rule | llm | auto")
    print("  --judge-model     : Judge model name")
    print("  --parallel, -p    : Parallel workers")
    print("  --dry-run         : Dry run without saving")


def main():
    """Run demos."""
    print("=" * 60)
    print("LMMS-Eval Judge Module Demo")
    print("=" * 60)
    
    demo_rule_judging()
    demo_cli_help()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print("\nTo use the judge command:")
    print("  lmms-eval judge -i results.jsonl -t task_name")
    print("\nExample with LLM judge:")
    print("  export JUDGE_API_KEY=sk-...")
    print("  lmms-eval judge -i results.jsonl --judge-mode llm")


if __name__ == "__main__":
    main()
