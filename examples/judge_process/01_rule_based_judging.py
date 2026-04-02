#!/usr/bin/env python
"""
Example 1: Rule-based Judging

This example shows how to use the judge command with rule-based scoring.
This is useful when you want to:
1. Re-judge results with updated rules
2. Compare different rule implementations
3. Validate existing results
"""

import json
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from unittest.mock import MagicMock


def create_sample_results(output_path: Path, num_samples: int = 10):
    """Create sample model output data."""
    print(f"Creating {num_samples} sample results...")
    
    samples = []
    for i in range(num_samples):
        # Mix of correct and incorrect answers
        is_correct = i % 3 != 0  # 2/3 correct
        answer = i + i
        prediction = answer if is_correct else answer + 1
        
        sample = {
            "doc_id": i,
            "doc": {
                "question": f"What is {i} + {i}?",
                "answer": str(answer),
                "question_type": "open",
                "difficulty": "easy" if i < 5 else "hard",
            },
            "filtered_resps": (
                f"<think>Adding {i} and {i} gives {prediction}.</think>"
                f"<answer>{prediction}</answer>"
            ),
            "target": str(answer),
            "input": f"What is {i} + {i}?",
        }
        samples.append(sample)
    
    # Write to JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"✓ Created {output_path}")
    return samples


def rule_based_judge(doc: dict, results: list) -> dict:
    """
    Custom rule-based judging function.
    
    This demonstrates how you can implement custom judging logic
    that checks both answer correctness and response format.
    """
    import re
    
    prediction = results[0] if results else ""
    answer = doc.get("answer", "")
    
    # Check format: must have <think> and <answer> tags
    has_think = bool(re.search(r"<think>.*?</think>", prediction, re.DOTALL))
    has_answer_tag = bool(re.search(r"<answer>.*?</answer>", prediction))
    
    # Extract answer
    match = re.search(r"<answer>(.*?)</answer>", prediction)
    extracted = match.group(1).strip() if match else prediction.strip()
    
    # Check correctness
    correct = str(answer).strip() == str(extracted).strip()
    
    return {
        "acc_score": 1.0 if correct else 0.0,
        "format_score": 1.0 if (has_think and has_answer_tag) else 0.0,
        "has_think_tag": has_think,
        "has_answer_tag": has_answer_tag,
        "extracted_answer": extracted,
    }


def main():
    print("=" * 70)
    print("Example 1: Rule-based Judging")
    print("=" * 70)
    print()
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Step 1: Create sample data
        input_file = tmp_path / "model_outputs.jsonl"
        create_sample_results(input_file, num_samples=10)
        
        # Step 2: Setup judge runner
        print("\nInitializing judge runner...")
        runner = JudgeRunner(judge_mode="rule")
        
        # Step 3: Create mock task with our custom judging function
        mock_task = MagicMock()
        mock_task.config.process_results = rule_based_judge
        
        # Step 4: Judge each sample
        print("\nJudging samples...")
        with open(input_file) as f:
            samples = [json.loads(line) for line in f]
        
        judged_samples = []
        for sample in samples:
            result = runner._judge_sample(
                sample, mock_task, rule_based_judge
            )
            judged_samples.append(result)
        
        # Step 5: Analyze results
        print("\n" + "-" * 70)
        print("Results Summary")
        print("-" * 70)
        
        total = len(judged_samples)
        correct = sum(1 for s in judged_samples if s["metrics"]["acc_score"] == 1.0)
        well_formatted = sum(1 for s in judged_samples if s["metrics"]["format_score"] == 1.0)
        
        print(f"\nTotal samples: {total}")
        print(f"Correct answers: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"Well formatted: {well_formatted}/{total} ({100*well_formatted/total:.1f}%)")
        
        # Show breakdown by difficulty
        print("\nBreakdown by difficulty:")
        easy_samples = [s for s in judged_samples if s["doc"].get("difficulty") == "easy"]
        hard_samples = [s for s in judged_samples if s["doc"].get("difficulty") == "hard"]
        
        if easy_samples:
            easy_correct = sum(1 for s in easy_samples if s["metrics"]["acc_score"] == 1.0)
            print(f"  Easy: {easy_correct}/{len(easy_samples)} correct")
        
        if hard_samples:
            hard_correct = sum(1 for s in hard_samples if s["metrics"]["acc_score"] == 1.0)
            print(f"  Hard: {hard_correct}/{len(hard_samples)} correct")
        
        # Step 6: Show some examples
        print("\n" + "-" * 70)
        print("Example Judgments")
        print("-" * 70)
        
        for i, sample in enumerate(judged_samples[:3]):
            doc_id = sample["doc_id"]
            question = sample["doc"]["question"]
            metrics = sample["metrics"]
            
            print(f"\nSample {doc_id}:")
            print(f"  Question: {question}")
            print(f"  Answer: {sample['doc']['answer']}")
            print(f"  Prediction: {metrics.get('extracted_answer', 'N/A')}")
            print(f"  Correct: {'✓' if metrics['acc_score'] == 1.0 else '✗'}")
            print(f"  Format valid: {'✓' if metrics['format_score'] == 1.0 else '✗'}")
        
        # Step 7: Save judged results
        output_file = tmp_path / "judged_results.jsonl"
        runner.save_results(judged_samples, output_file)
        print(f"\n✓ Saved judged results to {output_file}")
        
        # Show file content preview
        print("\n" + "-" * 70)
        print("Output File Preview")
        print("-" * 70)
        with open(output_file) as f:
            first_line = json.loads(f.readline())
            print(f"\nKeys in output: {list(first_line.keys())}")
            print(f"Metrics: {first_line['metrics']}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nTo use with the CLI:")
    print("  lmms-eval judge -i model_outputs.jsonl -t your_task")
    print("\nTo implement custom judging:")
    print("  1. Define your process_results function")
    print("  2. Pass it to JudgeRunner or configure in task YAML")


if __name__ == "__main__":
    main()
