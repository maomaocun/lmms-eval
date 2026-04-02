#!/usr/bin/env python
"""
Example 5: Model Comparison

This example shows how to judge outputs from multiple models
and generate comparison reports.
"""

import json
import sys
import tempfile
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from unittest.mock import MagicMock


def create_model_outputs(base_dir: Path):
    """Create outputs from multiple models on the same questions."""
    print("Creating model outputs for comparison...\n")
    
    # Same questions for all models
    questions = [
        {"id": 0, "q": "What is 2+2?", "answer": "4"},
        {"id": 1, "q": "Capital of France?", "answer": "Paris"},
        {"id": 2, "q": "Largest planet?", "answer": "Jupiter"},
        {"id": 3, "q": "Speed of light (m/s)?", "answer": "299792458"},
        {"id": 4, "q": "Number of continents?", "answer": "7"},
    ]
    
    # Model outputs with different accuracy levels
    models = {
        "GPT-4o": {
            "accuracy": 1.0,  # All correct
            "style": "detailed",
        },
        "Claude-3": {
            "accuracy": 0.8,  # 4/5 correct
            "style": "concise",
        },
        "Qwen2.5-VL": {
            "accuracy": 0.6,  # 3/5 correct
            "style": "medium",
        },
    }
    
    created_files = []
    
    for model_name, config in models.items():
        filename = f"samples_comparison_{model_name.lower().replace('-', '_')}.jsonl"
        filepath = base_dir / filename
        
        samples = []
        for i, q in enumerate(questions):
            # Determine if this model gets this question right
            is_correct = (i / len(questions)) < config["accuracy"]
            
            # Generate response based on style
            if config["style"] == "detailed":
                if is_correct:
                    resp = f"<think>Based on my knowledge, {q['q']} The answer is clearly {q['answer']}.</think><answer>{q['answer']}</answer>"
                else:
                    wrong = str(int(q["answer"]) + 1) if q["answer"].isdigit() else "Unknown"
                    resp = f"<think>I'm not entirely sure, but I believe the answer might be {wrong}.</think><answer>{wrong}</answer>"
            elif config["style"] == "concise":
                resp = f"<answer>{q['answer'] if is_correct else 'Wrong'}</answer>"
            else:
                resp = f"The answer is <answer>{q['answer'] if is_correct else 'Error'}</answer>."
            
            sample = {
                "doc_id": q["id"],
                "doc": {
                    "question": q["q"],
                    "answer": q["answer"],
                    "category": "knowledge",
                },
                "filtered_resps": resp,
                "target": q["answer"],
                "model": model_name,
            }
            samples.append(sample)
        
        # Write file
        with open(filepath, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        created_files.append((model_name, filepath, samples))
        print(f"  ✓ {model_name}: {len(samples)} samples ({config['accuracy']:.0%} expected accuracy)")
    
    return created_files


def simple_judge(doc: dict, results: list) -> dict:
    """Simple judging function."""
    import re
    
    prediction = results[0] if results else ""
    answer = doc.get("answer", "")
    
    # Extract from <answer> tags
    match = re.search(r"<answer>(.*?)</answer>", prediction)
    extracted = match.group(1) if match else prediction
    
    correct = str(answer).strip().lower() == str(extracted).strip().lower()
    
    return {
        "acc_score": 1.0 if correct else 0.0,
        "correct": correct,
    }


def demo_basic_comparison():
    """Basic comparison of multiple models."""
    print("\n" + "-" * 70)
    print("Demo 1: Basic Model Comparison")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_files = create_model_outputs(tmp_path)
        
        # Judge each model
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = simple_judge
        
        results = {}
        
        print("\nJudging results:\n")
        
        for model_name, filepath, samples in model_files:
            judged = []
            for sample in samples:
                result = runner._judge_sample(sample, mock_task, simple_judge)
                judged.append(result)
            
            # Calculate metrics
            correct = sum(1 for j in judged if j["metrics"]["acc_score"] == 1.0)
            total = len(judged)
            accuracy = correct / total
            
            results[model_name] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "samples": judged,
            }
            
            print(f"{model_name:<15} {correct}/{total} correct ({accuracy:.1%})")
        
        # Ranking
        print("\nRanking:")
        ranked = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for i, (name, stats) in enumerate(ranked, 1):
            print(f"  {i}. {name} ({stats['accuracy']:.1%})")


def demo_agreement_analysis():
    """Analyze agreement between models."""
    print("\n" + "-" * 70)
    print("Demo 2: Model Agreement Analysis")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_files = create_model_outputs(tmp_path)
        
        # Judge all models
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = simple_judge
        
        # Organize by question
        by_question = defaultdict(dict)
        
        for model_name, filepath, samples in model_files:
            for sample in samples:
                qid = sample["doc_id"]
                result = runner._judge_sample(sample, mock_task, simple_judge)
                by_question[qid][model_name] = {
                    "correct": result["metrics"]["acc_score"] == 1.0,
                    "answer": sample["filtered_resps"],
                }
        
        # Analyze agreement
        print("\nAgreement by question:\n")
        print(f"{'QID':<5} {'All Agree':<12} {'Majority':<10} {'Disagree'}")
        print("-" * 50)
        
        for qid, model_results in sorted(by_question.items()):
            correct_count = sum(1 for r in model_results.values() if r["correct"])
            total = len(model_results)
            
            all_agree = (correct_count == 0) or (correct_count == total)
            majority_correct = correct_count > total / 2
            
            status = "✓" if all_agree else "✗"
            majority = "Yes" if majority_correct else "No"
            disagree = total - correct_count if majority_correct else correct_count
            
            print(f"{qid:<5} {status:<12} {majority:<10} {disagree} models")


def demo_error_analysis():
    """Detailed error analysis."""
    print("\n" + "-" * 70)
    print("Demo 3: Error Analysis")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_files = create_model_outputs(tmp_path)
        
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = simple_judge
        
        # Collect errors by category
        errors_by_model = defaultdict(list)
        
        for model_name, filepath, samples in model_files:
            for sample in samples:
                result = runner._judge_sample(sample, mock_task, simple_judge)
                
                if result["metrics"]["acc_score"] == 0.0:
                    # Analyze error type
                    resp = sample["filtered_resps"]
                    if "<answer>" not in resp:
                        error_type = "format_error"
                    elif "not sure" in resp.lower() or "might be" in resp.lower():
                        error_type = "uncertain"
                    else:
                        error_type = "factual_error"
                    
                    errors_by_model[model_name].append({
                        "qid": sample["doc_id"],
                        "question": sample["doc"]["question"],
                        "type": error_type,
                    })
        
        # Report
        print("\nError breakdown:\n")
        
        for model, errors in errors_by_model.items():
            print(f"{model} ({len(errors)} errors):")
            
            # Group by type
            by_type = defaultdict(list)
            for e in errors:
                by_type[e["type"]].append(e)
            
            for error_type, items in by_type.items():
                print(f"  {error_type}: {len(items)}")
                for item in items[:2]:  # Show first 2
                    print(f"    - Q{item['qid']}: {item['question']}")
                if len(items) > 2:
                    print(f"    ... and {len(items) - 2} more")


def demo_generate_report():
    """Generate a comprehensive comparison report."""
    print("\n" + "-" * 70)
    print("Demo 4: Generate Comparison Report")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        model_files = create_model_outputs(tmp_path)
        
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = simple_judge
        
        # Judge all
        all_results = {}
        
        for model_name, filepath, samples in model_files:
            judged = []
            for sample in samples:
                result = runner._judge_sample(sample, mock_task, simple_judge)
                judged.append(result)
            
            correct = sum(1 for j in judged if j["metrics"]["acc_score"] == 1.0)
            
            all_results[model_name] = {
                "accuracy": correct / len(judged),
                "correct": correct,
                "total": len(judged),
                "samples": judged,
            }
        
        # Generate report
        report = {
            "comparison_date": "2024-03-28",
            "task": "knowledge_qa",
            "metrics": ["accuracy"],
            "models": {},
            "ranking": [],
        }
        
        for name, stats in all_results.items():
            report["models"][name] = {
                "accuracy": round(stats["accuracy"], 4),
                "correct": stats["correct"],
                "total": stats["total"],
            }
        
        # Ranking
        ranked = sorted(
            report["models"].items(),
            key=lambda x: x[1]["accuracy"],
            reverse=True
        )
        report["ranking"] = [{"model": n, "rank": i+1, **s} for i, (n, s) in enumerate(ranked)]
        
        # Print report
        print("\nComparison Report:")
        print(json.dumps(report, indent=2))
        
        # Save to file
        report_file = tmp_path / "comparison_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Report saved to {report_file}")


def main():
    print("=" * 70)
    print("Example 5: Model Comparison")
    print("=" * 70)
    print()
    print("This example demonstrates comparing outputs from multiple")
    print("models on the same set of questions.")
    print()
    
    demo_basic_comparison()
    demo_agreement_analysis()
    demo_error_analysis()
    demo_generate_report()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Judge all models with same criteria for fair comparison")
    print("  • Analyze agreement to find controversial questions")
    print("  • Error analysis reveals model weaknesses")
    print("  • Generate reports for documentation")
    print("\nCLI for model comparison:")
    print("  # Judge multiple model outputs")
    print("  for f in results/model_*.jsonl; do")
    print("    lmms-eval judge -i $f -d judged/")
    print("  done")


if __name__ == "__main__":
    main()
