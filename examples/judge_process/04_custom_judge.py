#!/usr/bin/env python
"""
Example 4: Custom Judge Logic

This example shows how to implement custom judging logic for specific needs,
such as partial credit, multi-dimensional scoring, or domain-specific criteria.
"""

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from unittest.mock import MagicMock


def create_code_samples(output_path: Path):
    """Create code generation samples for evaluation."""
    print("Creating code generation samples...\n")
    
    samples = [
        {
            "doc_id": 0,
            "doc": {
                "question": "Write a Python function to calculate factorial.",
                "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
                "test_cases": [("factorial(5)", "120"), ("factorial(0)", "1")],
                "criteria": ["correctness", "efficiency", "style"],
            },
            "filtered_resps": '''```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```''',
        },
        {
            "doc_id": 1,
            "doc": {
                "question": "Implement a function to check if a string is a palindrome.",
                "answer": "def is_palindrome(s): return s == s[::-1]",
                "test_cases": [("is_palindrome('racecar')", "True"), ("is_palindrome('hello')", "False")],
                "criteria": ["correctness", "edge_cases", "style"],
            },
            "filtered_resps": '''Here's my solution:
```python
def is_palindrome(s):
    return s.lower() == s.lower()[::-1]
```''',
        },
        {
            "doc_id": 2,
            "doc": {
                "question": "Write a function to find the maximum element in a list.",
                "answer": "def find_max(lst): return max(lst)",
                "test_cases": [("find_max([1,3,2])", "3"), ("find_max([-1,-5,-2])", "-1")],
                "criteria": ["correctness", "efficiency"],
            },
            "filtered_resps": '''```python
def find_max(lst):
    max_val = lst[0]
    for x in lst:
        if x > max_val:
            max_val = x
    return max_val
```''',
        },
    ]
    
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"✓ Created {len(samples)} code samples\n")
    return samples


def multidimensional_judge(doc: dict, results: list) -> dict:
    """
    Multi-dimensional judging for code quality.
    
    Scores on multiple criteria:
    - correctness: Does it produce correct output?
    - completeness: Does it handle all requirements?
    - style: Code formatting and documentation
    - efficiency: Algorithmic complexity
    """
    prediction = results[0] if results else ""
    
    # Extract code from markdown blocks
    code_match = re.search(r"```python\n(.*?)\n```", prediction, re.DOTALL)
    code = code_match.group(1) if code_match else prediction
    
    metrics = {}
    
    # 1. Correctness (0 or 1)
    # Check if code contains key elements from reference answer
    answer = doc.get("answer", "")
    key_elements = set(re.findall(r"\b\w+\b", answer.lower()))
    code_elements = set(re.findall(r"\b\w+\b", code.lower()))
    overlap = len(key_elements & code_elements)
    metrics["correctness"] = min(1.0, overlap / max(len(key_elements) * 0.5, 1))
    
    # 2. Style (0 to 1)
    style_score = 0.0
    if "def " in code:  # Has function definition
        style_score += 0.3
    if re.search(r"^    |^\t", code, re.MULTILINE):  # Has indentation
        style_score += 0.3
    if "return" in code:  # Has return statement
        style_score += 0.2
    if len(code.split("\n")) >= 3:  # Reasonable length
        style_score += 0.2
    metrics["style"] = style_score
    
    # 3. Efficiency (0 to 1)
    # Simple heuristic: shorter, cleaner code is better
    lines = [l for l in code.split("\n") if l.strip()]
    metrics["efficiency"] = max(0.0, 1.0 - (len(lines) - 3) * 0.1)
    
    # 4. Documentation
    has_docstring = '"""' in code or "'''" in code or "#" in code
    metrics["documentation"] = 1.0 if has_docstring else 0.0
    
    # Overall score (weighted average)
    weights = {"correctness": 0.5, "style": 0.2, "efficiency": 0.2, "documentation": 0.1}
    overall = sum(metrics[k] * weights[k] for k in weights)
    
    return {
        "acc_score": overall,
        "correctness": round(metrics["correctness"], 2),
        "style": round(metrics["style"], 2),
        "efficiency": round(metrics["efficiency"], 2),
        "documentation": round(metrics["documentation"], 2),
    }


def partial_credit_judge(doc: dict, results: list) -> dict:
    """
    Judge with partial credit for math problems.
    
    Awards partial points for:
    - Correct approach but wrong answer
    - Right answer but wrong format
    - Partial solution
    """
    prediction = results[0] if results else ""
    answer = doc.get("answer", "")
    
    # Extract numeric answer
    pred_nums = re.findall(r"-?\d+\.?\d*", prediction)
    answer_nums = re.findall(r"-?\d+\.?\d*", str(answer))
    
    scores = {
        "format": 0.0,
        "approach": 0.0,
        "calculation": 0.0,
    }
    
    # Format check (has reasoning/work shown)
    if any(tag in prediction for tag in ["<think>", "<analysis>", "Step", "Solution"]):
        scores["format"] = 1.0
    
    # Approach check (has relevant keywords)
    # This is simplified - real implementation would use more sophisticated checks
    scores["approach"] = 0.5  # Assume partial credit for having any response
    
    # Calculation check
    if pred_nums and answer_nums:
        if pred_nums[-1] == answer_nums[-1]:  # Last number matches
            scores["calculation"] = 1.0
        elif any(p == answer_nums[-1] for p in pred_nums):  # Any number matches
            scores["calculation"] = 0.5
    
    # Calculate weighted total
    total = scores["format"] * 0.2 + scores["approach"] * 0.3 + scores["calculation"] * 0.5
    
    return {
        "acc_score": round(total, 2),
        "format_score": scores["format"],
        "approach_score": scores["approach"],
        "calculation_score": scores["calculation"],
        "extracted_numbers": pred_nums,
    }


def demo_multidimensional():
    """Demonstrate multi-dimensional judging."""
    print("-" * 70)
    print("Demo 1: Multi-dimensional Code Evaluation")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create samples
        input_file = tmp_path / "code_samples.jsonl"
        samples = create_code_samples(input_file)
        
        # Judge with custom function
        runner = JudgeRunner(judge_mode="rule")
        mock_task = MagicMock()
        mock_task.config.process_results = multidimensional_judge
        
        print("\nJudging samples with multi-dimensional criteria:\n")
        
        judged = []
        for sample in samples:
            result = runner._judge_sample(sample, mock_task, multidimensional_judge)
            judged.append(result)
            
            print(f"Sample {sample['doc_id']}: {sample['doc']['question'][:40]}...")
            m = result["metrics"]
            print(f"  Overall: {m['acc_score']:.2f}")
            print(f"    - Correctness: {m['correctness']:.2f}")
            print(f"    - Style: {m['style']:.2f}")
            print(f"    - Efficiency: {m['efficiency']:.2f}")
            print(f"    - Documentation: {m['documentation']:.2f}")


def demo_partial_credit():
    """Demonstrate partial credit judging."""
    print("\n" + "-" * 70)
    print("Demo 2: Partial Credit for Math Problems")
    print("-" * 70)
    
    # Create math samples with varying quality
    math_samples = [
        {
            "doc_id": 0,
            "doc": {"question": "2+2=?", "answer": "4"},
            "filtered_resps": "The answer is 4.",  # Perfect
        },
        {
            "doc_id": 1,
            "doc": {"question": "10/2=?", "answer": "5"},
            "filtered_resps": "<think>Dividing 10 by 2 gives 4.</think><answer>4</answer>",  # Wrong answer, good format
        },
        {
            "doc_id": 2,
            "doc": {"question": "3*4=?", "answer": "12"},
            "filtered_resps": "12",  # Right answer, no work shown
        },
        {
            "doc_id": 3,
            "doc": {"question": "15-7=?", "answer": "8"},
            "filtered_resps": "<think>Starting from 15...</think><answer>9</answer>",  # Close, good format
        },
    ]
    
    runner = JudgeRunner(judge_mode="rule")
    mock_task = MagicMock()
    mock_task.config.process_results = partial_credit_judge
    
    print("\nSample outputs with partial credit:\n")
    print(f"{'ID':<5} {'Total':<8} {'Format':<8} {'Approach':<10} {'Calc':<8}")
    print("-" * 50)
    
    for sample in math_samples:
        result = runner._judge_sample(sample, mock_task, partial_credit_judge)
        m = result["metrics"]
        print(f"{sample['doc_id']:<5} {m['acc_score']:<8.2f} {m['format_score']:<8.1f} "
              f"{m['approach_score']:<10.1f} {m['calculation_score']:<8.1f}")
        print(f"      Answer: {sample['filtered_resps'][:50]}...")
        print()


def demo_confidence_scoring():
    """Demonstrate confidence-based scoring."""
    print("-" * 70)
    print("Demo 3: Confidence-based Judging")
    print("-" * 70)
    
    def confidence_judge(doc: dict, results: list) -> dict:
        """Judge with confidence score based on answer characteristics."""
        prediction = results[0] if results else ""
        answer = doc.get("answer", "")
        
        correct = str(answer).strip().lower() in str(prediction).strip().lower()
        
        # Confidence factors
        confidence_factors = []
        
        # Factor 1: Explicit answer format
        if re.search(r"<answer>|the answer is|answer:\s*", prediction, re.I):
            confidence_factors.append(("explicit_format", 0.3))
        
        # Factor 2: Reasoning shown
        if any(tag in prediction for tag in ["<think>", "because", "since", "as"]):
            confidence_factors.append(("reasoning", 0.3))
        
        # Factor 3: Short answer (less likely to be wrong)
        words = len(prediction.split())
        if 1 <= words <= 10:
            confidence_factors.append(("concise", 0.2))
        
        # Factor 4: Matches exactly
        if correct:
            confidence_factors.append(("correct", 0.2))
        
        confidence = sum(c for _, c in confidence_factors)
        
        return {
            "acc_score": 1.0 if correct else 0.0,
            "confidence": round(confidence, 2),
            "confidence_factors": [f for f, _ in confidence_factors],
            "word_count": words,
        }
    
    samples = [
        {"doc_id": 0, "doc": {"answer": "Paris"}, "filtered_resps": "Paris"},  # Low confidence
        {"doc_id": 1, "doc": {"answer": "Paris"}, "filtered_resps": "The answer is Paris."},  # Medium
        {"doc_id": 2, "doc": {"answer": "Paris"}, "filtered_resps": "<think>France's capital...</think><answer>Paris</answer>"},  # High
    ]
    
    runner = JudgeRunner(judge_mode="rule")
    mock_task = MagicMock()
    mock_task.config.process_results = confidence_judge
    
    print("\nConfidence scoring:\n")
    print(f"{'ID':<5} {'Correct':<10} {'Confidence':<12} {'Factors'}")
    print("-" * 60)
    
    for sample in samples:
        result = runner._judge_sample(sample, mock_task, confidence_judge)
        m = result["metrics"]
        factors = ", ".join(m["confidence_factors"])
        print(f"{sample['doc_id']:<5} {'Yes' if m['acc_score'] else 'No':<10} "
              f"{m['confidence']:<12} {factors}")


def main():
    print("=" * 70)
    print("Example 4: Custom Judge Logic")
    print("=" * 70)
    print()
    print("This example shows how to implement custom judging logic")
    print("for specific domains and requirements.")
    print()
    
    demo_multidimensional()
    demo_partial_credit()
    demo_confidence_scoring()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Multi-dimensional scoring gives detailed feedback")
    print("  • Partial credit rewards effort and progress")
    print("  • Confidence scores help identify uncertain predictions")
    print("\nTo use custom judging:")
    print("  1. Define your judging function")
    print("  2. Configure it in the task YAML, or")
    print("  3. Pass it directly to JudgeRunner")


if __name__ == "__main__":
    main()
