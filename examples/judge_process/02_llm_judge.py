#!/usr/bin/env python
"""
Example 2: LLM-as-Judge

This example shows how to use an LLM (like GPT-4o) as a judge.
This is useful when:
1. Rule-based judging is insufficient
2. You need semantic understanding
3. Questions have multiple valid answers

Note: This example requires JUDGE_API_KEY to be set.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from unittest.mock import MagicMock


def create_open_ended_questions(output_path: Path, num_samples: int = 5):
    """Create open-ended questions that benefit from LLM judging."""
    print(f"Creating {num_samples} open-ended questions...")
    
    # These questions have multiple valid ways to answer
    questions = [
        {
            "question": "Explain the water cycle in simple terms.",
            "answer": "Water evaporates from oceans, forms clouds, falls as rain, and flows back to oceans.",
            "prediction": "The water cycle involves evaporation of water from the Earth's surface, condensation into clouds, precipitation as rain or snow, and collection in bodies of water before repeating.",
        },
        {
            "question": "What are the main causes of climate change?",
            "answer": "Greenhouse gas emissions from burning fossil fuels, deforestation, and industrial processes.",
            "prediction": "Climate change is primarily driven by human activities like burning coal and oil, cutting down forests, and manufacturing goods that release CO2.",
        },
        {
            "question": "How does photosynthesis work?",
            "answer": "Plants convert sunlight, water, and CO2 into glucose and oxygen.",
            "prediction": "Plants use chlorophyll to capture light energy, which they combine with water and carbon dioxide to produce sugar for food and release oxygen as a byproduct.",
        },
        {
            "question": "What is the Pythagorean theorem?",
            "answer": "a² + b² = c² for right triangles.",
            "prediction": "In a right-angled triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
        },
        {
            "question": "Why is the sky blue?",
            "answer": "Rayleigh scattering scatters shorter blue wavelengths more than other colors.",
            "prediction": "Blue light has shorter wavelengths that get scattered in all directions by gas molecules in the atmosphere, making the sky appear blue.",
        },
    ]
    
    samples = []
    for i, q in enumerate(questions[:num_samples]):
        sample = {
            "doc_id": i,
            "doc": {
                "question": q["question"],
                "answer": q["answer"],
                "question_type": "open_ended",
            },
            "filtered_resps": q["prediction"],
            "target": q["answer"],
            "input": q["question"],
        }
        samples.append(sample)
    
    # Write to JSONL
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"✓ Created {output_path}")
    return samples


def simple_rule_judge(doc: dict, results: list) -> dict:
    """
    Simple rule-based judge that will fail for open-ended questions.
    This demonstrates why LLM judge is needed.
    """
    prediction = results[0] if results else ""
    answer = doc.get("answer", "")
    
    # Simple string matching (will fail for semantically equivalent answers)
    exact_match = str(answer).strip().lower() == str(prediction).strip().lower()
    
    # Check for keyword overlap
    answer_words = set(str(answer).lower().split())
    pred_words = set(str(prediction).lower().split())
    common_words = answer_words & pred_words
    keyword_overlap = len(common_words) / max(len(answer_words), 1)
    
    return {
        "acc_score": 1.0 if exact_match else 0.0,  # Will be 0 for most
        "keyword_overlap": round(keyword_overlap, 2),
        "exact_match": exact_match,
    }


def demo_rule_vs_llm():
    """Compare rule-based vs LLM judging."""
    print("\n" + "-" * 70)
    print("Comparing Rule-based vs LLM Judging")
    print("-" * 70)
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create sample data
        input_file = tmp_path / "open_ended_questions.jsonl"
        create_open_ended_questions(input_file, num_samples=5)
        
        # Load samples
        with open(input_file) as f:
            samples = [json.loads(line) for line in f]
        
        # Create mock task
        mock_task = MagicMock()
        
        # Test 1: Rule-based judging
        print("\n1. Rule-based Judging:")
        print("   (Simple string matching - expected to perform poorly)")
        
        mock_task.config.process_results = simple_rule_judge
        runner = JudgeRunner(judge_mode="rule")
        
        rule_results = []
        for sample in samples:
            result = runner._judge_sample(sample, mock_task, simple_rule_judge)
            rule_results.append(result)
        
        rule_correct = sum(1 for r in rule_results if r["metrics"]["acc_score"] == 1.0)
        print(f"   Results: {rule_correct}/{len(rule_results)} marked correct")
        print(f"   Note: Low score because semantic equivalence not captured")
        
        # Show details
        for i, r in enumerate(rule_results):
            print(f"   Sample {i}: exact_match={r['metrics']['exact_match']}, "
                  f"overlap={r['metrics']['keyword_overlap']}")
        
        # Test 2: LLM-based judging (if API key available)
        print("\n2. LLM-based Judging:")
        
        api_key = os.getenv("JUDGE_API_KEY")
        if not api_key:
            print("   ⚠ JUDGE_API_KEY not set. Skipping LLM judge demo.")
            print("   To enable, run: export JUDGE_API_KEY=sk-your-key")
            return
        
        print(f"   Using model: {os.getenv('JUDGE_MODEL', 'gpt-4o-mini')}")
        
        try:
            llm_runner = JudgeRunner(
                judge_mode="llm",
                judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
                judge_api_key=api_key,
            )
            
            llm_results = []
            for sample in samples[:2]:  # Test first 2 to save API calls
                print(f"   Judging sample {sample['doc_id']}...", end=" ")
                
                # Use dummy process_results (LLM judge doesn't need it)
                dummy_fn = lambda doc, res: {"acc_score": 0.0}
                mock_task.config.process_results = dummy_fn
                
                result = llm_runner._judge_sample(sample, mock_task, dummy_fn)
                llm_results.append(result)
                
                score = result["metrics"].get("llm_judge_score", "N/A")
                print(f"score={score}")
            
            llm_correct = sum(
                1 for r in llm_results 
                if r["metrics"].get("llm_judge_score") == 1
            )
            print(f"\n   Results: {llm_correct}/{len(llm_results)} marked correct")
            print("   ✓ Higher accuracy due to semantic understanding")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            print("   Make sure your API key is valid")


def demo_auto_mode():
    """Demonstrate auto mode (rule first, then LLM fallback)."""
    print("\n" + "-" * 70)
    print("Auto Mode: Rule first, LLM fallback")
    print("-" * 70)
    
    api_key = os.getenv("JUDGE_API_KEY")
    if not api_key:
        print("⚠ Skipping (JUDGE_API_KEY not set)")
        return
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create mixed samples (some exact match, some need semantic)
        samples = [
            {
                "doc_id": 0,
                "doc": {"question": "2+2=?", "answer": "4"},
                "filtered_resps": "4",  # Exact match
            },
            {
                "doc_id": 1,
                "doc": {"question": "Capital of France?", "answer": "Paris"},
                "filtered_resps": "The capital is Paris.",  # Needs semantic
            },
        ]
        
        input_file = tmp_path / "mixed.jsonl"
        with open(input_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        
        print("\nSample 0: Exact match (rule-based should work)")
        print("Sample 1: Semantic match (needs LLM fallback)")
        
        # Auto mode
        runner = JudgeRunner(
            judge_mode="auto",
            judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
            judge_api_key=api_key,
        )
        
        def strict_judge(doc, results):
            correct = str(doc.get("answer")) == str(results[0])
            return {"acc_score": 1.0 if correct else 0.0}
        
        mock_task = MagicMock()
        mock_task.config.process_results = strict_judge
        
        with open(input_file) as f:
            loaded = [json.loads(line) for line in f]
        
        for sample in loaded:
            result = runner._judge_sample(sample, mock_task, strict_judge)
            mode = result["judge_mode"]
            score = result["metrics"].get("llm_judge_score") or result["metrics"]["acc_score"]
            print(f"Sample {sample['doc_id']}: mode={mode}, score={score}")


def main():
    print("=" * 70)
    print("Example 2: LLM-as-Judge")
    print("=" * 70)
    print()
    print("This example demonstrates using an LLM (like GPT-4o) to judge")
    print("model outputs, especially for open-ended questions where rule-based")
    print("judging fails to capture semantic equivalence.")
    print()
    
    # Check for API key
    api_key = os.getenv("JUDGE_API_KEY")
    if api_key:
        print(f"✓ JUDGE_API_KEY is set (model: {os.getenv('JUDGE_MODEL', 'gpt-4o-mini')})")
    else:
        print("⚠ JUDGE_API_KEY not set - LLM judge demo will be skipped")
        print("  To enable: export JUDGE_API_KEY=sk-your-api-key")
    print()
    
    # Run demos
    demo_rule_vs_llm()
    demo_auto_mode()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Rule-based judging works for exact matches")
    print("  • LLM judging understands semantic equivalence")
    print("  • Auto mode combines efficiency and accuracy")
    print("\nCLI usage:")
    print("  export JUDGE_API_KEY=sk-...")
    print("  lmms-eval judge -i results.jsonl --judge-mode llm")


if __name__ == "__main__":
    main()
