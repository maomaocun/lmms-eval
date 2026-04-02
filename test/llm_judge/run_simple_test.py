#!/usr/bin/env python
"""Simple test runner for llm_judge that doesn't require pytest."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.standalone import JudgeRunner
from lmms_eval.cli.judge_cmd import _detect_task_from_filename, _get_output_path


def test_basic():
    """Test basic functionality."""
    print("Testing basic JudgeRunner initialization...")
    
    runner = JudgeRunner(
        judge_mode="auto",
        judge_model="gpt-4o-mini",
        parallel=4,
    )
    
    assert runner.judge_mode == "auto"
    assert runner.judge_model == "gpt-4o-mini"
    assert runner.parallel == 4
    
    print("✓ Basic initialization passed")


def test_task_detection():
    """Test task name detection from filename."""
    print("Testing task detection...")
    
    assert _detect_task_from_filename("20240328_samples_mathvision_reason_testmini.jsonl") == "mathvision_reason_testmini"
    assert _detect_task_from_filename("samples_wemath_testmini.jsonl") == "wemath_testmini"
    
    print("✓ Task detection passed")


def test_output_path():
    """Test output path determination."""
    print("Testing output path...")
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "input.jsonl"
        input_file.touch()
        
        # Test default suffix
        result = _get_output_path(input_file, None, None)
        assert result.name == "input_judged.jsonl"
        
        # Test explicit output
        result = _get_output_path(input_file, str(tmp_path / "output.jsonl"), None)
        assert result.name == "output.jsonl"
        
        # Test output dir
        result = _get_output_path(input_file, None, str(tmp_path / "output_dir"))
        assert result.name == "input.jsonl"
        assert "output_dir" in str(result)
    
    print("✓ Output path passed")


def test_jsonl_io():
    """Test JSONL loading and saving."""
    print("Testing JSONL I/O...")
    
    runner = JudgeRunner()
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create test file
        test_file = tmp_path / "test.jsonl"
        samples = [
            {"doc_id": 0, "doc": {"question": "Q1", "answer": "A1"}, "filtered_resps": "R1"},
            {"doc_id": 1, "doc": {"question": "Q2", "answer": "A2"}, "filtered_resps": "R2"},
        ]
        
        # Save using runner
        runner.save_results(samples, test_file)
        
        # Load using runner
        loaded = runner._load_jsonl(test_file)
        
        assert len(loaded) == 2
        assert loaded[0]["doc_id"] == 0
        assert loaded[1]["doc"]["answer"] == "A2"
    
    print("✓ JSONL I/O passed")


def test_needs_llm_judge():
    """Test detection of when LLM judge is needed."""
    print("Testing needs_llm_judge detection...")
    
    runner = JudgeRunner()
    
    # Should need LLM judge
    assert runner._needs_llm_judge({"acc_score": 0}) is True
    assert runner._needs_llm_judge({"accuracy": 0.0}) is True
    
    # Should not need LLM judge
    assert runner._needs_llm_judge({"acc_score": 1.0}) is False
    assert runner._needs_llm_judge({"accuracy": 0.9}) is False
    
    print("✓ needs_llm_judge detection passed")


def test_extract_question():
    """Test question extraction."""
    print("Testing question extraction...")
    
    runner = JudgeRunner()
    
    assert runner._extract_question({"question": "Q1"}) == "Q1"
    assert runner._extract_question({"query_wo": "Q2"}) == "Q2"
    
    print("✓ Question extraction passed")


def test_clean_for_json():
    """Test JSON cleaning."""
    print("Testing JSON cleaning...")
    
    runner = JudgeRunner()
    
    # Test basic types
    assert runner._clean_for_json(123) == 123
    assert runner._clean_for_json("str") == "str"
    
    # Test nested dict
    data = {"a": 1, "b": [1, 2, 3]}
    result = runner._clean_for_json(data)
    assert result == data
    
    # Test non-serializable (MagicMock)
    obj = {"key": MagicMock()}
    result = runner._clean_for_json(obj)
    assert isinstance(result["key"], str)
    
    print("✓ JSON cleaning passed")


def test_judge_sample():
    """Test judging a single sample."""
    print("Testing judge_sample...")
    
    runner = JudgeRunner(judge_mode="rule")
    
    # Create mock task
    mock_task = MagicMock()
    mock_task.config.process_results = MagicMock(return_value={
        "acc_score": 1.0,
        "format_score": 1.0,
    })
    
    sample = {
        "doc_id": 0,
        "doc": {"question": "Q", "answer": "A"},
        "filtered_resps": "Response",
    }
    
    result = runner._judge_sample(sample, mock_task, mock_task.config.process_results)
    
    assert "metrics" in result
    assert result["judge_mode"] == "rule"
    assert result["metrics"]["acc_score"] == 1.0
    
    print("✓ judge_sample passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running llm_judge tests")
    print("=" * 60)
    print()
    
    tests = [
        test_basic,
        test_task_detection,
        test_output_path,
        test_jsonl_io,
        test_needs_llm_judge,
        test_extract_question,
        test_clean_for_json,
        test_judge_sample,
    ]
    
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    if failed == 0:
        print(f"All {len(tests)} tests passed!")
    else:
        print(f"{failed}/{len(tests)} tests failed")
    print("=" * 60)
    
    return failed


if __name__ == "__main__":
    sys.exit(main())
