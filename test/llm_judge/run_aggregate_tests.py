#!/usr/bin/env python
"""Simple test runner for aggregate tests (without pytest dependency)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lmms_eval.llm_judge.aggregator import Aggregator


def test_init():
    """Test Aggregator initialization."""
    aggregator = Aggregator()
    assert aggregator._task_manager is None
    assert aggregator._cache == {}
    print("✅ test_init passed")


def test_get_special_config_wemath():
    """Test detection of WeMath special aggregation."""
    aggregator = Aggregator()
    config = aggregator._get_special_config("wemath_testmini_reasoning")
    assert config is not None
    assert "wemath" in config["module"]
    assert "loose_func" in config
    assert "strict_func" in config
    print("✅ test_get_special_config_wemath passed")


def test_get_special_config_mathvision():
    """Test detection of MathVision special aggregation."""
    aggregator = Aggregator()
    config = aggregator._get_special_config("mathvision_test")
    assert config is not None
    assert "mathvision" in config["module"]
    assert "accuracy_func" in config
    print("✅ test_get_special_config_mathvision passed")


def test_get_special_config_generic():
    """Test that generic tasks return None."""
    aggregator = Aggregator()
    config = aggregator._get_special_config("some_generic_task")
    assert config is None
    print("✅ test_get_special_config_generic passed")


def test_aggregate_generic_task():
    """Test aggregation for generic tasks."""
    aggregator = Aggregator()
    generic_samples = [
        {"doc_id": 0, "metrics": {"accuracy": 0.8, "score": 1.0}},
        {"doc_id": 1, "metrics": {"accuracy": 0.6, "score": 0.0}},
        {"doc_id": 2, "metrics": {"accuracy": 0.9, "score": 1.0}},
    ]
    results = aggregator.aggregate(generic_samples, "generic_task")
    assert "accuracy" in results
    assert "score" in results
    assert abs(results["accuracy"] - 0.7667) < 0.001
    assert abs(results["score"] - 0.6667) < 0.001
    print("✅ test_aggregate_generic_task passed")


def test_aggregate_generic_task_with_metric_filter():
    """Test aggregation with specific metric filter."""
    aggregator = Aggregator()
    generic_samples = [
        {"doc_id": 0, "metrics": {"accuracy": 0.8, "score": 1.0}},
        {"doc_id": 1, "metrics": {"accuracy": 0.6, "score": 0.0}},
    ]
    results = aggregator.aggregate(generic_samples, "generic_task", metric_name="accuracy")
    assert "accuracy" in results
    assert "score" not in results
    print("✅ test_aggregate_generic_task_with_metric_filter passed")


def test_aggregate_empty_samples():
    """Test aggregation with empty samples."""
    aggregator = Aggregator()
    results = aggregator.aggregate([], "generic_task")
    assert results == {}
    print("✅ test_aggregate_empty_samples passed")


def test_aggregate_with_nested_metrics():
    """Test aggregation handles nested metric structures correctly."""
    aggregator = Aggregator()
    samples = [
        {"doc_id": 0, "metrics": {"accuracy": 1.0, "nested": {"value": 1}}},
        {"doc_id": 1, "metrics": {"accuracy": 0.0}},
    ]
    results = aggregator.aggregate(samples, "generic_task")
    assert "accuracy" in results
    assert "nested" not in results
    print("✅ test_aggregate_with_nested_metrics passed")


def test_wemath_data_extraction():
    """Test extraction of WeMath data from samples."""
    aggregator = Aggregator()
    config = aggregator._get_special_config("wemath_testmini_reasoning")
    
    wemath_samples = [
        {
            "doc_id": 0,
            "wemath_loose": {
                "ID": "test_001",
                "key": "2steps_1",
                "acc_score": 1.0,
            },
        },
        {
            "doc_id": 1,
            "wemath_loose": {
                "ID": "test_001",
                "key": "2steps_2",
                "acc_score": 1.0,
            },
        },
    ]
    
    data_key = config["data_key"]
    extracted_data = []
    for sample in wemath_samples:
        if data_key in sample:
            extracted_data.append(sample[data_key])
    
    assert len(extracted_data) == 2
    assert all("ID" in d for d in extracted_data)
    assert all("key" in d for d in extracted_data)
    print("✅ test_wemath_data_extraction passed")


def test_mathvision_data_extraction():
    """Test extraction of MathVision data from samples."""
    aggregator = Aggregator()
    config = aggregator._get_special_config("mathvision_test")
    
    mathvision_samples = [
        {
            "doc_id": 0,
            "mathvision_standard_eval": {
                "response": ["Answer 1"],
                "scores": [True],
            },
        },
        {
            "doc_id": 1,
            "mathvision_standard_eval": {
                "response": ["Answer 2"],
                "scores": [False],
            },
        },
    ]
    
    data_key = config["data_key"]
    extracted_data = []
    for sample in mathvision_samples:
        if data_key in sample:
            extracted_data.append(sample[data_key])
    
    assert len(extracted_data) == 2
    assert all("scores" in d for d in extracted_data)
    print("✅ test_mathvision_data_extraction passed")


def simple_load_jsonl(path):
    """Simple JSONL loader for testing."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                pass  # Skip invalid lines
    return samples


def test_load_jsonl(tmp_path=None):
    """Test loading JSONL file."""
    if tmp_path is None:
        tmp_path = Path("/tmp")
    
    file_path = tmp_path / "test_samples.jsonl"
    
    # Create test file
    samples = [
        {"doc_id": 0, "metrics": {"accuracy": 1.0}},
        {"doc_id": 1, "metrics": {"accuracy": 0.0}},
    ]
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    loaded_samples = simple_load_jsonl(file_path)
    assert len(loaded_samples) == 2
    assert loaded_samples[0]["doc_id"] == 0
    
    # Cleanup
    file_path.unlink()
    print("✅ test_load_jsonl passed")


def test_load_jsonl_invalid_json(tmp_path=None):
    """Test loading file with invalid JSON."""
    if tmp_path is None:
        tmp_path = Path("/tmp")
    
    file_path = tmp_path / "invalid.jsonl"
    
    with open(file_path, "w") as f:
        f.write('{"valid": true}\n')
        f.write('invalid json\n')
        f.write('{"valid": false}\n')
    
    samples = simple_load_jsonl(file_path)
    assert len(samples) == 2  # One line should be skipped
    
    file_path.unlink()
    print("✅ test_load_jsonl_invalid_json passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Aggregate Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_init,
        test_get_special_config_wemath,
        test_get_special_config_mathvision,
        test_get_special_config_generic,
        test_aggregate_generic_task,
        test_aggregate_generic_task_with_metric_filter,
        test_aggregate_empty_samples,
        test_aggregate_with_nested_metrics,
        test_wemath_data_extraction,
        test_mathvision_data_extraction,
        test_load_jsonl,
        test_load_jsonl_invalid_json,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
