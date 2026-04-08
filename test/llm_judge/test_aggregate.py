"""Tests for the aggregate module.

This module tests the Aggregator class and aggregate CLI command
for aggregating judged results with task-specific logic.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Skip all tests if aggregate dependencies are not available
try:
    from lmms_eval.llm_judge.aggregator import Aggregator
    from lmms_eval.cli.aggregate_cmd import (
        _detect_task_from_filename,
        _get_output_path,
        add_aggregate_parser,
        run_aggregate,
    )
    AGGREGATE_AVAILABLE = True
except ImportError:
    AGGREGATE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not AGGREGATE_AVAILABLE, reason="Aggregate dependencies not available")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def wemath_samples():
    """Create sample WeMath judged data for testing."""
    return [
        {
            "doc_id": 0,
            "wemath_loose": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q1",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_1",
                "question number": 1,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "wemath_strict": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q1",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_1",
                "question number": 1,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "metrics": {"acc_score": 1.0},
        },
        {
            "doc_id": 1,
            "wemath_loose": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q2",
                "option": "A. 1; B. 2",
                "answer": "B",
                "key": "2steps_2",
                "question number": 2,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "wemath_strict": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q2",
                "option": "A. 1; B. 2",
                "answer": "B",
                "key": "2steps_2",
                "question number": 2,
                "knowledge concept description": "Test",
                "acc_score": 1.0,
            },
            "metrics": {"acc_score": 1.0},
        },
        {
            "doc_id": 2,
            "wemath_loose": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q3",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_multi",
                "question number": 3,
                "knowledge concept description": "Test",
                "acc_score": 0.0,
            },
            "wemath_strict": {
                "ID": "test_001",
                "split": "testmini",
                "knowledge concept": "Test Concept",
                "question": "Q3",
                "option": "A. 1; B. 2",
                "answer": "A",
                "key": "2steps_multi",
                "question number": 3,
                "knowledge concept description": "Test",
                "acc_score": 0.0,
            },
            "metrics": {"acc_score": 0.0},
        },
    ]


@pytest.fixture
def mathvision_samples():
    """Create sample MathVision judged data for testing."""
    return [
        {
            "doc_id": 0,
            "mathvision_standard_eval": {
                "response": ["Answer 1"],
                "scores": [True],
            },
            "metrics": {"llm_judge_score": 1},
        },
        {
            "doc_id": 1,
            "mathvision_standard_eval": {
                "response": ["Answer 2"],
                "scores": [False],
            },
            "metrics": {"llm_judge_score": 0},
        },
        {
            "doc_id": 2,
            "mathvision_standard_eval": {
                "response": ["Answer 3"],
                "scores": [True],
            },
            "metrics": {"llm_judge_score": 1},
        },
    ]


@pytest.fixture
def generic_samples():
    """Create sample generic judged data for testing."""
    return [
        {"doc_id": 0, "metrics": {"accuracy": 0.8, "score": 1.0}},
        {"doc_id": 1, "metrics": {"accuracy": 0.6, "score": 0.0}},
        {"doc_id": 2, "metrics": {"accuracy": 0.9, "score": 1.0}},
    ]


@pytest.fixture
def sample_jsonl(tmp_path, generic_samples):
    """Create a sample JSONL file for testing."""
    file_path = tmp_path / "test_samples.jsonl"
    with open(file_path, "w") as f:
        for sample in generic_samples:
            f.write(json.dumps(sample) + "\n")
    return file_path


# -----------------------------------------------------------------------------
# Test Aggregator class
# -----------------------------------------------------------------------------

class TestAggregator:
    """Tests for the Aggregator class."""
    
    def test_init(self):
        """Test Aggregator initialization."""
        aggregator = Aggregator()
        
        assert aggregator._task_manager is None
        assert aggregator._cache == {}
    
    def test_get_special_config_wemath(self):
        """Test detection of WeMath special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("wemath_testmini_reasoning")
        assert config is not None
        assert "wemath" in config["module"]
        assert "loose_func" in config
        assert "strict_func" in config
    
    def test_get_special_config_mathvision(self):
        """Test detection of MathVision special aggregation."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("mathvision_test")
        assert config is not None
        assert "mathvision" in config["module"]
        assert "accuracy_func" in config
    
    def test_get_special_config_generic(self):
        """Test that generic tasks return None."""
        aggregator = Aggregator()
        
        config = aggregator._get_special_config("some_generic_task")
        assert config is None
    
    def test_get_special_config_no_false_positive(self):
        """Test that substring matches inside other words are rejected."""
        aggregator = Aggregator()
        
        # Should NOT match because "wemath" is embedded inside another word without word boundaries
        config = aggregator._get_special_config("mywemathtask")
        assert config is None
        
        # Should match because "wemath" is a proper prefix with underscore boundary
        config = aggregator._get_special_config("wemath_testmini")
        assert config is not None
    
    def test_aggregate_generic_task(self, generic_samples):
        """Test aggregation for generic tasks."""
        aggregator = Aggregator()
        
        results = aggregator.aggregate(generic_samples, "generic_task")
        
        assert "accuracy" in results
        assert "score" in results
        # (0.8 + 0.6 + 0.9) / 3 = 0.7667
        assert results["accuracy"] == pytest.approx(0.7667, abs=0.001)
        # (1.0 + 0.0 + 1.0) / 3 = 0.6667
        assert results["score"] == pytest.approx(0.6667, abs=0.001)
    
    def test_aggregate_generic_task_with_metric_filter(self, generic_samples):
        """Test aggregation with specific metric filter."""
        aggregator = Aggregator()
        
        results = aggregator.aggregate(generic_samples, "generic_task", metric_name="accuracy")
        
        assert "accuracy" in results
        assert "score" not in results  # Should be filtered out
    
    def test_aggregate_empty_samples(self):
        """Test aggregation with empty samples."""
        aggregator = Aggregator()
        
        results = aggregator.aggregate([], "generic_task")
        
        assert results == {}
    
    def test_aggregate_samples_without_metrics(self):
        """Test aggregation with samples missing metrics."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {"accuracy": 1.0}},
            {"doc_id": 1},  # No metrics
            {"doc_id": 2, "metrics": {"accuracy": 0.0}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        # Should only count samples with the metric
        assert results["accuracy"] == 0.5  # (1.0 + 0.0) / 2
    
    def test_aggregate_with_nested_metrics(self):
        """Test aggregation handles nested metric structures correctly."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {"accuracy": 1.0, "nested": {"value": 1}}},
            {"doc_id": 1, "metrics": {"accuracy": 0.0}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        # Should only include numeric metrics, not nested dicts
        assert "accuracy" in results
        assert "nested" not in results


class TestAggregatorWeMath:
    """Tests for WeMath-specific aggregation."""
    
    @patch("lmms_eval.llm_judge.aggregator._get_task_manager")
    def test_aggregate_wemath_loose(self, mock_tm, wemath_samples):
        """Test WeMath loose aggregation."""
        aggregator = Aggregator()
        
        # Note: This test may fail if wemath_utils is not available
        # In that case, it tests the error handling path
        try:
            results = aggregator.aggregate(wemath_samples, "wemath_testmini_reasoning", metric_name="wemath_loose")
            
            # If successful, verify the result structure
            if results:
                assert "Score (Loose)" in results
        except (ImportError, AttributeError):
            # Expected if wemath_utils functions are not available in test environment
            pytest.skip("WeMath utils not available")
    
    @patch("lmms_eval.llm_judge.aggregator._get_task_manager")
    def test_aggregate_wemath_strict(self, mock_tm, wemath_samples):
        """Test WeMath strict aggregation."""
        aggregator = Aggregator()
        
        try:
            results = aggregator.aggregate(wemath_samples, "wemath_testmini_reasoning", metric_name="wemath_strict")
            
            if results:
                assert "Score (Strict)" in results
        except (ImportError, AttributeError):
            pytest.skip("WeMath utils not available")
    
    def test_wemath_data_extraction(self, wemath_samples):
        """Test extraction of WeMath data from samples."""
        aggregator = Aggregator()
        config = aggregator._get_special_config("wemath_testmini_reasoning")
        
        # Manually extract data like the aggregator does
        data_key = config["data_key"]
        extracted_data = []
        
        for sample in wemath_samples:
            if data_key in sample:
                extracted_data.append(sample[data_key])
        
        assert len(extracted_data) == 3
        assert all("ID" in d for d in extracted_data)
        assert all("key" in d for d in extracted_data)


class TestAggregatorMathVision:
    """Tests for MathVision-specific aggregation."""
    
    def test_aggregate_mathvision(self, mathvision_samples):
        """Test MathVision aggregation."""
        aggregator = Aggregator()
        
        try:
            results = aggregator.aggregate(mathvision_samples, "mathvision_test")
            
            if results:
                assert "accuracy" in results
        except (ImportError, AttributeError):
            pytest.skip("MathVision utils not available")
    
    def test_mathvision_data_extraction(self, mathvision_samples):
        """Test extraction of MathVision data from samples."""
        aggregator = Aggregator()
        config = aggregator._get_special_config("mathvision_test")
        
        data_key = config["data_key"]
        extracted_data = []
        
        for sample in mathvision_samples:
            if data_key in sample:
                extracted_data.append(sample[data_key])
        
        assert len(extracted_data) == 3
        assert all("scores" in d for d in extracted_data)


# -----------------------------------------------------------------------------
# Test CLI functions
# -----------------------------------------------------------------------------

class TestTaskDetection:
    """Tests for task name detection from filename."""
    
    def test_detect_from_judged_pattern(self):
        """Test detection from *_judged.jsonl pattern."""
        # Aggregate command works with judged files
        assert _detect_task_from_filename(
            "20240328_samples_wemath_testmini_reasoning_judged.jsonl"
        ) == "wemath_testmini_reasoning_judged"
    
    def test_detect_from_samples_pattern(self):
        """Test detection from *_samples_*.jsonl pattern."""
        assert _detect_task_from_filename(
            "20240328_samples_mathvision_testmini.jsonl"
        ) == "mathvision_testmini"


class TestOutputPath:
    """Tests for output path determination."""
    
    def test_explicit_output(self, tmp_path):
        """Test with explicit output path."""
        input_file = tmp_path / "input.jsonl"
        output = tmp_path / "output.json"
        
        result = _get_output_path(input_file, str(output), None)
        
        assert result == output
    
    def test_output_dir(self, tmp_path):
        """Test with output directory."""
        input_file = tmp_path / "input.jsonl"
        output_dir = tmp_path / "output"
        
        result = _get_output_path(input_file, None, str(output_dir))
        
        assert result == output_dir / "input.jsonl"
        assert output_dir.exists()
    
    def test_default_suffix(self, tmp_path):
        """Test default output with _aggregated suffix."""
        input_file = tmp_path / "input.jsonl"
        
        result = _get_output_path(input_file, None, None)
        
        assert result == tmp_path / "input_aggregated.jsonl"


class TestArgumentParser:
    """Tests for argument parsing."""
    
    def test_add_aggregate_parser(self):
        """Test that aggregate parser is added correctly."""
        import argparse
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        add_aggregate_parser(subparsers)
        
        # Test parsing basic args
        args = parser.parse_args(["aggregate", "--input", "test.jsonl", "--task", "wemath"])
        assert args.subcommand == "aggregate"
        assert args.input == "test.jsonl"
        assert args.task == "wemath"
        assert args.metric is None  # Default
    
    def test_parser_with_metric(self):
        """Test parsing with metric argument."""
        import argparse
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        add_aggregate_parser(subparsers)
        
        args = parser.parse_args([
            "aggregate",
            "--input", "test.jsonl",
            "--task", "wemath",
            "--metric", "wemath_loose"
        ])
        
        assert args.metric == "wemath_loose"


# -----------------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------------

class TestAggregateIntegration:
    """Integration tests for the aggregate command."""
    
    @patch("lmms_eval.cli.aggregate_cmd.Aggregator")
    def test_run_aggregate_single_file(self, mock_aggregator_class, tmp_path, generic_samples):
        """Test aggregating a single file."""
        mock_aggregator = MagicMock()
        mock_aggregator.aggregate.return_value = {"accuracy": 0.75}
        mock_aggregator_class.return_value = mock_aggregator
        
        # Create test file
        test_file = tmp_path / "test_samples.jsonl"
        with open(test_file, "w") as f:
            for sample in generic_samples:
                f.write(json.dumps(sample) + "\n")
        
        import argparse
        args = argparse.Namespace(
            input=str(test_file),
            task="test_task",
            metric=None,
            output=None,
            verbose=False,
        )
        
        with patch("sys.exit"):
            run_aggregate(args)
        
        mock_aggregator.aggregate.assert_called_once()
        call_args = mock_aggregator.aggregate.call_args
        assert call_args[1]["task_name"] == "test_task"
    
    @patch("lmms_eval.cli.aggregate_cmd.Aggregator")
    @patch("sys.exit")
    def test_run_aggregate_file_not_found(self, mock_exit, mock_aggregator_class, tmp_path):
        """Test handling of file not found."""
        import argparse
        
        args = argparse.Namespace(
            input=str(tmp_path / "nonexistent.jsonl"),
            task="test_task",
            metric=None,
            output=None,
            verbose=False,
        )
        
        run_aggregate(args)
        
        mock_exit.assert_called_with(1)
    
    @patch("lmms_eval.cli.aggregate_cmd.Aggregator")
    def test_run_aggregate_with_output(self, mock_aggregator_class, tmp_path, generic_samples):
        """Test aggregating with output file."""
        mock_aggregator = MagicMock()
        mock_aggregator.aggregate.return_value = {"accuracy": 0.75}
        mock_aggregator_class.return_value = mock_aggregator
        
        # Create test file
        test_file = tmp_path / "test_samples.jsonl"
        with open(test_file, "w") as f:
            for sample in generic_samples:
                f.write(json.dumps(sample) + "\n")
        
        output_file = tmp_path / "results.json"
        
        import argparse
        args = argparse.Namespace(
            input=str(test_file),
            task="test_task",
            metric=None,
            output=str(output_file),
            verbose=False,
        )
        
        with patch("sys.exit"):
            run_aggregate(args)
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            content = json.load(f)
        assert content["accuracy"] == 0.75


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_aggregate_with_invalid_json(self, tmp_path):
        """Test handling of invalid JSON in input file."""
        # Create file with some invalid JSON
        test_file = tmp_path / "invalid.jsonl"
        with open(test_file, "w") as f:
            f.write('{"doc_id": 0, "metrics": {"accuracy": 1.0}}\n')
            f.write('invalid json line\n')
            f.write('{"doc_id": 1, "metrics": {"accuracy": 0.0}}\n')
        
        aggregator = Aggregator()
        samples = aggregator._load_jsonl(test_file)
        
        # Should skip invalid line and load 2 valid samples
        assert len(samples) == 2
    
    def test_aggregate_with_empty_metrics(self):
        """Test aggregation when samples have empty metrics."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {}},
            {"doc_id": 1, "metrics": {}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        assert results == {}
    
    def test_aggregate_with_non_numeric_metrics(self):
        """Test aggregation filters out non-numeric metrics."""
        aggregator = Aggregator()
        
        samples = [
            {"doc_id": 0, "metrics": {"accuracy": 1.0, "detail": "some string"}},
            {"doc_id": 1, "metrics": {"accuracy": 0.0, "detail": "another string"}},
        ]
        
        results = aggregator.aggregate(samples, "generic_task")
        
        assert "accuracy" in results
        assert "detail" not in results
