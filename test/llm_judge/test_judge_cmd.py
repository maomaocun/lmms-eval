"""Tests for the judge CLI command.

This module tests the CLI interface for the judge subcommand.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Skip all tests if judge dependencies are not available
try:
    from lmms_eval.cli.judge_cmd import (
        _detect_task_from_filename,
        _get_output_path,
        add_judge_parser,
        run_judge,
    )
    JUDGE_AVAILABLE = True
except ImportError:
    JUDGE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not JUDGE_AVAILABLE, reason="Judge dependencies not available")


class TestTaskDetection:
    """Tests for task name detection from filename."""
    
    def test_detect_from_samples_pattern(self):
        """Test detection from *_samples_*.jsonl pattern."""
        assert _detect_task_from_filename(
            "20240328_samples_mathvision_reason_testmini.jsonl"
        ) == "mathvision_reason_testmini"
        
        assert _detect_task_from_filename(
            "model_Qwen_samples_mmmu_val.jsonl"
        ) == "mmmu_val"
    
    def test_detect_from_simple_pattern(self):
        """Test detection from samples_*.jsonl pattern."""
        assert _detect_task_from_filename(
            "samples_wemath_testmini.jsonl"
        ) == "wemath_testmini"
    
    def test_detect_failure(self):
        """Test detection failure."""
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_task_from_filename("invalid_file.jsonl")
        
        with pytest.raises(ValueError):
            _detect_task_from_filename("results.jsonl")


class TestOutputPath:
    """Tests for output path determination."""
    
    def test_explicit_output(self, tmp_path):
        """Test with explicit output path."""
        input_file = tmp_path / "input.jsonl"
        output = tmp_path / "explicit_output.jsonl"
        
        result = _get_output_path(input_file, str(output), None)
        
        assert result == output
    
    def test_output_dir(self, tmp_path):
        """Test with output directory."""
        input_file = tmp_path / "subdir" / "input.jsonl"
        output_dir = tmp_path / "output"
        
        result = _get_output_path(input_file, None, str(output_dir))
        
        assert result == output_dir / "input.jsonl"
        assert output_dir.exists()  # Directory should be created
    
    def test_default_suffix(self, tmp_path):
        """Test default output with _judged suffix."""
        input_file = tmp_path / "input.jsonl"
        
        result = _get_output_path(input_file, None, None)
        
        assert result == tmp_path / "input_judged.jsonl"


class TestArgumentParser:
    """Tests for argument parsing."""
    
    def test_add_judge_parser(self):
        """Test that judge parser is added correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        
        add_judge_parser(subparsers)
        
        # Test parsing basic args
        args = parser.parse_args(["judge", "--input_result", "test.jsonl"])
        assert args.subcommand == "judge"
        assert args.input_result == "test.jsonl"
        assert args.task == "auto-detect"  # Default
        assert args.judge_mode == "auto"  # Default
    
    def test_parser_env_defaults(self, monkeypatch):
        """Test that parser reads from env vars."""
        monkeypatch.setenv("JUDGE_MODE", "llm")
        monkeypatch.setenv("JUDGE_MODEL", "gpt-4o")
        monkeypatch.setenv("JUDGE_MAX_CONCURRENT", "8")
        
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        add_judge_parser(subparsers)
        
        args = parser.parse_args(["judge", "-i", "test.jsonl"])
        
        assert args.judge_mode == "llm"
        assert args.judge_model == "gpt-4o"
        assert args.parallel == 8


class TestRunJudge:
    """Tests for the run_judge function."""
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    def test_run_judge_single_file(self, mock_runner_class):
        """Test judging a single file."""
        mock_runner = MagicMock()
        mock_runner.judge_file.return_value = [{"id": 1}]
        mock_runner_class.return_value = mock_runner
        
        args = argparse.Namespace(
            input_result="test.jsonl",
            task="test_task",
            output="output.jsonl",
            output_dir=None,
            judge_mode="rule",
            judge_model="gpt-4o-mini",
            judge_api_key=None,
            judge_base_url=None,
            parallel=1,
            dry_run=False,
            verbose=False,
        )
        
        # Create test file
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.glob", return_value=[Path("test.jsonl")]):
                run_judge(args)
        
        mock_runner.judge_file.assert_called_once()
        mock_runner.save_results.assert_called_once()
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    def test_run_judge_auto_detect(self, mock_runner_class, tmp_path, monkeypatch):
        """Test auto-detection of task from filename."""
        mock_runner = MagicMock()
        mock_runner.judge_file.return_value = [{"id": 1}]
        mock_runner_class.return_value = mock_runner
        
        # Create test file
        test_file = tmp_path / "20240328_samples_mathvision_reason_testmini.jsonl"
        test_file.write_text('{"doc_id": 0}\n')
        
        args = argparse.Namespace(
            input_result=str(test_file),
            task="auto-detect",
            output=None,
            output_dir=str(tmp_path / "output"),
            judge_mode="rule",
            judge_model="gpt-4o-mini",
            judge_api_key=None,
            judge_base_url=None,
            parallel=1,
            dry_run=False,
            verbose=False,
        )
        
        with monkeypatch.context() as m:
            m.chdir(tmp_path)
            run_judge(args)
        
        # Verify task was auto-detected
        call_args = mock_runner.judge_file.call_args
        assert call_args[0][1] == "mathvision_reason_testmini"
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    def test_run_judge_dry_run(self, mock_runner_class):
        """Test dry run mode."""
        mock_runner = MagicMock()
        mock_runner.judge_file.return_value = [{"id": 1}]
        mock_runner_class.return_value = mock_runner
        
        args = argparse.Namespace(
            input_result="test.jsonl",
            task="test_task",
            output="output.jsonl",
            output_dir=None,
            judge_mode="rule",
            judge_model="gpt-4o-mini",
            judge_api_key=None,
            judge_base_url=None,
            parallel=1,
            dry_run=True,  # Dry run
            verbose=False,
        )
        
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.glob", return_value=[Path("test.jsonl")]):
                run_judge(args)
        
        mock_runner.judge_file.assert_called_once()
        mock_runner.save_results.assert_not_called()  # Should not save in dry run
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    @patch("sys.exit")
    def test_run_judge_file_not_found(self, mock_exit, mock_runner_class):
        """Test handling of file not found."""
        args = argparse.Namespace(
            input_result="nonexistent.jsonl",
            task="test_task",
            output=None,
            output_dir=None,
            judge_mode="rule",
            judge_model="gpt-4o-mini",
            judge_api_key=None,
            judge_base_url=None,
            parallel=1,
            dry_run=False,
            verbose=False,
        )
        
        with patch("pathlib.Path.exists", return_value=False):
            run_judge(args)
        
        mock_exit.assert_called_with(1)
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    @patch("sys.exit")
    def test_run_judge_auto_detect_failure(self, mock_exit, mock_runner_class, tmp_path):
        """Test handling of auto-detect failure."""
        mock_runner = MagicMock()
        mock_runner_class.return_value = mock_runner
        
        # Create file with invalid name pattern
        test_file = tmp_path / "invalid_name.jsonl"
        test_file.write_text('{"doc_id": 0}\n')
        
        args = argparse.Namespace(
            input_result=str(test_file),
            task="auto-detect",
            output=None,
            output_dir=str(tmp_path / "output"),
            judge_mode="rule",
            judge_model="gpt-4o-mini",
            judge_api_key=None,
            judge_base_url=None,
            parallel=1,
            dry_run=False,
            verbose=False,
        )
        
        with patch("pathlib.Path.glob", return_value=[test_file]):
            run_judge(args)
        
        # Should exit with error
        mock_exit.assert_called_with(1)
        mock_runner.judge_file.assert_not_called()
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    def test_run_judge_batch(self, mock_runner_class, tmp_path, monkeypatch):
        """Test batch processing with wildcards."""
        mock_runner = MagicMock()
        mock_runner.judge_file.return_value = [{"id": 1}]
        mock_runner_class.return_value = mock_runner
        
        # Create test files
        for i in range(3):
            test_file = tmp_path / f"samples_task{i}.jsonl"
            test_file.write_text(f'{{"doc_id": {i}}}\n')
        
        args = argparse.Namespace(
            input_result="samples_task*.jsonl",
            task="test_task",
            output=None,
            output_dir=str(tmp_path / "output"),
            judge_mode="rule",
            judge_model="gpt-4o-mini",
            judge_api_key=None,
            judge_base_url=None,
            parallel=1,
            dry_run=False,
            verbose=False,
        )
        
        with monkeypatch.context() as m:
            m.chdir(tmp_path)
            run_judge(args)
        
        # Should process all 3 files
        assert mock_runner.judge_file.call_count == 3
