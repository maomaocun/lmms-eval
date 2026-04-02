"""Integration tests for the judge module.

These tests verify end-to-end functionality of the judge system
using actual task configurations and sample data.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Skip all tests if judge dependencies are not available
try:
    from lmms_eval.llm_judge.standalone import JudgeRunner
    from lmms_eval.cli.judge_cmd import run_judge
    JUDGE_AVAILABLE = True
except ImportError:
    JUDGE_AVAILABLE = False


pytestmark = pytest.mark.skipif(not JUDGE_AVAILABLE, reason="Judge dependencies not available")


@pytest.fixture
def math_sample_jsonl(tmp_path):
    """Create a sample JSONL file with math problems."""
    samples = [
        {
            "doc_id": 0,
            "doc": {
                "question": "What is 2+2?",
                "answer": "4",
                "options": ["2", "3", "4", "5"],
                "question_type": "multi_choice",
            },
            "filtered_resps": "<think>The sum of 2 and 2 is 4.</think><answer>4</answer>",
            "target": "4",
            "input": "What is 2+2?\nA. 2\nB. 3\nC. 4\nD. 5",
        },
        {
            "doc_id": 1,
            "doc": {
                "question": "Solve for x: 2x = 10",
                "answer": "5",
                "question_type": "open",
            },
            "filtered_resps": "<think>Dividing both sides by 2 gives x = 5.</think><answer>5</answer>",
            "target": "5",
            "input": "Solve for x: 2x = 10",
        },
        {
            "doc_id": 2,
            "doc": {
                "question": "What is 3*3?",
                "answer": "9",
            },
            "filtered_resps": "<think>3 times 3 equals 9.</think><answer>9</answer>",
            "target": "9",
            "input": "What is 3*3?",
        },
    ]
    
    file_path = tmp_path / "samples_math_test.jsonl"
    with open(file_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    return file_path


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_judge_file_with_mock_task(self, math_sample_jsonl):
        """Test judging a file with a mock task that mimics real behavior."""
        
        def mock_process_results(doc, results):
            """Mock process_results that checks answer correctness."""
            prediction = results[0] if results else ""
            answer = doc.get("answer", "")
            
            # Simple extraction from <answer> tags
            if "<answer>" in prediction:
                import re
                match = re.search(r"<answer>(.*?)</answer>", prediction)
                if match:
                    prediction = match.group(1)
            
            # Check correctness
            correct = str(answer).strip() == str(prediction).strip()
            
            return {
                "acc_score": 1.0 if correct else 0.0,
                "format_score": 1.0 if "<answer>" in results[0] else 0.0,
            }
        
        # Create mock task
        mock_task = MagicMock()
        mock_task.config.process_results = mock_process_results
        
        runner = JudgeRunner(judge_mode="rule")
        
        # Mock task loading
        with patch.object(runner, "_load_task", return_value=mock_task):
            results = runner.judge_file(math_sample_jsonl, "mock_task")
        
        assert len(results) == 3
        
        # All should have correct format
        for r in results:
            assert r["metrics"]["format_score"] == 1.0
            assert r["judge_mode"] == "rule"
    
    def test_judge_file_with_format_checking(self, tmp_path):
        """Test that format checking works correctly."""
        
        # Create samples with different formats
        samples = [
            {
                "doc_id": 0,
                "doc": {"answer": "4"},
                "filtered_resps": "<think>...</think><answer>4</answer>",
            },
            {
                "doc_id": 1,
                "doc": {"answer": "5"},
                "filtered_resps": "5",  # No format
            },
        ]
        
        file_path = tmp_path / "format_test.jsonl"
        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        
        def mock_process_results(doc, results):
            pred = results[0]
            has_format = "<answer>" in pred and "</answer>" in pred
            return {
                "acc_score": 1.0,
                "format_score": 1.0 if has_format else 0.0,
            }
        
        mock_task = MagicMock()
        mock_task.config.process_results = mock_process_results
        
        runner = JudgeRunner(judge_mode="rule")
        
        with patch.object(runner, "_load_task", return_value=mock_task):
            results = runner.judge_file(file_path, "mock_task")
        
        assert results[0]["metrics"]["format_score"] == 1.0
        assert results[1]["metrics"]["format_score"] == 0.0


class TestJSONLIO:
    """Tests for JSONL input/output handling."""
    
    def test_roundtrip(self, tmp_path):
        """Test that data survives a roundtrip through save/load."""
        
        runner = JudgeRunner()
        
        original_data = [
            {
                "doc_id": 0,
                "doc": {"question": "Q1", "answer": "A1"},
                "filtered_resps": "Response 1",
                "metrics": {"acc_score": 1.0},
            },
            {
                "doc_id": 1,
                "doc": {"question": "Q2", "answer": "A2"},
                "filtered_resps": "Response 2",
                "metrics": {"acc_score": 0.0},
            },
        ]
        
        output_path = tmp_path / "roundtrip.jsonl"
        
        # Save
        runner.save_results(original_data, output_path)
        
        # Load
        loaded = runner._load_jsonl(output_path)
        
        # Verify
        assert len(loaded) == 2
        assert loaded[0]["doc_id"] == 0
        assert loaded[1]["metrics"]["acc_score"] == 0.0
    
    def test_handles_special_characters(self, tmp_path):
        """Test handling of special characters in JSON."""
        
        runner = JudgeRunner()
        
        data = [
            {
                "doc_id": 0,
                "doc": {
                    "question": "What is $\LaTeX$?",
                    "answer": "$\frac{1}{2}$",
                },
                "filtered_resps": "Answer with <special> chars & \"quotes\"",
            },
        ]
        
        output_path = tmp_path / "special.jsonl"
        runner.save_results(data, output_path)
        
        loaded = runner._load_jsonl(output_path)
        
        assert len(loaded) == 1
        assert "$\\LaTeX$" in loaded[0]["doc"]["question"]
    
    def test_handles_unicode(self, tmp_path):
        """Test handling of unicode characters."""
        
        runner = JudgeRunner()
        
        data = [
            {
                "doc_id": 0,
                "doc": {
                    "question": "你好，世界",
                    "answer": "答案",
                },
                "filtered_resps": "这是回答 🎉",
            },
        ]
        
        output_path = tmp_path / "unicode.jsonl"
        runner.save_results(data, output_path)
        
        loaded = runner._load_jsonl(output_path)
        
        assert loaded[0]["doc"]["question"] == "你好，世界"
        assert "🎉" in loaded[0]["filtered_resps"]


class TestCLIIntegration:
    """Integration tests for the CLI interface."""
    
    @patch("lmms_eval.cli.judge_cmd.JudgeRunner")
    def test_cli_with_mock_files(self, mock_runner_class, tmp_path, monkeypatch):
        """Test CLI execution with mock files."""
        import argparse
        
        mock_runner = MagicMock()
        mock_runner.judge_file.return_value = [
            {"doc_id": 0, "metrics": {"acc_score": 1.0}}
        ]
        mock_runner_class.return_value = mock_runner
        
        # Create test file
        test_file = tmp_path / "samples_test_task.jsonl"
        test_file.write_text('{"doc_id": 0}\n')
        
        args = argparse.Namespace(
            input=str(test_file),
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
        
        run_judge(args)
        
        # Verify runner was created with correct args
        mock_runner_class.assert_called_once()
        call_kwargs = mock_runner_class.call_args.kwargs
        assert call_kwargs["judge_mode"] == "rule"
        assert call_kwargs["judge_model"] == "gpt-4o-mini"
        
        # Verify judging was performed
        mock_runner.judge_file.assert_called_once()
        mock_runner.save_results.assert_called_once()


class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_handles_missing_doc_field(self, tmp_path):
        """Test handling of samples missing doc field."""
        
        # Create file with missing doc
        file_path = tmp_path / "bad.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"doc_id": 0, "filtered_resps": "resp"}) + "\n")
        
        def mock_process_results(doc, results):
            # Should still work with empty doc
            return {"acc_score": 0.0}
        
        mock_task = MagicMock()
        mock_task.config.process_results = mock_process_results
        
        runner = JudgeRunner(judge_mode="rule")
        
        with patch.object(runner, "_load_task", return_value=mock_task):
            results = runner.judge_file(file_path, "mock_task")
        
        assert len(results) == 1
        assert "metrics" in results[0]
    
    def test_handles_missing_filtered_resps(self, tmp_path):
        """Test handling of samples missing filtered_resps."""
        
        file_path = tmp_path / "bad.jsonl"
        with open(file_path, "w") as f:
            f.write(json.dumps({"doc_id": 0, "doc": {"answer": "A"}}) + "\n")
        
        def mock_process_results(doc, results):
            return {"acc_score": 0.0 if not results else 1.0}
        
        mock_task = MagicMock()
        mock_task.config.process_results = mock_process_results
        
        runner = JudgeRunner(judge_mode="rule")
        
        with patch.object(runner, "_load_task", return_value=mock_task):
            results = runner.judge_file(file_path, "mock_task")
        
        assert results[0]["metrics"]["acc_score"] == 0.0  # Empty results list


@pytest.mark.skipif(
    not os.getenv("JUDGE_API_KEY"),
    reason="JUDGE_API_KEY not set, skipping live API tests"
)
class TestLiveAPI:
    """Tests that require live API access.
    
    These are skipped unless JUDGE_API_KEY is set.
    """
    
    def test_live_llm_judge(self):
        """Test with actual LLM API (requires API key)."""
        runner = JudgeRunner(
            judge_mode="llm",
            judge_model=os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
            judge_api_key=os.getenv("JUDGE_API_KEY"),
            judge_base_url=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )
        
        doc = {
            "question": "What is the capital of France?",
            "answer": "Paris",
        }
        results = ["The capital of France is Paris."]
        
        metrics = runner._apply_llm_judge(doc, results, {})
        
        assert metrics["llm_judge_success"] is True
        assert metrics["llm_judge_score"] == 1
        assert "llm_judge_raw" in metrics
