"""Tests for llm_judge utility modules.

This module tests JudgePromptBuilder, ResponseParser, and other helpers.
"""

import pytest

from lmms_eval.llm_judge.utils import JudgePromptBuilder, ResponseParser


class TestResponseParser:
    """Tests for ResponseParser."""

    def test_parse_binary_response_zero_one(self):
        """Test parsing 0/1 responses."""
        assert ResponseParser.parse_binary_response("1", "0/1") == 1
        assert ResponseParser.parse_binary_response("0", "0/1") == 0
        assert ResponseParser.parse_binary_response("[1]", "0/1") == 1
        assert ResponseParser.parse_binary_response("[0]", "0/1") == 0
        assert ResponseParser.parse_binary_response("score: 1", "0/1") == 1
        assert ResponseParser.parse_binary_response("score: 0", "0/1") == 0
        assert ResponseParser.parse_binary_response("answer: 1.", "0/1") == 1
        assert ResponseParser.parse_binary_response("answer: 0.", "0/1") == 0

    def test_parse_binary_response_edge_cases(self):
        """Test edge cases that previously caused false positives."""
        # "10" should NOT be treated as 1
        assert ResponseParser.parse_binary_response("10", "0/1") == 0
        # "score: 10" should NOT be treated as 1
        assert ResponseParser.parse_binary_response("score: 10", "0/1") == 0
        # "21" should NOT be treated as 1
        assert ResponseParser.parse_binary_response("21", "0/1") == 0
        # Empty string should be 0
        assert ResponseParser.parse_binary_response("", "0/1") == 0
        # Mixed content with standalone 1 should still match
        assert ResponseParser.parse_binary_response("The answer is 1.", "0/1") == 1

    def test_parse_binary_response_yes_no(self):
        """Test parsing yes/no responses."""
        assert ResponseParser.parse_binary_response("yes", "yes/no") is True
        assert ResponseParser.parse_binary_response("Yes", "yes/no") is True
        assert ResponseParser.parse_binary_response("no", "yes/no") is False
        assert ResponseParser.parse_binary_response("No", "yes/no") is False

    def test_parse_score_response(self):
        """Test parsing score responses."""
        assert ResponseParser.parse_score_response("The score is 8.5", (1, 10)) == 8.5
        assert ResponseParser.parse_score_response("8", (1, 10)) == 8.0
        assert ResponseParser.parse_score_response("15", (1, 10)) == 10.0  # clamped
        assert ResponseParser.parse_score_response("-5", (1, 10)) == 1.0  # clamped
        assert ResponseParser.parse_score_response("abc", (1, 10)) == 1.0  # default to min

    def test_parse_comparative_response(self):
        """Test parsing comparative scores."""
        assert ResponseParser.parse_comparative_response("8 9") == (8.0, 9.0)
        assert ResponseParser.parse_comparative_response("8, 9") == (8.0, 9.0)
        assert ResponseParser.parse_comparative_response("8; 9") == (8.0, 9.0)
        assert ResponseParser.parse_comparative_response("invalid") == (-1.0, -1.0)

    def test_parse_json_response(self):
        """Test parsing JSON responses."""
        assert ResponseParser.parse_json_response('{"score": 8.5}') == {"score": 8.5}
        assert ResponseParser.parse_json_response('Some text {"score": 8.5} more text') == {"score": 8.5}
        assert ResponseParser.parse_json_response("invalid json") == {}
