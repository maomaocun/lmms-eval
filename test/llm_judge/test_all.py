"""Master test suite for llm_judge module.

This module imports and runs all tests for the llm_judge functionality.
"""

# Import all test modules to ensure they're discovered
from test.llm_judge.test_integration import *
from test.llm_judge.test_judge_cmd import *
from test.llm_judge.test_standalone import *
