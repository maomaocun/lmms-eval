"""Master test suite for llm_judge module.

This module imports and runs all tests for the llm_judge functionality.
"""

import pytest

ALL_AVAILABLE = True

# Aggregate tests require torch and may be unavailable in some CI envs.
# Only re-export them when their dependencies are loadable.
try:
    import test.llm_judge.test_aggregate as _ta
    if getattr(_ta, "AGGREGATE_AVAILABLE", True):
        from test.llm_judge.test_aggregate import *
except ImportError:
    pass

try:
    import test.llm_judge.test_standalone as _ts
    from test.llm_judge.test_standalone import *
    ALL_AVAILABLE = ALL_AVAILABLE and getattr(_ts, "JUDGE_AVAILABLE", True)
except ImportError:
    ALL_AVAILABLE = False

try:
    from test.llm_judge.test_utils import *
except ImportError:
    ALL_AVAILABLE = False

pytestmark = pytest.mark.skipif(not ALL_AVAILABLE, reason="Judge dependencies not available")
