#!/usr/bin/env python
"""Test runner for llm_judge module.

This script runs all tests for the llm_judge module.

Usage:
    python test_llm_judge.py
    python test_llm_judge.py -v  # verbose
    python test_llm_judge.py -k test_name  # specific test
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run the llm_judge tests."""
    # Get the project root
    project_root = Path(__file__).parent
    test_dir = project_root / "test" / "llm_judge"
    
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("Running llm_judge tests")
    print("=" * 60)
    print(f"Test directory: {test_dir}")
    print(f"Python: {sys.executable}")
    print()
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", str(test_dir)]
    
    # Pass through additional arguments
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    print()
    
    # Run tests
    result = subprocess.run(cmd)
    
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
