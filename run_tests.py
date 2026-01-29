#!/usr/bin/env python3
"""
README für Tests nach Projektstruktur-Migration.

Die Tests wurden von verschiedenen Orten in das zentrale tests/ Verzeichnis migriert.
"""

# Test-Verzeichnis nutzen
# =====================

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

def run_tests():
    """Führe alle Tests aus."""
    print("Running all tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=PROJECT_ROOT
    )
    return result.returncode

def run_single_test(test_file):
    """Führe einzelnen Test aus."""
    print(f"Running {test_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/{test_file}", "-v"],
        cwd=PROJECT_ROOT
    )
    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        exit_code = run_single_test(test_name)
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)
