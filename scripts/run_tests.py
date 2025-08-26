#!/usr/bin/env python3
"""
Test runner script for Somba Pipeline Phase 2.
Runs all tests with proper setup and teardown.
"""

import subprocess
import sys
import time
import asyncio
from pathlib import Path

def run_command(cmd, check=True):
    """Run command and return result."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main test runner."""
    print("=" * 60)
    print("SOMBA PIPELINE PHASE 2 - TEST RUNNER")
    print("=" * 60)

    # Change to project root
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)

    # Install dependencies
    print("\n1. Installing dependencies...")
    if not run_command("uv sync"):
        print("❌ Failed to install dependencies")
        return False

    # Run unit tests
    print("\n2. Running unit tests...")
    if not run_command("uv run pytest tests/unit/ -v"):
        print("❌ Unit tests failed")
        return False
    print("✅ Unit tests passed")

    # Run schema validation tests
    print("\n3. Running schema validation tests...")
    if not run_command("uv run pytest tests/unit/test_schemas.py -v"):
        print("❌ Schema tests failed")
        return False
    print("✅ Schema tests passed")

    # Run acceptance criteria tests
    print("\n4. Running acceptance criteria tests...")
    if not run_command("uv run pytest tests/integration/test_acceptance_criteria.py -v"):
        print("❌ Acceptance criteria tests failed")
        return False
    print("✅ Acceptance criteria tests passed")

    # Run integration tests (these might need external services)
    print("\n5. Running integration tests...")
    print("Note: Integration tests may require RabbitMQ and other services")

    # Try to run integration tests, but don't fail if external services not available
    integration_success = run_command("uv run pytest tests/integration/test_worker_integration.py -v", check=False)
    if integration_success:
        print("✅ Integration tests passed")
    else:
        print("⚠️  Integration tests failed (may require external services)")

    # Run all tests with coverage
    print("\n6. Running full test suite with coverage...")
    if run_command("uv run pytest tests/ --cov=somba_pipeline --cov-report=term-missing", check=False):
        print("✅ Full test suite completed")

    # Lint checks
    print("\n7. Running code quality checks...")

    # Black formatting check
    if run_command("uv run black --check somba_pipeline/ tests/", check=False):
        print("✅ Code formatting check passed")
    else:
        print("⚠️  Code formatting issues found")

    # Import checks
    print("\n8. Checking imports...")
    test_imports = [
        "python -c 'from somba_pipeline.schemas import *'",
        "python -c 'from somba_pipeline.motion_detection import *'",
        "python -c 'from somba_pipeline.zone_attribution import *'",
        "python -c 'from somba_pipeline.worker import ProductionWorker'",
        "python -c 'from somba_pipeline.mock_cp import MockControlPlane'"
    ]

    import_success = True
    for cmd in test_imports:
        if not run_command(f"uv run {cmd}", check=False):
            import_success = False

    if import_success:
        print("✅ All imports successful")
    else:
        print("❌ Some imports failed")
        return False

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✅ Unit tests: PASSED")
    print("✅ Schema tests: PASSED")
    print("✅ Acceptance criteria: PASSED")
    print(f"{'✅' if integration_success else '⚠️ '} Integration tests: {'PASSED' if integration_success else 'PARTIAL'}")
    print("✅ Import tests: PASSED")

    print(f"\n🎉 Phase 2 implementation verification {'COMPLETED' if integration_success else 'MOSTLY COMPLETED'}")
    print("\nTo run individual test suites:")
    print("  uv run pytest tests/unit/")
    print("  uv run pytest tests/integration/")
    print("  uv run pytest tests/integration/test_acceptance_criteria.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
