"""
Pytest configuration and fixtures.

Registers custom command-line options for controlling test execution.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests that require GPU (marked with @pytest.mark.gpu)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests unless explicitly requested."""
    run_gpu = config.getoption("--run-gpu")

    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run")

    for item in items:
        # Skip GPU tests unless --run-gpu is provided
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)
