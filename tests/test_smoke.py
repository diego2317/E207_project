"""Smoke tests for the initial project scaffold."""

import importlib


MODULES = [
    "scripts.config",
    "scripts.data_io",
    "scripts.features",
    "scripts.metrics",
    "scripts.models",
    "scripts.offline_dtw",
    "scripts.oltw",
    "scripts.online_baselines",
    "scripts.kalman_online",
    "scripts.evaluation",
    "scripts.run_benchmark",
    "scripts.aggregate_benchmark",
    "scripts.visualization",
    "scripts.utils",
]


def test_scaffold_modules_import() -> None:
    """All scaffold modules should import without syntax errors."""
    for module_name in MODULES:
        importlib.import_module(module_name)
