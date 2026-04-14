"""Online baseline discovery helpers and adapter placeholders."""

from __future__ import annotations

import importlib.util


DEFAULT_CANDIDATE_PACKAGES = {
    "madmom": "Audio and music-information-retrieval package with online tracking utilities.",
    "partitura": "Symbolic music package that may still be useful for alignment tooling.",
    "librosa": "Installed baseline dependency, useful if external OLTW code is wrapped locally.",
}


def discover_available_packages(
    candidate_packages: dict[str, str] | None = None,
) -> dict[str, bool]:
    """Check which candidate packages are importable in the current environment."""

    packages = candidate_packages or DEFAULT_CANDIDATE_PACKAGES
    return {
        package_name: importlib.util.find_spec(package_name) is not None
        for package_name in packages
    }


def get_adapter_notes() -> dict[str, str]:
    """Describe the common interface expected from future online baselines."""

    return {
        "input": "FeatureSequence for reference and query or a streaming query wrapper.",
        "output": "AlignmentResult with monotonic time mapping and method metadata.",
        "next_step": "Wrap an existing OLTW or OLTW-global implementation behind this contract.",
    }
