"""Online baseline adapters and registration helpers."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable

from scripts.models import AlignmentResult, FeatureSequence

OnlineBaselineRunner = Callable[[FeatureSequence, FeatureSequence], AlignmentResult]

DEFAULT_CANDIDATE_PACKAGES = {
    "madmom": "Audio and music-information-retrieval package with online tracking utilities.",
    "partitura": "Symbolic music package that may still be useful for alignment tooling.",
    "librosa": "Installed baseline dependency, useful if external OLTW code is wrapped locally.",
}

_REGISTERED_BASELINES: dict[str, OnlineBaselineRunner] = {}


def discover_available_packages(
    candidate_packages: dict[str, str] | None = None,
) -> dict[str, bool]:
    """Check which candidate packages are importable in the current environment."""

    packages = candidate_packages or DEFAULT_CANDIDATE_PACKAGES
    return {
        package_name: importlib.util.find_spec(package_name) is not None
        for package_name in packages
    }


def register_online_baseline(method_name: str, runner: OnlineBaselineRunner) -> None:
    """Register an online baseline adapter under a stable method name."""

    normalized_name = method_name.strip().lower()
    if not normalized_name:
        raise ValueError("method_name must not be empty.")
    _REGISTERED_BASELINES[normalized_name] = runner


def unregister_online_baseline(method_name: str) -> None:
    """Remove a previously registered online baseline adapter."""

    _REGISTERED_BASELINES.pop(method_name.strip().lower(), None)


def list_registered_online_baselines() -> tuple[str, ...]:
    """Return the registered online baseline names in sorted order."""

    return tuple(sorted(_REGISTERED_BASELINES))


def run_oltw(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    **runner_kwargs: object,
) -> AlignmentResult:
    """Run OLTW, preferring a registered override but defaulting to the in-repo runner."""

    runner = _REGISTERED_BASELINES.get("oltw")
    if runner is not None:
        return runner(reference_features, query_features, **runner_kwargs)

    from scripts import oltw

    return oltw.run_oltw(reference_features, query_features, **runner_kwargs)


def run_oltw_global(
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
    **runner_kwargs: object,
) -> AlignmentResult:
    """Run OLTW-global, preferring a registered override or falling back to the stub."""

    runner = _REGISTERED_BASELINES.get("oltw_global")
    if runner is not None:
        return runner(reference_features, query_features, **runner_kwargs)

    from scripts import oltw

    return oltw.run_oltw_global(reference_features, query_features, **runner_kwargs)


def get_adapter_notes() -> dict[str, str]:
    """Describe the common interface expected from online baseline adapters."""

    return {
        "input": "FeatureSequence for reference and query.",
        "output": "AlignmentResult with monotonic time mapping and method metadata.",
        "registration": "Call register_online_baseline('oltw', runner) or register_online_baseline('oltw_global', runner) to override the defaults.",
        "default_oltw": "The built-in `oltw` and `oltw_global` methods call PerformanceMatcher.jar by default.",
        "audio_paths": "The default Java-backed baselines consume audio paths from FeatureSequence.metadata['audio_path'] and ignore feature values.",
    }


def _run_registered_baseline(
    method_name: str,
    reference_features: FeatureSequence,
    query_features: FeatureSequence,
) -> AlignmentResult:
    runner = _REGISTERED_BASELINES.get(method_name)
    if runner is None:
        available_packages = discover_available_packages()
        package_summary = ", ".join(
            f"{package}={'yes' if is_available else 'no'}"
            for package, is_available in sorted(available_packages.items())
        )
        raise NotImplementedError(
            f"No adapter has been registered for '{method_name}'. "
            "Register one via scripts.online_baselines.register_online_baseline(...) "
            f"before benchmarking this method. Candidate package availability: {package_summary}."
        )
    return runner(reference_features, query_features)


run_oltw.uses_audio_paths = True
run_oltw_global.uses_audio_paths = True
