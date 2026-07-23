"""Public API compatibility checker for cusum-watch.

Introspects the installed cusum_watch package and compares against a saved
baseline. Exits 1 if any public signature changed, 0 if unchanged.

Usage:
    python scripts/check_api_compat.py              # check against baseline
    python scripts/check_api_compat.py --update     # regenerate baseline
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


BASELINE_PATH = Path(__file__).resolve().parent.parent / "api_baseline.json"

# Public API surface — every class/function listed in INTERFACES.md §1-8
PUBLIC_API = [
    # §1 — calibration/generate.py
    ("cusum_watch.calibration.generate", "CalibrationSample"),
    ("cusum_watch.calibration.generate", "generate_calibration_set"),
    ("cusum_watch.calibration.generate", "save_calibration_set"),
    ("cusum_watch.calibration.generate", "load_calibration_set"),
    ("cusum_watch.calibration.generate", "combined_values_from_calibration_set"),
    # §2 — observable/compute.py
    ("cusum_watch.observable.compute", "StepObservable"),
    ("cusum_watch.observable.compute", "ObservableFn"),
    ("cusum_watch.observable.compute", "default_observable"),
    # §3 — stats/null_model.py
    ("cusum_watch.stats.null_model", "NullModel"),
    ("cusum_watch.stats.null_model", "fit_null"),
    ("cusum_watch.stats.null_model", "null_loglik_ratio"),
    # §4 — stats/cusum.py
    ("cusum_watch.stats.cusum", "CusumState"),
    ("cusum_watch.stats.cusum", "CusumAlert"),
    ("cusum_watch.stats.cusum", "ECusum"),
    # §5 — calibration/threshold.py
    ("cusum_watch.calibration.threshold", "calibrate_threshold"),
    # §6 — proxy/litellm_hook.py
    ("cusum_watch.proxy.litellm_hook", "TwoSidedCusumState"),
    ("cusum_watch.proxy.litellm_hook", "MonitorConfig"),
    ("cusum_watch.proxy.litellm_hook", "CusumWatchAlert"),
    ("cusum_watch.proxy.litellm_hook", "CusumWatchLogger"),
    # §7 — metrics/server.py
    ("cusum_watch.metrics.server", "MetricsConfig"),
    ("cusum_watch.metrics.server", "MetricsRegistry"),
    ("cusum_watch.metrics.server", "create_app"),
]


def _get_signature_str(obj) -> str:
    """Get a string representation of an object's signature."""
    try:
        if isinstance(obj, type):
            # Class — get __init__ signature
            sig = inspect.signature(obj.__init__)
            params = [
                p.name for p in sig.parameters.values()
                if p.name != "self"
            ]
            return f"class {obj.__name__}({', '.join(params)})"
        elif callable(obj):
            sig = inspect.signature(obj)
            return f"def {obj.__name__}{sig}"
        else:
            return f"<{type(obj).__name__}>"
    except (ValueError, TypeError):
        return f"<no signature: {type(obj).__name__}>"


def _get_fields(obj) -> list[str]:
    """Get dataclass fields or class attributes."""
    if hasattr(obj, "__dataclass_fields__"):
        return sorted(obj.__dataclass_fields__.keys())
    return []


def _scan_api() -> dict:
    """Scan the public API surface and return a baseline dict."""
    import importlib

    baseline = {}
    for module_path, name in PUBLIC_API:
        try:
            module = importlib.import_module(module_path)
            obj = getattr(module, name)
        except (ImportError, AttributeError) as e:
            baseline[f"{module_path}.{name}"] = {
                "error": str(e),
                "signature": None,
                "fields": [],
            }
            continue

        baseline[f"{module_path}.{name}"] = {
            "signature": _get_signature_str(obj),
            "fields": _get_fields(obj),
        }

    return baseline


def check_compat() -> tuple[bool, str]:
    """Check current API against baseline. Returns (is_compatible, diff_text)."""
    if not BASELINE_PATH.exists():
        return False, "No baseline found. Run with --update-baseline first."

    baseline = json.loads(BASELINE_PATH.read_text())
    current = _scan_api()

    diffs = []

    # Check for removed/renamed APIs
    for key in baseline:
        if key not in current:
            diffs.append(f"REMOVED: {key}")

    # Check for signature/field changes
    for key in baseline:
        if key not in current:
            continue
        old = baseline[key]
        new = current[key]

        if old.get("error") or new.get("error"):
            if old.get("error") != new.get("error"):
                diffs.append(f"ERROR CHANGE: {key}: {old.get('error')} -> {new.get('error')}")
            continue

        if old["signature"] != new["signature"]:
            diffs.append(f"SIGNATURE CHANGED: {key}")
            diffs.append(f"  was: {old['signature']}")
            diffs.append(f"  now: {new['signature']}")

        if old["fields"] != new["fields"]:
            diffs.append(f"FIELDS CHANGED: {key}")
            diffs.append(f"  was: {old['fields']}")
            diffs.append(f"  now: {new['fields']}")

    # Check for new APIs (informational, not a failure)
    new_apis = [k for k in current if k not in baseline]
    if new_apis:
        diffs.append("NEW APIs (not a breakage, but noted):")
        for k in new_apis:
            diffs.append(f"  + {k}: {current[k]['signature']}")

    is_compatible = len(diffs) == 0 or all("NEW APIs" in d for d in diffs)
    return is_compatible, "\n".join(diffs) if diffs else "No changes."


def main():
    parser = argparse.ArgumentParser(description="Check public API compatibility")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Regenerate api_baseline.json from current code")
    args = parser.parse_args()

    if args.update_baseline:
        baseline = _scan_api()
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Baseline updated: {BASELINE_PATH}")
        return

    compatible, diff_text = check_compat()
    print(diff_text)
    sys.exit(0 if compatible else 1)


if __name__ == "__main__":
    main()
