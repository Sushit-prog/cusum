"""Tests for check_api_compat.py (M13)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_api_compat.py"
BASELINE = Path(__file__).resolve().parent.parent / "api_baseline.json"


def test_compat_check_passes():
    """check_api_compat.py exits 0 against the current baseline."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True,
        cwd=str(SCRIPT.parent.parent),
    )
    assert result.returncode == 0, f"Expected exit 0, got {result.returncode}:\n{result.stdout}\n{result.stderr}"
    assert "No changes" in result.stdout or "NEW APIs" in result.stdout


def test_compat_check_detects_signature_change():
    """Deliberately rename a parameter and verify exit 1 with clear diff."""
    # Monkeypatch: temporarily modify the baseline to simulate a signature change
    original = json.loads(BASELINE.read_text())
    modified = dict(original)

    # Simulate ECusum.__init__ losing the 'alt_shift' parameter
    key = "cusum_watch.stats.cusum.ECusum"
    old_sig = modified[key]["signature"]
    modified[key]["signature"] = old_sig.replace("alt_shift", "REMOVED_param")

    temp_baseline = BASELINE.parent / "_test_baseline.json"
    temp_baseline.write_text(json.dumps(modified, indent=2))

    try:
        # Patch BASELINE_PATH in the script
        result = subprocess.run(
            [sys.executable, "-c", f"""
import sys
from pathlib import Path
sys.path.insert(0, '{SCRIPT.parent}')
import check_api_compat
check_api_compat.BASELINE_PATH = Path('{temp_baseline}')
compatible, diff = check_api_compat.check_compat()
print(diff)
sys.exit(0 if compatible else 1)
"""],
            capture_output=True, text=True,
            cwd=str(SCRIPT.parent.parent),
        )
        assert result.returncode == 1, "Should fail when signature changed"
        assert "SIGNATURE CHANGED" in result.stdout or "ECusum" in result.stdout
    finally:
        temp_baseline.unlink(missing_ok=True)


def test_update_baseline():
    """--update-baseline regenerates the file and a subsequent check passes."""
    # Save original
    original = BASELINE.read_text()

    try:
        # Update baseline
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--update-baseline"],
            capture_output=True, text=True,
            cwd=str(SCRIPT.parent.parent),
        )
        assert result.returncode == 0
        assert BASELINE.exists()

        # Check should pass against new baseline
        result2 = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True, text=True,
            cwd=str(SCRIPT.parent.parent),
        )
        assert result2.returncode == 0, f"Check failed after update: {result2.stdout}"
    finally:
        # Restore original
        BASELINE.write_text(original)
