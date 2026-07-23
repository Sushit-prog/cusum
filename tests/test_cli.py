"""Tests for cusum-watch CLI (cat 1, 6)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from cusum_watch.cli.main import cli
from cusum_watch.calibration.generate import CalibrationSample
from cusum_watch.calibration.threshold import calibrate_threshold
from cusum_watch.stats.null_model import (
    NullModel,
    combined_values_from_calibration_set,
    fit_null,
)


def _make_fake_samples(count=50, k=5):
    """Hand-constructed calibration samples for testing without a model."""
    import numpy as np
    rng = np.random.default_rng(42)
    samples = []
    for i in range(count):
        base = -0.5 + rng.normal(0, 0.05)
        topk = sorted([base - 0.5 * j + rng.normal(0, 0.02) for j in range(k)], reverse=True)
        # Repeat topk for multiple steps per sample
        topk_logprobs = [topk for _ in range(20)]
        tokens = [f"tok_{j}" for j in range(20)]
        logprobs = [t[0] for t in topk_logprobs]
        samples.append(CalibrationSample(
            prompt=f"prompt_{i}",
            tokens=tokens,
            logprobs=logprobs,
            topk_logprobs=topk_logprobs,
            hidden_state_deltas=None,
        ))
    return samples


def test_calibrate_end_to_end_no_model(tmp_path):
    """Test calibrate pipeline with fake samples (no GGUF model needed).

    We can't call generate_calibration_set without a model, so we test the
    pipeline logic by directly creating samples and running the downstream steps.
    """
    samples = _make_fake_samples(count=50)
    combined = combined_values_from_calibration_set(samples)
    assert len(combined) > 0

    # Fit null
    null_model = fit_null(combined)
    assert null_model.distribution in ("norm", "beta")

    # Calibrate both directions
    thresh_pos, report_pos = calibrate_threshold(
        combined, 0.05, null_model, alt_shift=0.002,
    )
    thresh_neg, report_neg = calibrate_threshold(
        combined, 0.05, null_model, alt_shift=-0.002,
    )
    assert thresh_pos >= 0
    assert thresh_neg >= 0
    assert "empirical_false_alarm_rate" in report_pos
    assert "empirical_false_alarm_rate" in report_neg


def test_calibrate_missing_model_path():
    """calibrate with missing model path produces clear error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["calibrate"])
    assert result.exit_code != 0
    assert "Missing required option" in result.output or "model-path" in result.output or "required" in result.output.lower()


def test_inspect_known_calibration_file(tmp_path):
    """inspect on a known calibration file produces expected output."""
    # Create a calibration file
    cal_data = {
        "null_model": {
            "distribution": "norm",
            "params": {"loc": 0.5, "scale": 0.1},
            "fit_diagnostics": {"ks_statistic": 0.05, "ks_pvalue": 0.8, "sample_size": 200},
        },
        "threshold_positive": 1.5,
        "threshold_negative": 1.8,
        "calibration_report_positive": {"empirical_false_alarm_rate": 0.04, "num_simulated_sequences": 100},
        "calibration_report_negative": {"empirical_false_alarm_rate": 0.06, "num_simulated_sequences": 100},
        "alt_shift_positive": 0.002,
        "alt_shift_negative": 0.002,
        "target_far": 0.05,
        "calibration_set_size": 200,
    }
    cal_file = tmp_path / "cal.json"
    cal_file.write_text(json.dumps(cal_data))

    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", str(cal_file)])

    assert result.exit_code == 0
    assert "norm" in result.output
    assert "1.5" in result.output or "1.5000" in result.output
    assert "1.8" in result.output or "1.8000" in result.output
    assert "0.04" in result.output or "0.040" in result.output
    assert "200" in result.output


def test_inspect_missing_file():
    """inspect on nonexistent file produces error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["inspect", "/nonexistent/cal.json"])
    assert result.exit_code != 0


def test_serve_metrics_starts_and_responds():
    """serve-metrics starts and /metrics responds."""
    import threading
    import time
    import urllib.request

    from cusum_watch.metrics.server import create_app

    app, _ = create_app()

    # Start server in a thread
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=18765, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(1.0)  # Wait for server to start

    try:
        resp = urllib.request.urlopen("http://127.0.0.1:18765/metrics")
        assert resp.status == 200
        body = resp.read().decode()
        assert "cusum_watch_alarms_total" in body
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_cli_help():
    """CLI --help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "cusum-watch" in result.output


def test_calibrate_help():
    """calibrate --help works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["calibrate", "--help"])
    assert result.exit_code == 0
    assert "model-path" in result.output
