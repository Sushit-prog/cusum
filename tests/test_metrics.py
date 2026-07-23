"""Tests for metrics server — cat 8 (metrics/observability)."""

import numpy as np
import pytest

from cusum_watch.metrics.server import MetricsConfig, MetricsRegistry
from cusum_watch.observable.compute import default_observable
from cusum_watch.proxy.litellm_hook import CusumWatchLogger, MonitorConfig
from cusum_watch.stats.null_model import NullModel


def _make_null_model(loc=0.5, scale=0.1):
    return NullModel(distribution="norm", params={"loc": loc, "scale": scale}, fit_diagnostics={})


def _make_config(**kwargs):
    defaults = {
        "null_model_path": "dummy.json",
        "threshold_positive": 1.0,
        "threshold_negative": 1.0,
        "degrade_to_logprob_only": True,
        "alt_shift": 0.05,
    }
    defaults.update(kwargs)
    return MonitorConfig(**defaults)


def _make_high_entropy_topk(k=5, n=20):
    """Flat/uniform topk_logprobs — high entropy."""
    return [[-1.0] * k for _ in range(n)]


def _make_low_entropy_topk(k=5, n=20):
    """Peaked topk_logprobs — low entropy."""
    return [[-0.01] + [-10.0] * (k - 1) for _ in range(n)]


# ---------------------------------------------------------------------------
# Test 1: /metrics returns valid Prometheus exposition format
# ---------------------------------------------------------------------------


def test_metrics_exposition_format():
    """Metrics endpoint returns valid Prometheus exposition format."""
    reg = MetricsRegistry(MetricsConfig(model="test"))
    reg.record_alert("positive", 5)
    reg.record_alert("negative", 10)

    text = reg.generate_metrics().decode()
    families = list(text_string_to_metric_families(text))

    names = {f.name for f in families}
    assert "cusum_watch_alarms_total" in names
    assert "cusum_watch_mean_time_to_detect_tokens" in names
    assert "cusum_watch_calibration_drift" in names


# ---------------------------------------------------------------------------
# Test 2: alarms_total incremented correctly per direction
# ---------------------------------------------------------------------------


def test_alarms_total_per_direction():
    """Alarms counter increments correctly for positive and negative directions."""
    reg = MetricsRegistry(MetricsConfig(model="test-model"))

    # Directly record alerts (bypasses the full logger pipeline which needs
    # litellm response format — the metrics wiring is tested here)
    reg.record_alert("positive", 5)
    reg.record_alert("positive", 8)
    reg.record_alert("negative", 12)

    text = reg.generate_metrics().decode()
    for line in text.split("\n"):
        if "alarms_total" in line and not line.startswith("#"):
            if 'direction="positive"' in line:
                assert "2.0" in line, f"expected positive=2, got: {line}"
            elif 'direction="negative"' in line:
                assert "1.0" in line, f"expected negative=1, got: {line}"


# ---------------------------------------------------------------------------
# Test 3: histogram reflects actual triggered_at_step values
# ---------------------------------------------------------------------------


def test_histogram_reflects_step_values():
    """Histogram sum/count reflects actual triggered_at_step values."""
    reg = MetricsRegistry(MetricsConfig(model="test"))

    # Manually record alerts with known step values
    reg.record_alert("positive", 5)
    reg.record_alert("positive", 10)
    reg.record_alert("negative", 3)

    text = reg.generate_metrics().decode()

    # Check histogram count = 3
    for line in text.split("\n"):
        if "mean_time_to_detect_tokens_count" in line and not line.startswith("#"):
            assert "3.0" in line, f"expected count=3.0, got: {line}"
    # Check histogram sum = 5 + 10 + 3 = 18
    for line in text.split("\n"):
        if "mean_time_to_detect_tokens_sum" in line and not line.startswith("#"):
            assert "18.0" in line, f"expected sum=18.0, got: {line}"


# ---------------------------------------------------------------------------
# Test 4: calibration_drift distinguishes in-distribution vs shifted data
# ---------------------------------------------------------------------------


def test_calibration_drift_distinguishes_distributions():
    """Calibration drift gauge is low for in-distribution data, high for shifted."""
    null_model = _make_null_model(loc=0.5, scale=0.1)

    # In-distribution window
    reg_in = MetricsRegistry(MetricsConfig(model="in-dist"))
    rng = np.random.default_rng(42)
    for v in rng.normal(0.5, 0.1, 200):
        reg_in.record_combined(float(v))
    reg_in.update_calibration_drift("norm", {"loc": 0.5, "scale": 0.1})
    drift_in = reg_in.calibration_drift.labels(model="in-dist")._value.get()

    # Shifted window (mean=0.8, far from null mean=0.5)
    reg_shift = MetricsRegistry(MetricsConfig(model="shifted"))
    for v in rng.normal(0.8, 0.1, 200):
        reg_shift.record_combined(float(v))
    reg_shift.update_calibration_drift("norm", {"loc": 0.5, "scale": 0.1})
    drift_shift = reg_shift.calibration_drift.labels(model="shifted")._value.get()

    # In-distribution drift should be low, shifted should be high
    assert drift_in < drift_shift, (
        f"in-dist drift ({drift_in:.4f}) should be < shifted drift ({drift_shift:.4f})"
    )
    assert drift_shift > 0.3, (
        f"shifted drift ({drift_shift:.4f}) should be clearly > 0"
    )


# ---------------------------------------------------------------------------
# Test 5: calibration_drift with insufficient data
# ---------------------------------------------------------------------------


def test_calibration_drift_insufficient_data():
    """Calibration drift is 0 when window has fewer than 30 values."""
    reg = MetricsRegistry(MetricsConfig(model="test"))
    for v in [0.5, 0.6, 0.7]:
        reg.record_combined(v)
    reg.update_calibration_drift("norm", {"loc": 0.5, "scale": 0.1})
    drift = reg.calibration_drift.labels(model="test")._value.get()
    assert drift == 0.0


# Need this import for the exposition format test
from prometheus_client.parser import text_string_to_metric_families
