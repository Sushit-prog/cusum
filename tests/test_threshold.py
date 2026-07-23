"""Calibration validity and adversarial tests for threshold calibration (cat 2 + 6)."""

import numpy as np
import pytest

from cusum_watch.calibration.threshold import calibrate_threshold
from cusum_watch.stats.cusum import CusumState, ECusum
from cusum_watch.stats.null_model import NullModel


def _make_norm_null(loc: float = 0.5, scale: float = 0.1) -> NullModel:
    """Helper to create a simple norm-based NullModel."""
    return NullModel(
        distribution="norm",
        params={"loc": loc, "scale": scale},
        fit_diagnostics={},
    )


# ---------------------------------------------------------------------------
# Calibration validity (cat 2) — fast (default CI)
# ---------------------------------------------------------------------------


def test_end_to_end_calibration_validity():
    """After calibrate_threshold, run CUSUM over a third independent sample
    and confirm empirical alarm rate is in a reasonable range.

    Uses reduced simulation counts (100/50) for fast CI. See
    test_calibration_validity_full for the thorough version.
    """
    rng = np.random.default_rng(99)
    data = rng.normal(loc=0.5, scale=0.1, size=300).tolist()
    data = [max(1e-6, min(1.0 - 1e-6, v)) for v in data]

    null_model = _make_norm_null(loc=0.5, scale=0.1)
    target_rate = 0.10

    threshold, report = calibrate_threshold(
        null_observables=data,
        target_false_alarm_rate=target_rate,
        null_model=null_model,
        alt_shift=0.05,
        num_simulations=100,
        sequence_length=50,
        rng_seed=77,
    )

    assert "empirical_false_alarm_rate" in report
    assert "threshold" in report
    assert report["threshold"] >= 0

    cusum = ECusum(null=null_model, threshold=threshold, alt_shift=0.05)
    third_data = rng.normal(loc=0.5, scale=0.1, size=300).tolist()
    third_data = [max(1e-6, min(1.0 - 1e-6, v)) for v in third_data]

    num_alarms = 0
    num_seqs = 50
    third_arr = np.array(third_data)
    for _ in range(num_seqs):
        seq = rng.choice(third_arr, size=50, replace=True)
        state = CusumState()
        for obs in seq:
            state, alert = cusum.update(state, float(obs))
            if alert is not None:
                num_alarms += 1
                break

    empirical_rate = num_alarms / num_seqs
    assert 0.0 < empirical_rate < 1.0, (
        f"Empirical alarm rate {empirical_rate:.3f} is degenerate (0 or 1)"
    )


@pytest.mark.slow
def test_calibration_validity_full():
    """Full calibration validity check with higher simulation counts.

    Marked slow — run separately from CI via: pytest -m slow
    """
    rng = np.random.default_rng(99)
    data = rng.normal(loc=0.5, scale=0.1, size=500).tolist()
    data = [max(1e-6, min(1.0 - 1e-6, v)) for v in data]

    null_model = _make_norm_null(loc=0.5, scale=0.1)
    target_rate = 0.10

    threshold, report = calibrate_threshold(
        null_observables=data,
        target_false_alarm_rate=target_rate,
        null_model=null_model,
        alt_shift=0.05,
        num_simulations=500,
        sequence_length=100,
        rng_seed=77,
    )

    cusum = ECusum(null=null_model, threshold=threshold, alt_shift=0.05)
    third_data = rng.normal(loc=0.5, scale=0.1, size=500).tolist()
    third_data = [max(1e-6, min(1.0 - 1e-6, v)) for v in third_data]

    num_alarms = 0
    num_seqs = 200
    third_arr = np.array(third_data)
    for _ in range(num_seqs):
        seq = rng.choice(third_arr, size=100, replace=True)
        state = CusumState()
        for obs in seq:
            state, alert = cusum.update(state, float(obs))
            if alert is not None:
                num_alarms += 1
                break

    empirical_rate = num_alarms / num_seqs
    assert 0.0 < empirical_rate < 1.0, (
        f"Empirical alarm rate {empirical_rate:.3f} is degenerate (0 or 1)"
    )


# ---------------------------------------------------------------------------
# Adversarial tests (cat 6)
# ---------------------------------------------------------------------------


def test_target_outside_01_raises():
    """target_false_alarm_rate outside (0, 1) raises ValueError."""
    null_model = _make_norm_null()
    data = [0.5] * 200

    with pytest.raises(ValueError, match="must be in"):
        calibrate_threshold(data, 0.0, null_model)

    with pytest.raises(ValueError, match="must be in"):
        calibrate_threshold(data, 1.0, null_model)

    with pytest.raises(ValueError, match="must be in"):
        calibrate_threshold(data, -0.1, null_model)


def test_too_few_observables_raises():
    """Too few null observables raises ValueError."""
    null_model = _make_norm_null()
    data = [0.5] * 50

    with pytest.raises(ValueError, match="at least 100"):
        calibrate_threshold(data, 0.05, null_model)
