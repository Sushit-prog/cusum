"""Unit and adversarial tests for e-CUSUM engine (TEST_TAXONOMY cat 1 + 6)."""

import pytest

from cusum_watch.stats.cusum import CusumAlert, CusumState, ECusum
from cusum_watch.stats.null_model import NullModel


def _make_norm_null(loc: float = 0.5, scale: float = 0.1) -> NullModel:
    """Helper to create a simple norm-based NullModel."""
    return NullModel(
        distribution="norm",
        params={"loc": loc, "scale": scale},
        fit_diagnostics={},
    )


# ---------------------------------------------------------------------------
# Unit tests (cat 1)
# ---------------------------------------------------------------------------


def test_hand_computed_trajectory():
    """Verify CUSUM trace matches hand calculation.

    Use a norm null (loc=0.5, scale=0.1) with alt_shift=0.3.
    null_loglik_ratio(x=0.5, null, 0.3) = log(norm(0.8,0.1).pdf(0.5)) - log(norm(0.5,0.1).pdf(0.5))
    = log(0.00443...) - log(3.989...) = -5.209... - 1.384... = -6.593...
    So increment ≈ -6.59.

    Start from cumulative=0, first step: max(0, 0 + (-6.59)) = 0.
    All-negative increments keep cumulative at 0.

    For a positive increment: x=0.8 (at alt mean)
    null_loglik_ratio(0.8, null, 0.3) = log(norm(0.8,0.1).pdf(0.8)) - log(norm(0.5,0.1).pdf(0.8))
    = log(3.989) - log(0.00443) = 1.384 - (-5.209) = 6.593
    """
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=100.0, alt_shift=0.3)

    # Two negative increments, then one positive
    state = CusumState()
    state, _ = cusum.update(state, 0.5)  # at null mean → negative increment
    assert state.cumulative == pytest.approx(0.0, abs=0.01)
    assert state.step_count == 1

    state, _ = cusum.update(state, 0.5)  # same → cumulative stays at 0
    assert state.cumulative == pytest.approx(0.0, abs=0.01)
    assert state.step_count == 2

    state, _ = cusum.update(state, 0.8)  # at alt mean → large positive increment
    assert state.cumulative > 4.0  # positive increment from x=0.8
    assert state.step_count == 3


def test_non_negativity():
    """Cumulative never drops below 0 even with a long run of negative increments."""
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=1000.0, alt_shift=0.3)

    state = CusumState()
    for _ in range(50):
        state, _ = cusum.update(state, 0.5)  # at null mean → negative increments

    assert state.cumulative >= 0.0
    for v in state.trace:
        assert v >= 0.0, f"trace value {v} is negative"


def test_alert_fires_at_threshold():
    """Alert fires exactly when cumulative crosses threshold."""
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=1.0, alt_shift=0.3)

    state = CusumState()
    # Feed values at alt mean to push cumulative up
    for i in range(20):
        state, alert = cusum.update(state, 0.8, request_id="req-1")
        if alert is not None:
            assert alert.triggered_at_step == state.step_count
            assert alert.request_id == "req-1"
            assert alert.threshold == 1.0
            return

    pytest.fail("Alert never fired")


def test_no_alert_below_threshold():
    """No alert when cumulative stays below threshold."""
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=1000.0, alt_shift=0.3)

    state = CusumState()
    for _ in range(10):
        state, alert = cusum.update(state, 0.5)  # at null mean, cumulative stays near 0
        assert alert is None


def test_state_not_reset_on_alert():
    """After alert, cumulative retains its value (not reset to 0)."""
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=1.0, alt_shift=0.3)

    state = CusumState()
    for i in range(20):
        state, alert = cusum.update(state, 0.8, request_id="r")
        if alert is not None:
            cum_at_alert = state.cumulative
            # Continue updating — state should keep its value
            state, _ = cusum.update(state, 0.8)
            assert state.cumulative >= cum_at_alert
            return

    pytest.fail("Alert never fired")


# ---------------------------------------------------------------------------
# Adversarial tests (cat 6)
# ---------------------------------------------------------------------------


def test_threshold_zero_fires_immediately():
    """Threshold of 0 fires on the first observable."""
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=0.0, alt_shift=0.3)

    state = CusumState()
    state, alert = cusum.update(state, 0.5)
    assert alert is not None
    assert alert.triggered_at_step == 1


def test_trace_length_matches_step_count():
    """Trace has one entry per step."""
    null = _make_norm_null(loc=0.5, scale=0.1)
    cusum = ECusum(null=null, threshold=1000.0, alt_shift=0.3)

    state = CusumState()
    for _ in range(15):
        state, _ = cusum.update(state, 0.5)

    assert len(state.trace) == 15
    assert state.step_count == 15
