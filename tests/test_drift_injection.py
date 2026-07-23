"""Unit, drift-injection, and adversarial tests for drift injection (cat 1, 3, 6)."""

import numpy as np
import pytest

from cusum_watch.calibration.generate import CalibrationSample
from cusum_watch.drift_injection.inject import (
    DriftInjectionResult,
    inject_degenerate_flattening,
    inject_entropy_spike,
    inject_repetition_collapse,
)
from cusum_watch.observable.compute import default_observable
from cusum_watch.stats.cusum import CusumState, ECusum
from cusum_watch.stats.null_model import NullModel


def _make_realistic_sample(num_steps: int = 50, k: int = 5, seed: int = 42) -> CalibrationSample:
    """Hand-constructed sample with realistic logprob patterns.

    topk_logprobs: top1 around -0.5, others spread -1 to -3, with small
    per-step noise to create realistic variance in combined values.
    No model needed — runs in CI.
    """
    rng = np.random.default_rng(seed)
    topk_logprobs = []
    for i in range(num_steps):
        base = -0.5 - 0.01 * i
        step = [base - 0.5 * j for j in range(k)]
        # Add small noise to create variance
        noise = rng.normal(0, 0.05, size=k)
        step = [v + n for v, n in zip(step, noise)]
        topk_logprobs.append(step)

    tokens = [f"tok_{i}" for i in range(num_steps)]
    logprobs = [topk[0] for topk in topk_logprobs]

    return CalibrationSample(
        prompt="test prompt",
        tokens=tokens,
        logprobs=logprobs,
        topk_logprobs=topk_logprobs,
        hidden_state_deltas=None,
    )


def _run_cusum_pipeline(
    sample: CalibrationSample,
    null_model: NullModel,
    threshold: float,
    alt_shift: float = 0.3,
) -> tuple[list, list[float]]:
    """Run observable -> CUSUM pipeline over a sample.

    Returns (list_of_alerts, cumulative_trace).
    """
    cusum = ECusum(null=null_model, threshold=threshold, alt_shift=alt_shift)
    state = CusumState()
    alerts = []
    trace = []

    for topk in sample.topk_logprobs:
        obs = default_observable(topk)
        state, alert = cusum.update(state, obs.combined)
        trace.append(state.cumulative)
        if alert is not None:
            alerts.append(alert)

    return alerts, trace


def _make_test_null_model(sample: CalibrationSample | None = None) -> NullModel:
    """Create a null model that matches the sample's combined values.

    If sample is provided, compute combined values and fit norm around them.
    Otherwise use a default norm(0.5, 0.1).
    """
    if sample is not None:
        combined = [default_observable(step).combined for step in sample.topk_logprobs]
        mean_c = float(np.mean(combined))
        std_c = float(np.std(combined))
        if std_c < 1e-6:
            std_c = 0.1  # fallback for zero-variance samples
        return NullModel(
            distribution="norm",
            params={"loc": mean_c, "scale": std_c},
            fit_diagnostics={},
        )
    return NullModel(
        distribution="norm",
        params={"loc": 0.5, "scale": 0.1},
        fit_diagnostics={},
    )


# ---------------------------------------------------------------------------
# Unit tests (cat 1)
# ---------------------------------------------------------------------------


def test_pre_injection_steps_untouched():
    """Each injection function only modifies steps >= injection_step."""
    sample = _make_realistic_sample()
    injection_step = 20

    for fn in [inject_repetition_collapse, inject_entropy_spike, inject_degenerate_flattening]:
        injected = fn(sample, injection_step)
        for i in range(injection_step):
            assert injected.topk_logprobs[i] == sample.topk_logprobs[i], (
                f"{fn.__name__} modified step {i} before injection_step {injection_step}"
            )


def test_original_unmutated():
    """Injection functions do not mutate the original sample."""
    sample = _make_realistic_sample()
    original_topk = [list(step) for step in sample.topk_logprobs]

    inject_repetition_collapse(sample, 20)
    inject_entropy_spike(sample, 20)
    inject_degenerate_flattening(sample, 20)

    assert sample.topk_logprobs == original_topk


def test_post_injection_steps_modified():
    """Injection functions modify steps >= injection_step."""
    sample = _make_realistic_sample()
    injection_step = 20

    for fn, kind in [
        (inject_repetition_collapse, "repetition_collapse"),
        (inject_entropy_spike, "entropy_spike"),
        (inject_degenerate_flattening, "degenerate_flattening"),
    ]:
        injected = fn(sample, injection_step)
        # At least one post-injection step should differ
        changed = any(
            injected.topk_logprobs[i] != sample.topk_logprobs[i]
            for i in range(injection_step, len(sample.topk_logprobs))
        )
        assert changed, f"{kind} did not modify any post-injection steps"


# ---------------------------------------------------------------------------
# Synthetic drift-injection (cat 3) — core detection tests
# ---------------------------------------------------------------------------


def test_repetition_collapse_detected():
    """Repetition collapse triggers alert within 20 tokens of injection.

    Injection at step 25. Top-1 becomes near-certain (-0.01), others -10.
    This DROPS combined (more certain = lower entropy), so we need a
    negative alt_shift CUSUM to detect it. We run both positive and
    negative CUSUM and check either fires.
    """
    sample = _make_realistic_sample(num_steps=50)
    injected = inject_repetition_collapse(sample, injection_step=25)

    null_model = _make_test_null_model(sample)
    # Two-sided: positive CUSUM for entropy increase, negative for decrease
    cusum_pos = ECusum(null=null_model, threshold=5.0, alt_shift=0.3)
    cusum_neg = ECusum(null=null_model, threshold=5.0, alt_shift=-0.3)

    state_pos = CusumState()
    state_neg = CusumState()
    alert_step = None
    for i, topk in enumerate(injected.topk_logprobs):
        obs = default_observable(topk)
        state_pos, alert_pos = cusum_pos.update(state_pos, obs.combined)
        state_neg, alert_neg = cusum_neg.update(state_neg, obs.combined)
        if alert_step is None and (alert_pos is not None or alert_neg is not None):
            alert_step = i + 1
            break

    assert alert_step is not None, "repetition_collapse was never detected"
    assert alert_step <= 25 + 20, (
        f"repetition_collapse detected at step {alert_step}, "
        f"expected within 20 tokens of injection at step 25"
    )


def test_entropy_spike_detected():
    """Entropy spike triggers alert within 20 tokens of injection.

    Injection at step 25. All topk values become equal (-1.0), giving
    maximum entropy and zero margin. The combined observable shifts
    significantly from the null's expected range.
    """
    sample = _make_realistic_sample(num_steps=50)
    injected = inject_entropy_spike(sample, injection_step=25)

    null_model = _make_test_null_model(sample)
    cusum = ECusum(null=null_model, threshold=5.0, alt_shift=0.3)

    state = CusumState()
    alert_step = None
    for i, topk in enumerate(injected.topk_logprobs):
        obs = default_observable(topk)
        state, alert = cusum.update(state, obs.combined)
        if alert is not None and alert_step is None:
            alert_step = i + 1
            break

    assert alert_step is not None, "entropy_spike was never detected"
    assert alert_step <= 25 + 20, (
        f"entropy_spike detected at step {alert_step}, "
        f"expected within 20 tokens of injection at step 25"
    )


def test_degenerate_flattening_no_alert():
    """Negative control: uniform additive shift should NOT trigger alert.

    This is the perturbation M2's observable is designed to ignore.
    If this test fails, the observable is not quantization-robust in
    practice — report as-is, do not modify to pass.
    """
    sample = _make_realistic_sample(num_steps=50)
    injected = inject_degenerate_flattening(sample, injection_step=25, magnitude=3.0)

    null_model = _make_test_null_model(sample)
    cusum = ECusum(null=null_model, threshold=5.0, alt_shift=0.3)

    state = CusumState()
    for topk in injected.topk_logprobs:
        obs = default_observable(topk)
        state, alert = cusum.update(state, obs.combined)
        assert alert is None, (
            f"degenerate_flattening triggered alert at step {state.step_count} "
            f"— observable is not quantization-robust in practice"
        )


# ---------------------------------------------------------------------------
# Adversarial tests (cat 6)
# ---------------------------------------------------------------------------


def test_injection_step_zero():
    """injection_step=0 perturbs entire sequence."""
    sample = _make_realistic_sample(num_steps=10)

    for fn in [inject_repetition_collapse, inject_entropy_spike, inject_degenerate_flattening]:
        injected = fn(sample, injection_step=0)
        # Every step should be modified
        for i in range(len(sample.topk_logprobs)):
            assert injected.topk_logprobs[i] != sample.topk_logprobs[i]


def test_injection_step_beyond_length():
    """injection_step beyond sequence length is a no-op."""
    sample = _make_realistic_sample(num_steps=10)

    for fn in [inject_repetition_collapse, inject_entropy_spike, inject_degenerate_flattening]:
        injected = fn(sample, injection_step=100)
        # All steps should be identical to original
        for i in range(len(sample.topk_logprobs)):
            assert injected.topk_logprobs[i] == sample.topk_logprobs[i]
