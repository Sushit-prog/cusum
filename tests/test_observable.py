"""Unit and adversarial tests for observable computation (TEST_TAXONOMY cat 1 + 6)."""

import math

import pytest

from cusum_watch.observable.compute import StepObservable, default_observable


# ---------------------------------------------------------------------------
# Unit tests (cat 1)
# ---------------------------------------------------------------------------


def test_uniform_distribution_entropy_ratio():
    """Uniform top-k distribution should have entropy_ratio ≈ 1.0.

    For k=4 uniform: all probs = 0.25, entropy = -4*(0.25*log(0.25)) = log(4),
    so entropy_ratio = log(4)/log(4) = 1.0.
    """
    topk = [0.0, 0.0, 0.0, 0.0]
    obs = default_observable(topk)
    assert obs.entropy_ratio == pytest.approx(1.0)


def test_peaked_distribution_entropy_ratio():
    """Maximally peaked distribution should have entropy_ratio ≈ 0.0.

    With top1=0.0 and rest at -100/-200, softmax puts ~1.0 probability on
    the first element, so entropy ≈ 0.
    """
    topk = [0.0, -100.0, -200.0]
    obs = default_observable(topk)
    assert obs.entropy_ratio == pytest.approx(0.0, abs=1e-6)


def test_known_margin_ratio():
    """Hand-computed margin_ratio for known input.

    topk_logprobs = [-0.5, -1.5, -2.5]
    spread = -0.5 - (-2.5) = 2.0
    margin  = (-0.5 - (-1.5)) / 2.0 = 1.0 / 2.0 = 0.5
    """
    topk = [-0.5, -1.5, -2.5]
    obs = default_observable(topk)
    assert obs.margin_ratio == pytest.approx(0.5)


def test_combined_matches_formula():
    """combined = 0.5 * entropy_ratio + 0.5 * (1 - margin_ratio)."""
    topk = [-0.5, -1.5, -2.5]
    obs = default_observable(topk)
    expected = 0.5 * obs.entropy_ratio + 0.5 * (1.0 - obs.margin_ratio)
    assert obs.combined == pytest.approx(expected)


def test_scale_invariance():
    """entropy_ratio is invariant to adding a constant to all logprobs.

    This is the core quantization-robustness property: quantization noise
    tends to introduce a uniform additive shift, which this observable
    should ignore.
    """
    inputs = [
        [-0.5, -1.5, -2.5],
        [0.0, 0.0, 0.0],
        [-10.0, -20.0, -30.0, -40.0],
    ]
    shifts = [0.0, 5.0, -100.0, 1000.0]

    for base in inputs:
        obs_base = default_observable(base)
        for shift in inputs:
            pass  # placeholder
        for c in shifts:
            shifted = [v + c for v in base]
            obs_shifted = default_observable(shifted)
            assert obs_shifted.entropy_ratio == pytest.approx(
                obs_base.entropy_ratio
            ), f"entropy_ratio changed with shift {c} on {base}"
            assert obs_shifted.margin_ratio == pytest.approx(
                obs_base.margin_ratio
            ), f"margin_ratio changed with shift {c} on {base}"
            assert obs_shifted.combined == pytest.approx(
                obs_base.combined
            ), f"combined changed with shift {c} on {base}"


# ---------------------------------------------------------------------------
# Adversarial tests (cat 6)
# ---------------------------------------------------------------------------


def test_empty_list():
    """Empty topk_logprobs: entropy=0, margin=0, combined=0.5."""
    obs = default_observable([])
    assert obs.entropy_ratio == 0.0
    assert obs.margin_ratio == 0.0
    assert obs.combined == pytest.approx(0.5)


def test_single_element():
    """Single element: entropy=0, margin=0, combined=0.5."""
    obs = default_observable([-1.0])
    assert obs.entropy_ratio == 0.0
    assert obs.margin_ratio == 0.0
    assert obs.combined == pytest.approx(0.5)


def test_all_identical_spread_zero():
    """All identical values (spread=0): margin_ratio=0, entropy_ratio=1."""
    obs = default_observable([-3.0, -3.0, -3.0])
    assert obs.margin_ratio == 0.0
    assert obs.entropy_ratio == pytest.approx(1.0)


def test_nan_raises_valueerror():
    """NaN in input must raise ValueError, not produce NaN output."""
    with pytest.raises(ValueError, match="non-finite"):
        default_observable([float("nan"), -1.0])


def test_inf_raises_valueerror():
    """+Inf in input must raise ValueError."""
    with pytest.raises(ValueError, match="non-finite"):
        default_observable([float("inf"), -1.0])


def test_negative_inf_raises_valueerror():
    """-Inf in input must raise ValueError."""
    with pytest.raises(ValueError, match="non-finite"):
        default_observable([float("-inf"), -1.0])
