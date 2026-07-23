"""Synthetic drift-injection framework for cusum-watch.

Perturbs real calibration samples at a known step to simulate off-distribution
failures, enabling end-to-end detection testing of the observable -> null ->
CUSUM pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cusum_watch.calibration.generate import CalibrationSample


@dataclass
class DriftInjectionResult:
    original_sample: CalibrationSample
    injected_sample: CalibrationSample
    injection_step: int
    injection_kind: str


class InjectionFn(Protocol):
    def __call__(
        self, sample: CalibrationSample, injection_step: int
    ) -> CalibrationSample: ...


def _copy_sample_with_new_topk(
    sample: CalibrationSample, new_topk: list[list[float]]
) -> CalibrationSample:
    """Return a new CalibrationSample with replaced topk_logprobs."""
    return CalibrationSample(
        prompt=sample.prompt,
        tokens=list(sample.tokens),
        logprobs=list(sample.logprobs),
        topk_logprobs=new_topk,
        hidden_state_deltas=sample.hidden_state_deltas,
    )


def inject_repetition_collapse(
    sample: CalibrationSample, injection_step: int
) -> CalibrationSample:
    """Simulate repetition collapse: top-1 becomes near-certain.

    From injection_step onward, sets topk_logprobs[0] to -0.01 (high
    confidence) and all others to -10.0 (very unlikely). This simulates
    a model stuck confidently repeating one token.
    """
    new_topk = [list(step) for step in sample.topk_logprobs]
    for i in range(injection_step, len(new_topk)):
        if len(new_topk[i]) >= 1:
            new_topk[i][0] = -0.01
        for j in range(1, len(new_topk[i])):
            new_topk[i][j] = -10.0
    return _copy_sample_with_new_topk(sample, new_topk)


def inject_entropy_spike(
    sample: CalibrationSample, injection_step: int, magnitude: float = 3.0
) -> CalibrationSample:
    """Simulate entropy spike: all top-k values become equal (uniform).

    From injection_step onward, sets all topk values to the same value,
    flattening the distribution to maximum entropy. Simulates the model
    losing confidence / going incoherent.
    """
    new_topk = [list(step) for step in sample.topk_logprobs]
    for i in range(injection_step, len(new_topk)):
        flat_val = -1.0
        new_topk[i] = [flat_val] * len(new_topk[i])
    return _copy_sample_with_new_topk(sample, new_topk)


def inject_degenerate_flattening(
    sample: CalibrationSample, injection_step: int, magnitude: float = 3.0
) -> CalibrationSample:
    """Negative control: uniform additive shift to all top-k values.

    From injection_step onward, adds `magnitude` to every value in each
    step's topk_logprobs. This is exactly the kind of perturbation M2's
    entropy_ratio / margin_ratio are designed to be invariant to. The
    pipeline should NOT flag this as drift.
    """
    new_topk = [list(step) for step in sample.topk_logprobs]
    for i in range(injection_step, len(new_topk)):
        new_topk[i] = [v + magnitude for v in new_topk[i]]
    return _copy_sample_with_new_topk(sample, new_topk)
