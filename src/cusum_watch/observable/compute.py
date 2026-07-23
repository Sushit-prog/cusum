"""Quantization-robust observable computation for cusum-watch.

Computes a scale-invariant observable from per-token top-k log-probabilities.
The observable is designed to be stable across quantization-induced shifts
in logprob magnitude — it measures distributional shape, not absolute level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StepObservable:
    entropy_ratio: float  # normalized top-k Shannon entropy in [0, 1]
    margin_ratio: float  # (top1 - top2) / spread, in [0, 1]
    combined: float  # placeholder 0.5/0.5 weighting


class ObservableFn(Protocol):
    def __call__(self, topk_logprobs: list[float],
                 hidden_state_deltas: list[float] | None = None) -> StepObservable: ...


def default_observable(topk_logprobs: list[float],
                       hidden_state_deltas: list[float] | None = None) -> StepObservable:
    """Compute the quantization-robust observable from top-k logprobs.

    Parameters
    ----------
    topk_logprobs:
        Top-k log-probabilities for a single token step, sorted descending.
        Values must be finite (no NaN or Inf).
    hidden_state_deltas:
        Accepted for interface compatibility but ignored in this logprob-only
        implementation. A future backend that exposes hidden states would
        provide a different ObservableFn implementation.

    Returns
    -------
    StepObservable with:
        - entropy_ratio: Shannon entropy of the renormalized top-k
          distribution, divided by log(k) so the result is in [0, 1].
          Invariant to uniform additive shift in all k logprobs.
        - margin_ratio: (top1 - top2) / (top1 - topk). Returns 0.0 when
          fewer than 2 elements or spread is 0.
        - combined: 0.5 * entropy_ratio + 0.5 * (1 - margin_ratio).
          Placeholder weighting — will be replaced by calibration-fit
          combination in M3.
    """
    for v in topk_logprobs:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"topk_logprobs contains non-finite value: {v}")

    k = len(topk_logprobs)

    # --- entropy_ratio ---
    if k <= 1:
        entropy_ratio = 0.0
    else:
        # softmax renormalization over top-k slice
        # shift by max for numerical stability (logprobs can be large negative)
        max_lp = max(topk_logprobs)
        exps = [math.exp(lp - max_lp) for lp in topk_logprobs]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]
        # Shannon entropy normalized by log(k) so result is in [0, 1]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        entropy_ratio = entropy / math.log(k)

    # --- margin_ratio ---
    if k < 2:
        margin_ratio = 0.0  # undefined with fewer than 2 elements
    else:
        spread = topk_logprobs[0] - topk_logprobs[-1]
        if spread == 0.0:
            margin_ratio = 0.0  # all values identical, no margin to measure
        else:
            margin_ratio = (topk_logprobs[0] - topk_logprobs[1]) / spread

    # --- combined ---
    # Placeholder 0.5/0.5 weighting — will be calibration-fit in M3
    combined = 0.5 * entropy_ratio + 0.5 * (1.0 - margin_ratio)

    return StepObservable(
        entropy_ratio=entropy_ratio,
        margin_ratio=margin_ratio,
        combined=combined,
    )
