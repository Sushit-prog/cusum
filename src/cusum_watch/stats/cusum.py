"""e-CUSUM engine for cusum-watch.

Implements the cumulative sum change-point detection recursion.
The cumulative statistic never goes negative (reset-at-zero form).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cusum_watch.stats.null_model import NullModel, null_loglik_ratio


@dataclass
class CusumState:
    cumulative: float = 0.0
    step_count: int = 0
    trace: list[float] = field(default_factory=list)


@dataclass
class CusumAlert:
    request_id: str
    triggered_at_step: int
    threshold: float
    trace: list[float]


class ECusum:
    """e-CUSUM change-point detector.

    One ECusum instance is stateless with respect to individual requests.
    The per-request CusumState is threaded through explicitly by the caller.
    """

    def __init__(self, null: NullModel, threshold: float, alt_shift: float):
        self.null = null
        self.threshold = threshold
        self.alt_shift = alt_shift

    def update(
        self, state: CusumState, observable: float, request_id: str = ""
    ) -> tuple[CusumState, CusumAlert | None]:
        """Process one observable value and return updated state + optional alert.

        The recursion: S_t = max(0, S_{t-1} + null_loglik_ratio(...)).
        State does NOT reset on alert — the caller decides what to do.
        """
        increment = null_loglik_ratio(observable, self.null, self.alt_shift)
        new_cumulative = max(0.0, state.cumulative + increment)

        new_state = CusumState(
            cumulative=new_cumulative,
            step_count=state.step_count + 1,
            trace=state.trace + [new_cumulative],
        )

        alert = None
        if new_cumulative >= self.threshold:
            alert = CusumAlert(
                request_id=request_id,
                triggered_at_step=new_state.step_count,
                threshold=self.threshold,
                trace=new_state.trace,
            )

        return new_state, alert
