"""litellm proxy hook for cusum-watch.

Integrates the two-sided CUSUM drift monitor into litellm's custom logger
interface. Maintains per-request CusumState, cleans up on completion.

litellm is an optional dependency — this module can be imported without it
for testing, but CusumWatchLogger requires litellm at runtime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from cusum_watch.observable.compute import ObservableFn, default_observable
from cusum_watch.stats.cusum import CusumState, ECusum
from cusum_watch.stats.null_model import NullModel

logger = logging.getLogger(__name__)


@dataclass
class TwoSidedCusumState:
    positive: CusumState  # detects entropy-spike-like drift (alt_shift > 0)
    negative: CusumState  # detects repetition-collapse-like drift (alt_shift < 0)


@dataclass
class MonitorConfig:
    null_model_path: str
    threshold_positive: float  # independently calibrated for positive direction
    threshold_negative: float  # independently calibrated for negative direction
    degrade_to_logprob_only: bool = False
    alert_webhook: str | None = None
    alt_shift: float = 0.002  # magnitude of shift for both directions (calibration investigation: shift>=0.01 gives FAR>15%; shift=0.002 gives FAR~2%/8% pos/neg)


@dataclass
class CusumWatchAlert:
    """Alert raised by the monitoring hook."""

    request_id: str
    triggered_at_step: int
    threshold: float
    trace: list[float]
    direction: str  # "positive" or "negative"


class CusumWatchLogger:
    """Two-sided CUSUM drift monitor for litellm.

    Maintains one TwoSidedCusumState per active request_id.
    State is cleaned up in async_log_success_event.
    """

    def __init__(self, config: MonitorConfig, null_model: NullModel,
                 observable_fn: ObservableFn | None = None):
        self.config = config
        self.null_model = null_model

        # Select observable implementation
        if observable_fn is not None:
            self.observable_fn = observable_fn
        elif config.degrade_to_logprob_only:
            self.observable_fn = default_observable
        else:
            raise NotImplementedError(
                "degrade_to_logprob_only=False requires an alternative "
                "ObservableFn injection. No hidden-state-aware implementation "
                "is available because no CPU-only backend exposes hidden states. "
                "Pass observable_fn= to CusumWatchLogger, or set "
                "degrade_to_logprob_only=True."
            )

        self.cusum_positive = ECusum(
            null=null_model,
            threshold=config.threshold_positive,
            alt_shift=config.alt_shift,
        )
        self.cusum_negative = ECusum(
            null=null_model,
            threshold=config.threshold_negative,
            alt_shift=-config.alt_shift,
        )
        self._active_states: dict[str, TwoSidedCusumState] = {}

    def _get_or_create_state(self, request_id: str) -> TwoSidedCusumState:
        """Get or initialize the two-sided CUSUM state for a request."""
        if request_id not in self._active_states:
            self._active_states[request_id] = TwoSidedCusumState(
                positive=CusumState(),
                negative=CusumState(),
            )
        return self._active_states[request_id]

    def _process_observables(
        self, request_id: str, topk_logprobs_list: list[list[float]]
    ) -> list[CusumWatchAlert]:
        """Process a sequence of topk_logprobs through both CUSUM directions."""
        state = self._get_or_create_state(request_id)
        alerts: list[CusumWatchAlert] = []

        for topk in topk_logprobs_list:
            obs = self.observable_fn(topk)

            # Positive direction (entropy increase)
            state.positive, alert_pos = self.cusum_positive.update(
                state.positive, obs.combined, request_id, direction="positive"
            )
            if alert_pos is not None:
                alerts.append(
                    CusumWatchAlert(
                        request_id=alert_pos.request_id,
                        triggered_at_step=alert_pos.triggered_at_step,
                        threshold=alert_pos.threshold,
                        trace=alert_pos.trace,
                        direction="positive",
                    )
                )

            # Negative direction (repetition collapse)
            state.negative, alert_neg = self.cusum_negative.update(
                state.negative, obs.combined, request_id, direction="negative"
            )
            if alert_neg is not None:
                alerts.append(
                    CusumWatchAlert(
                        request_id=alert_neg.request_id,
                        triggered_at_step=alert_neg.triggered_at_step,
                        threshold=alert_neg.threshold,
                        trace=alert_neg.trace,
                        direction="negative",
                    )
                )

        return alerts

    def _dispatch_alerts(self, alerts: list[CusumWatchAlert]) -> None:
        """Dispatch alerts: log structurally and POST to webhook if configured."""
        for alert in alerts:
            logger.warning(
                "CUSUM drift alert: request_id=%s step=%d direction=%s threshold=%.4f",
                alert.request_id,
                alert.triggered_at_step,
                alert.direction,
                alert.threshold,
            )
            if self.config.alert_webhook:
                try:
                    import urllib.request

                    payload = json.dumps(
                        {
                            "request_id": alert.request_id,
                            "triggered_at_step": alert.triggered_at_step,
                            "direction": alert.direction,
                            "threshold": alert.threshold,
                        }
                    ).encode()
                    req = urllib.request.Request(
                        self.config.alert_webhook,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=5)
                except Exception:
                    logger.exception("Failed to send alert to webhook")

    def cleanup_request(self, request_id: str) -> None:
        """Remove state for a completed request."""
        self._active_states.pop(request_id, None)

    def async_log_pre_api_call(
        self, model: Any, messages: Any, kwargs: dict
    ) -> None:
        """Hook called before an API call. Initializes state for the request."""
        request_id = kwargs.get("request_id", id(kwargs))
        self._get_or_create_state(str(request_id))

    def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Hook called after a successful API call.

        Extracts topk_logprobs from the response, runs the CUSUM pipeline,
        dispatches any alerts, and cleans up per-request state.
        """
        request_id = str(kwargs.get("request_id", id(kwargs)))

        # Extract topk_logprobs from response
        topk_logprobs_list = self._extract_topk_logprobs(response_obj)

        if topk_logprobs_list:
            alerts = self._process_observables(request_id, topk_logprobs_list)
            if alerts:
                self._dispatch_alerts(alerts)

        # Clean up state — no unbounded growth
        self.cleanup_request(request_id)

    def _extract_topk_logprobs(self, response_obj: Any) -> list[list[float]]:
        """Extract topk_logprobs from a litellm response object.

        Handles both the standard litellm response format and
        StreamingChoices format. Returns empty list if extraction fails.
        """
        try:
            choices = getattr(response_obj, "choices", None)
            if choices is None and isinstance(response_obj, dict):
                choices = response_obj.get("choices", [])

            if not choices:
                return []

            all_topk = []
            for choice in choices:
                # Try to get logprobs from the choice
                logprobs_data = None
                if hasattr(choice, "logprobs") and choice.logprobs is not None:
                    logprobs_data = (
                        choice.logprobs
                        if isinstance(choice.logprobs, dict)
                        else getattr(choice.logprobs, "content", None)
                    )
                elif isinstance(choice, dict):
                    logprobs_data = choice.get("logprobs")

                if logprobs_data is None:
                    continue

                # Extract top-k logprobs per token
                token_logprobs = None
                if isinstance(logprobs_data, dict):
                    token_logprobs = logprobs_data.get("token_logprobs")
                    top_logprobs = logprobs_data.get("top_logprobs")
                else:
                    token_logprobs = getattr(logprobs_data, "token_logprobs", None)
                    top_logprobs = getattr(logprobs_data, "top_logprobs", None)

                if top_logprobs:
                    for step_top in top_logprobs:
                        if isinstance(step_top, dict):
                            vals = sorted(step_top.values(), reverse=True)
                        else:
                            vals = sorted(
                                getattr(step_top, "values", lambda: [])(),
                                reverse=True,
                            )
                        all_topk.append(vals)

            return all_topk
        except Exception:
            logger.debug("Failed to extract topk_logprobs from response")
            return []
