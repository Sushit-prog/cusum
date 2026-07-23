"""FastAPI /metrics endpoint for cusum-watch.

Exposes Prometheus metrics for the CUSUM drift monitor. Uses the default
prometheus_client registry as the shared metrics store — both this module
and proxy/litellm_hook.py write to it, and /metrics reads from it.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, Histogram, generate_latest
from prometheus_client.parser import text_string_to_metric_families
from scipy import stats as sp_stats


@dataclass
class MetricsConfig:
    model: str = "default"
    calibration_window_size: int = 1000  # rolling window for drift computation


class MetricsRegistry:
    """Shared metrics store used by both the metrics server and the proxy hook.

    Uses a dedicated CollectorRegistry (not the global default) so tests can
    create isolated registries without cross-contamination.
    """

    def __init__(self, config: MetricsConfig | None = None):
        self.config = config or MetricsConfig()
        self.registry = CollectorRegistry()

        self.alarms_total = Gauge(
            "cusum_watch_alarms_total",
            "Total CUSUM alerts raised",
            ["model", "direction"],
            registry=self.registry,
        )

        self.detect_tokens = Histogram(
            "cusum_watch_mean_time_to_detect_tokens",
            "Tokens until detection (triggered_at_step)",
            ["model"],
            buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500],
            registry=self.registry,
        )

        self.calibration_drift = Gauge(
            "cusum_watch_calibration_drift",
            "KS-test stat between null model and live combined values",
            ["model"],
            registry=self.registry,
        )

        # Rolling window for calibration drift computation
        self._combined_window: deque[float] = deque(
            maxlen=self.config.calibration_window_size
        )

    def record_alert(self, direction: str, triggered_at_step: int) -> None:
        """Record a CUSUM alert."""
        self.alarms_total.labels(model=self.config.model, direction=direction).inc()
        self.detect_tokens.labels(model=self.config.model).observe(triggered_at_step)

    def record_combined(self, combined: float) -> None:
        """Record an observed combined value for drift monitoring."""
        self._combined_window.append(combined)

    def update_calibration_drift(
        self, null_distribution: str, null_params: dict[str, float]
    ) -> None:
        """Recompute calibration_drift gauge from the rolling window.

        Compares the null model's fitted distribution against recent live
        combined values using a KS-test.
        """
        if len(self._combined_window) < 30:
            # Not enough data for a meaningful KS-test
            self.calibration_drift.labels(model=self.config.model).set(0.0)
            return

        window = list(self._combined_window)

        if null_distribution == "norm":
            loc = null_params.get("loc", 0.0)
            scale = null_params.get("scale", 1.0)
            ks_stat, _ = sp_stats.kstest(
                window, lambda x: sp_stats.norm.cdf(x, loc=loc, scale=scale)
            )
        elif null_distribution == "beta":
            a = null_params.get("a", 1.0)
            b = null_params.get("b", 1.0)
            loc = null_params.get("loc", 0.0)
            scale = null_params.get("scale", 1.0)
            ks_stat, _ = sp_stats.kstest(
                window, lambda x: sp_stats.beta.cdf(x, a, b, loc=loc, scale=scale)
            )
        else:
            ks_stat = 0.0

        self.calibration_drift.labels(model=self.config.model).set(
            float(ks_stat) if not math.isnan(ks_stat) else 0.0
        )

    def generate_metrics(self) -> bytes:
        """Generate Prometheus exposition format."""
        return generate_latest(self.registry)

    def parse_metrics(self) -> dict:
        """Parse generated metrics back into a dict for testing.

        Keys are (name, frozenset_of_label_items) tuples for hashability.
        """
        text = self.generate_metrics().decode()
        families = text_string_to_metric_families(text)
        result = {}
        for family in families:
            for sample in family.samples:
                key = (family.name, frozenset(sample.labels.items()))
                result[key] = sample.value
        return result


# ---------------------------------------------------------------------------
# Standalone server (run with: python -m cusum_watch.metrics.server)
# ---------------------------------------------------------------------------

def create_app(registry=None):
    from fastapi import FastAPI, Response
    app = FastAPI(title='cusum-watch metrics')
    reg = registry or MetricsRegistry()

    @app.get('/metrics')
    def metrics():
        return Response(content=reg.generate_metrics(), media_type=CONTENT_TYPE_LATEST)

    return app, reg


if __name__ == '__main__':
    import uvicorn
    app, _ = create_app()
    uvicorn.run(app, host='0.0.0.0', port=9090)
