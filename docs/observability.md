# Observability Guide

## Metrics

cusum-watch exposes three Prometheus metrics via /metrics:

### cusum_watch_alarms_total

Counter tracking total CUSUM alerts raised.

- Labels: model (model name), direction (positive for entropy-spike, negative for repetition-collapse)
- Use: rate(cusum_watch_alarms_total[5m]) for alarm rate

### cusum_watch_mean_time_to_detect_tokens

Histogram recording triggered_at_step for every alert.

- Labels: model
- Buckets: 1, 2, 5, 10, 20, 50, 100, 200, 500
- Use: histogram_quantile(0.50, ...) for median, histogram_quantile(0.95, ...) for p95

### cusum_watch_calibration_drift

Gauge showing KS-test statistic between null model and live combined values.

- Labels: model
- Warning threshold: 0.2 (yellow) - live data has drifted from null model. Re-run calibrate_threshold.
- Critical threshold: 0.4 (red) - significant drift detected.

## Dashboard

Import dashboards/cusum-watch.json into Grafana via Dashboards > Import.

## Known Limitations

- In-memory only: metrics reset on proxy restart. Not suitable for historical analysis.
- Rolling window: calibration_drift uses last N combined values (default 1000).
