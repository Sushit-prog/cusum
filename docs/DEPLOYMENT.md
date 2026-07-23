# Deployment Guide

## Quick Start

### 1. Start the metrics server

    python -m cusum_watch.metrics.server

### 2. Configure Prometheus

Add scrape target pointing to localhost:9090.

### 3. Import Grafana dashboard

Dashboards > Import > upload dashboards/cusum-watch.json.

### 4. Wire CusumWatchLogger

    from cusum_watch.metrics.server import MetricsRegistry, MetricsConfig
    from cusum_watch.proxy.litellm_hook import CusumWatchLogger, MonitorConfig
    metrics = MetricsRegistry(MetricsConfig(model="default"))
    logger = CusumWatchLogger(config, null_model, metrics=metrics)

## Known Gaps

- Metrics in-memory only. Proxy restart resets counters.
- No persistence for long-term trending.
