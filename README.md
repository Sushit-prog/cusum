# cusum-watch

[![PyPI version](https://img.shields.io/pypi/v/cusum-watch)](https://pypi.org/project/cusum-watch/)
[![Python versions](https://img.shields.io/pypi/pyversions/cusum-watch)](https://pypi.org/project/cusum-watch/)
[![CI](https://github.com/Sushit-prog/cusum/actions/workflows/ci.yml/badge.svg)](https://github.com/Sushit-prog/cusum/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Decoding-time drift monitor for quantized LLMs.

## The Problem

Raw token log-probability is miscalibrated as a drift signal under INT4/INT8 quantization. Quantization shifts logprob magnitudes uniformly, making threshold-based monitors either too sensitive or too blind. This project monitors a scale-invariant observable instead: the shape of the top-k log-probability distribution (entropy ratio + margin ratio), which is invariant to the uniform additive shifts that quantization introduces. This observable feeds into a two-sided calibrated e-CUSUM detector that catches both entropy increases (incoherence) and entropy decreases (over-confident repetition).

## Install

```bash
pip install cusum-watch
```

### Development install

```bash
git clone https://github.com/Sushit-prog/cusum
cd cusum
pip install -e ".[dev]"
```

## Quick Start

### 1. Calibrate

Run the full calibration pipeline against a GGUF model:

```bash
cusum-watch calibrate --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --output calibration.json
```

This generates calibration samples, fits a null distribution, and calibrates thresholds for both positive (entropy-spike) and negative (repetition-collapse) CUSUM directions. Use `python scripts/fetch_reference_model.py` to download the default GGUF. Calibration requires `llama-cpp-python`, which builds from source — you need a C/C++ compiler and CMake (e.g. `build-essential cmake` on Debian/Ubuntu, or Xcode Command Line Tools on macOS).

### 2. Inspect calibration results

```bash
cusum-watch inspect calibration.json
```

Prints null model distribution, both thresholds, and empirical false-alarm rates.

### 3. Wire into litellm

```python
import json
from cusum_watch.metrics.server import MetricsRegistry, MetricsConfig
from cusum_watch.proxy.litellm_hook import CusumWatchLogger, MonitorConfig
from cusum_watch.stats.null_model import NullModel

# Load the null model produced by `cusum-watch calibrate`
cal = json.loads(open("calibration.json").read())
null_model = NullModel(
    distribution=cal["null_model"]["distribution"],
    params=cal["null_model"]["params"],
    fit_diagnostics=cal["null_model"]["fit_diagnostics"],
)

config = MonitorConfig(
    null_model_path="calibration.json",
    threshold_positive=cal["threshold_positive"],
    threshold_negative=cal["threshold_negative"],
    degrade_to_logprob_only=True,
    alt_shift=cal.get("alt_shift_positive", 0.002),
)
metrics = MetricsRegistry(MetricsConfig(model="my-model"))
logger = CusumWatchLogger(config, null_model, metrics=metrics)
```

### 4. Start metrics server

```bash
cusum-watch serve-metrics
```

Prometheus scrapes `localhost:9090/metrics`. Import `dashboards/cusum-watch.json` into Grafana.

## Architecture

```
Token logprobs
    |
    v
default_observable()          -- entropy ratio + margin ratio
    |
    v
fit_null()                    -- fit null distribution (scipy.stats)
    |
    v
calibrate_threshold()         -- bootstrap simulation for positive/negative
    |
    v
ECusum (positive) + ECusum (negative)
    |                           -- two-sided detector on combined observable
    v
CusumAlert                    -- direction, threshold, trace
    |
    v
MetricsRegistry -> /metrics   -- Prometheus: alarms_total, time-to-detect, calibration_drift
```

The two-sided CUSUM catches failure modes that shift the combined observable in opposite directions: entropy spike (combined increases) and repetition collapse (combined decreases). Each direction is independently calibrated.

## Limitations

These are engineering findings from validation, not aspirational notes.

- **Logprob-only mode**: No CPU-only backend exposes hidden states. The `ObservableFn` interface supports pluggable backends, but only `default_observable` exists today.
- **i.i.d. bootstrap calibration diverges from sequential data**: Calibration uses i.i.d. bootstrap resampling, which discards token-to-token correlation. On sequential data, the positive-direction false-alarm rate measured 2.6x higher than the bootstrap predicted (0.23 actual vs 0.09 predicted). The negative direction was close (0.9x).
- **In-memory metrics only**: Prometheus metrics reset on proxy restart. Not suitable for historical trending without external persistence.
- **CPU-only reference model**: Calibration requires a GGUF model file. The default (Qwen2.5-1.5B-Instruct Q4_K_M) needs ~1.2 GB RAM.

## Documentation

- [Observability Guide](docs/observability.md) — metrics, dashboard, failure-mode taxonomy
- [Deployment Guide](docs/DEPLOYMENT.md) — Prometheus/Grafana setup
- [Security Taxonomy](docs/observability.md#failure-mode-taxonomy) — OWASP LLM Top 10 mapping
- [Changelog](CHANGELOG.md) — release history
- [Contributing](CONTRIBUTING.md) — CI structure, API compat checking, model fetching

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"       # fast tests (~30s)
pytest tests/ -v -m slow             # slow tests (weekly/manual)
pytest tests/ -v                     # all tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on CI structure, API compatibility checking, and model fetching.
