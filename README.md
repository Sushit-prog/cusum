# cusum-watch

Decoding-time drift monitor for quantized LLMs. Teams running INT4/INT8 reasoning models in production need to catch generation failures (repetition collapse, degenerate looping) before they reach users. Raw token log-probability is the wrong monitoring signal for this — quantization shifts logprob magnitudes uniformly, making threshold-based monitors either too sensitive or too blind. cusum-watch instead monitors a **quantization-robust observable**: the shape of the top-k log-probability distribution (entropy ratio + margin ratio), which is invariant to the uniform additive shifts that quantization introduces. This observable feeds into a two-sided e-CUSUM detector that catches both entropy increases (incoherence) and entropy decreases (over-confident repetition).

**Status**: Functional monitoring pipeline with calibrated thresholds, Prometheus metrics, Grafana dashboard, and CLI. Current limitation: logprob-only mode (no hidden-state backend available). Version 0.2.0.

## Install

```bash
pip install -e .
```

(Once published: `pip install cusum-watch`)

## Quick Start

### 1. Calibrate

Run the full calibration pipeline against a GGUF model:

```bash
cusum-watch calibrate --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --output calibration.json
```

This generates calibration samples, fits a null distribution, and calibrates thresholds for both positive (entropy-spike) and negative (repetition-collapse) CUSUM directions. See `python scripts/fetch_reference_model.py` to download the GGUF.

### 2. Inspect calibration results

```bash
cusum-watch inspect calibration.json
```

Prints null model distribution, both thresholds, and empirical false-alarm rates.

### 3. Wire into litellm

```python
from cusum_watch.metrics.server import MetricsRegistry, MetricsConfig
from cusum_watch.proxy.litellm_hook import CusumWatchLogger, MonitorConfig
from cusum_watch.stats.null_model import NullModel

# Load your calibrated null model (from calibration.json)
null_model = NullModel(distribution="norm", params={"loc": 0.5, "scale": 0.1}, fit_diagnostics={})

config = MonitorConfig(
    null_model_path="calibration.json",
    threshold_positive=1.5,
    threshold_negative=1.8,
    degrade_to_logprob_only=True,  # only supported mode
    alt_shift=0.002,
)
metrics = MetricsRegistry(MetricsConfig(model="my-model"))
logger = CusumWatchLogger(config, null_model, metrics=metrics)
```

### 4. Start metrics server

```bash
python -m cusum_watch.metrics.server
# Prometheus scrapes localhost:9090/metrics
# Import dashboards/cusum-watch.json into Grafana
```

## Architecture

```
Token logprobs ? default_observable() ? StepObservable.combined
                                          ?
                                    fit_null() ? NullModel
                                          ?
                         calibrate_threshold() ? thresholds
                                          ?
                    ECusum (positive) + ECusum (negative) ? CusumAlert
                                          ?
                              MetricsRegistry ? /metrics (Prometheus)
```

The two-sided CUSUM catches failure modes that shift `combined` in opposite directions: entropy spike (combined increases) and repetition collapse (combined decreases). Each direction is independently calibrated.

## Limitations

- **Logprob-only mode**: No CPU-only backend exposes hidden states. The `ObservableFn` interface supports pluggable backends, but only `default_observable` exists today.
- **i.i.d. bootstrap vs sequential data**: Calibration uses i.i.d. bootstrap resampling, which discards token-to-token correlation. On sequential data, the positive-direction FAR diverged 2.6x from bootstrap prediction (0.230 sequential vs 0.090 bootstrap); the negative direction was 0.9x. See M12's adversarial validation for details.
- **In-memory metrics**: Prometheus metrics reset on proxy restart. Not suitable for historical trending without external persistence.
- **CPU-only reference model**: Calibration requires a GGUF model file. The default (Qwen2.5-1.5B-Instruct Q4_K_M) needs ~1.2 GB RAM.

## Documentation

- [Observability Guide](docs/observability.md) — metrics, dashboard, failure-mode taxonomy
- [Deployment Guide](docs/DEPLOYMENT.md) — Prometheus/Grafana setup
- [Changelog](CHANGELOG.md) — release history

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v -m "not slow"       # fast tests (~30s)
pytest tests/ -v -m slow             # slow tests (weekly/manual)
pytest tests/ -v                     # all tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on CI structure, API compat checking, and model fetching.
