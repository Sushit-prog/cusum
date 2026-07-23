# INTERFACES.md — cusum-watch

Phase 0 contract. This defines module boundaries and public interfaces before any
milestone code is written. Any change to a signature listed here during a milestone
is a **change-control event**: update this file in the same commit and say why in
CHANGELOG.md. Milestone prompts will refuse to proceed if a diff changes an
interface here without updating this doc.

## Design invariant

Raw token log-probability is never the monitored signal. Every layer downstream of
`observable/` operates on the derived observable defined in §2. This is the paper's
central claim and the reason this project exists — it must not quietly regress back
to logprob-thresholding.

## 1. `calibration/generate.py` — calibration-set generator

```python
@dataclass
class CalibrationSample:
    prompt: str
    tokens: list[str]
    logprobs: list[float]              # raw, kept for comparison/ablation only
    topk_logprobs: list[list[float]]   # top-k logprob per step, k configurable
    hidden_state_deltas: list[float] | None  # None if model doesn't expose hidden states

def generate_calibration_set(
    model_path: str,
    prompts: list[str],
    k: int = 10,
    max_new_tokens: int = 256,
) -> list[CalibrationSample]: ...

def save_calibration_set(samples: list[CalibrationSample], path: str) -> None: ...
def load_calibration_set(path: str) -> list[CalibrationSample]: ...
```

Backend: local GGUF/INT4 model via `llama-cpp-python`, CPU-only.
Reference model configurable, default target: Qwen2.5-1.5B-Instruct Q4_K_M.

**Added in M3** (change-control):
```python
def combined_values_from_calibration_set(samples: list[CalibrationSample]) -> list[float]: ...
```

## 2. `observable/compute.py` — the quantization-robust observable

```python
@dataclass
class StepObservable:
    entropy_ratio: float     # normalized top-k Shannon entropy (scale-invariant)
    margin_ratio: float      # (top1 - top2) / spread
    combined: float          # placeholder 0.5/0.5 weighting (M2)

class ObservableFn(Protocol):
    def __call__(self, topk_logprobs: list[float],
                 hidden_state_deltas: list[float] | None = None) -> StepObservable: ...

def default_observable(topk_logprobs: list[float],
                       hidden_state_deltas: list[float] | None = None) -> StepObservable: ...
```

`ObservableFn` is a Protocol, not a hard-coded function, specifically so the
choice of observable can be swapped without touching `stats/` or `proxy/`.
The optional `hidden_state_deltas` parameter exists for future backend
compatibility (M7) — currently always ignored.

## 3. `stats/null_model.py` — null distribution fitting

```python
@dataclass
class NullModel:
    distribution: str          # scipy.stats distribution name
    params: dict[str, float]
    fit_diagnostics: dict      # KS-test stat/p-value, sample size, candidates tried

def fit_null(observables: list[float]) -> NullModel: ...
def null_loglik_ratio(x: float, null: NullModel, alt_shift: float) -> float: ...
```

## 4. `stats/cusum.py` — e-CUSUM engine

```python
@dataclass
class CusumState:
    cumulative: float
    step_count: int
    trace: list[float]         # per-step cumulative value, for the alert payload

@dataclass
class CusumAlert:
    request_id: str
    triggered_at_step: int
    threshold: float
    trace: list[float]
    direction: str             # "positive" or "negative" (added in M6)

class ECusum:
    def __init__(self, null: NullModel, threshold: float, alt_shift: float): ...
    def update(self, state: CusumState, observable: float,
               request_id: str = "", direction: str = "") -> tuple[CusumState, CusumAlert | None]: ...
```

## 5. `calibration/threshold.py` — conformal-style threshold calibration

```python
def calibrate_threshold(
    null_observables: list[float],
    target_false_alarm_rate: float,   # e.g. 0.05
    null_model: NullModel,
    alt_shift: float = 0.1,
    num_simulations: int = 500,
    sequence_length: int = 100,
    rng_seed: int = 42,
) -> tuple[float, dict]:  # (threshold, calibration_report incl. Type-I bound)
    ...
```

Raises `ValueError` if `null_model.fit_diagnostics["ks_pvalue"] < 0.05`
(M12 — rejects poor null-model fits before calibrating).

`calibration_report` must include: `target_false_alarm_rate`,
`empirical_false_alarm_rate`, `num_simulated_sequences`, `sequence_length`,
`threshold`.

## 6. `proxy/litellm_hook.py` — the deployed integration

**Two-sided monitoring (added after M5's finding):** The deployed hook maintains
**two** `ECusum` instances per request — one positive `alt_shift` (entropy-spike)
and one negative `alt_shift` (repetition-collapse) — both fed the same per-step
`combined` observable, either one able to raise an alert independently.

```python
@dataclass
class TwoSidedCusumState:
    positive: CusumState   # detects entropy-spike-like drift (alt_shift > 0)
    negative: CusumState   # detects repetition-collapse-like drift (alt_shift < 0)

@dataclass
class MonitorConfig:
    null_model_path: str
    threshold_positive: float  # independently calibrated for positive direction
    threshold_negative: float  # independently calibrated for negative direction
    degrade_to_logprob_only: bool
    alert_webhook: str | None
    alt_shift: float = 0.002   # magnitude for both directions (M6 calibrated default)

@dataclass
class CusumWatchAlert:
    request_id: str
    triggered_at_step: int
    threshold: float
    trace: list[float]
    direction: str   # "positive" or "negative"

class CusumWatchLogger:
    def __init__(self, config: MonitorConfig, null_model: NullModel,
                 observable_fn: ObservableFn | None = None,
                 metrics: object | None = None): ...
    def async_log_pre_api_call(self, model, messages, kwargs): ...
    def async_log_success_event(self, kwargs, response_obj, start_time, end_time): ...
```

Degradation rule (M7): `degrade_to_logprob_only=True` selects `default_observable`.
`degrade_to_logprob_only=False` without an injected `observable_fn` raises
`NotImplementedError`. This project currently only supports logprob-only monitoring.

## 7. `metrics/server.py` — FastAPI `/metrics`

```python
@dataclass
class MetricsConfig:
    model: str = "default"
    calibration_window_size: int = 1000

class MetricsRegistry:
    def __init__(self, config: MetricsConfig | None = None): ...
    def record_alert(self, direction: str, triggered_at_step: int) -> None: ...
    def record_combined(self, combined: float) -> None: ...
    def update_calibration_drift(self, null_distribution: str, null_params: dict[str, float]) -> None: ...
    def generate_metrics(self) -> bytes: ...

# Prometheus metrics (labeled by model):
# cusum_watch_alarms_total{model, direction}
# cusum_watch_mean_time_to_detect_tokens{model}   # Histogram
# cusum_watch_calibration_drift{model}             # Gauge (KS-stat)

def create_app(registry: MetricsRegistry | None = None) -> tuple: ...
```

Standalone server: `python -m cusum_watch.metrics.server` (port 9090).
Metrics are in-memory only — reset on restart.

## 8. `cli/main.py`

Subcommands: `calibrate`, `inspect`, `serve-metrics`.
Entry point: `cusum-watch` (via `project.scripts` in pyproject.toml).

## Versioning

Public API = everything in this file. See VERSIONING_POLICY.md.
