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

Backend: local GGUF/INT4 model via `transformers`+`optimum` or `llama-cpp-python`,
CPU-only. Reference model configurable, default target: Qwen2.5-1.5B-Instruct or
Llama-3.2-1B-Instruct, INT4.

## 2. `observable/compute.py` — the quantization-robust observable

```python
@dataclass
class StepObservable:
    entropy_ratio: float     # normalized top-k Shannon entropy (scale-invariant)
    margin_ratio: float      # (top1_logit - top2_logit) / local_logit_spread
    combined: float          # calibration-fitted linear combination of the above

class ObservableFn(Protocol):
    def __call__(self, topk_logprobs: list[float]) -> StepObservable: ...

def default_observable(topk_logprobs: list[float]) -> StepObservable: ...
```

`ObservableFn` is a Protocol, not a hard-coded function, specifically so the
choice of observable can be swapped (e.g. once the paper's exact definition is
confirmed against the PDF) without touching `stats/` or `proxy/`.

## 3. `stats/null_model.py` — null distribution fitting

```python
@dataclass
class NullModel:
    distribution: str          # scipy.stats distribution name
    params: dict[str, float]
    fit_diagnostics: dict      # KS-test stat/p-value, sample size, etc.

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

class ECusum:
    def __init__(self, null: NullModel, threshold: float, alt_shift: float): ...
    def update(self, state: CusumState, observable: float) -> tuple[CusumState, CusumAlert | None]: ...
```

## 5. `calibration/threshold.py` — conformal-style threshold calibration

```python
def calibrate_threshold(
    null_observables: list[float],
    target_false_alarm_rate: float,   # e.g. 0.01
    null_model: NullModel,
) -> tuple[float, dict]:  # (threshold, calibration_report incl. Type-I bound)
    ...
```

`calibration_report` must include the achieved/guaranteed false-alarm bound and
the sample size it was computed on — this is the artifact that justifies the
"calibrated" claim in the resume bullet, not a hand-tuned number.

## 6. `proxy/litellm_hook.py` — the deployed integration

**Two-sided monitoring (added after M5's finding):** M5's drift-injection
tests showed that different failure modes shift the observable in opposite
directions — repetition collapse (excess certainty) decreases `combined`,
while entropy spike (loss of coherence) increases it. A single one-sided
`ECusum` only catches one direction. The deployed hook therefore maintains
**two** `ECusum` instances per request — one with a positive `alt_shift`
(catches entropy-spike-like drift) and one with a negative `alt_shift`
(catches repetition-collapse-like drift) — both fed the same per-step
`combined` observable, either one able to raise an alert independently.

```python
class CusumWatchLogger(CustomLogger):  # litellm.integrations.custom_logger.CustomLogger
    def __init__(self, config: MonitorConfig): ...
    async def async_log_pre_api_call(self, model, messages, kwargs): ...
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time): ...
    # per-token streaming hook maintains a pair of CusumState (positive-shift,
    # negative-shift) per request_id — see TwoSidedCusumState below

@dataclass
class TwoSidedCusumState:
    positive: CusumState   # detects entropy-spike-like drift (alt_shift > 0)
    negative: CusumState   # detects repetition-collapse-like drift (alt_shift < 0)

@dataclass
class MonitorConfig:
    null_model_path: str
    threshold: float             # shared threshold; consider whether the two
                                  # directions need independently-calibrated
                                  # thresholds in M6 — false-alarm rate may not
                                  # be symmetric between the two ECusum instances
    degrade_to_logprob_only: bool   # True if hidden states aren't exposed by the backend
    alert_webhook: str | None

@dataclass
class CusumAlert:
    request_id: str
    triggered_at_step: int
    threshold: float
    trace: list[float]
    direction: str   # "positive" (entropy-spike-like) or "negative" (repetition-collapse-like)
```

Open question for M6 to resolve, not assume: does a single shared threshold
(calibrated once, per M4) hold for both directions, or does the asymmetry
found in M5 mean the positive- and negative-shift `ECusum` instances need
independently calibrated thresholds via two separate `calibrate_threshold`
calls? Flag this explicitly in the M6 milestone summary rather than picking
one silently.

Degradation rule (M7): if `hidden_state_deltas` is unavailable from the backend,
`observable/compute.py`'s `default_observable` falls back to a logprob-only-derived
entropy/margin computation (still not raw logprob thresholding — same statistic,
computed from fewer inputs). This must be a config flag, not a silent fallback.

## 7. `metrics/server.py` — FastAPI `/metrics`

```python
# Prometheus counters/gauges, minimum set:
# cusum_watch_alarms_total{model}
# cusum_watch_mean_time_to_detect_tokens{model}
# cusum_watch_calibration_drift{model}   # null-model KS-stat vs. live traffic sample
```

## 8. `cli/main.py`

Subcommands: `calibrate` (runs generate → fit_null → calibrate_threshold end to
end), `inspect` (dump a NullModel/threshold's calibration_report), `serve-metrics`.

## Versioning

Public API = everything in this file. See VERSIONING_POLICY.md.
