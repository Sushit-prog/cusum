# Changelog

## Unreleased

### Added
- Package scaffold with `calibration/`, `observable/`, `stats/`, `proxy/`, `metrics/`, `cli/` module layout.
- `calibration/generate.py`: calibration-set generator using llama-cpp-python for CPU-only INT4 GGUF inference. Extracts real per-token top-k logprobs. `hidden_state_deltas` set to `None` (llama-cpp-python does not expose hidden states).
- `scripts/fetch_reference_model.py`: downloads Qwen2.5-1.5B-Instruct Q4_K_M GGUF from HuggingFace.
- Unit, adversarial, and integration tests for calibration-set generation.
- GitHub Actions CI workflow for unit and adversarial tests.
- `observable/compute.py`: quantization-robust observable (`StepObservable`) with `entropy_ratio`, `margin_ratio`, `combined`. Scale-invariant to uniform additive logprob shifts.
- `stats/null_model.py`: null-distribution fitting (`NullModel`, `fit_null`, `null_loglik_ratio`) with candidate distribution selection via KS-test and `combined_values_from_calibration_set` helper.
- `stats/cusum.py`: e-CUSUM engine (`CusumState`, `CusumAlert`, `ECusum`) with reset-at-zero recursion.
- `calibration/threshold.py`: conformal-style threshold calibration (`calibrate_threshold`) with bootstrap simulation and held-out validation.
  - **Known limitation**: bootstrap resamples individual observable values i.i.d. with replacement, discarding real token-to-token correlation in actual generation. May over/understate the real-world false-alarm rate versus true autocorrelated sequences. Should be revisited once M5's drift-injection framework provides real sequences to validate the bootstrap assumption against.
- `drift_injection/inject.py`: synthetic drift-injection framework with three failure modes: repetition collapse (top-1 near-certain), entropy spike (uniform distribution), and degenerate flattening (uniform additive shift — negative control). End-to-end detection tests confirm the pipeline flags real drift within ~10 tokens and does NOT flag quantization-invariant perturbations. Detection results are consistent with M4's bootstrap-based expectations.
- `proxy/litellm_hook.py`: two-sided CUSUM drift monitor for litellm (`CusumWatchLogger`, `MonitorConfig`, `TwoSidedCusumState`). Two independently calibrated thresholds (`threshold_positive` for entropy-spike detection, `threshold_negative` for repetition-collapse detection). **Calibration investigation**: both directions have similar FAR at the same shift value (the negative direction is NOT inherently worse). Calibration breaks for BOTH directions at shift >= 0.01 (FAR > 15%). Default alt_shift=0.002 gives FAR ~2% positive / ~8% negative, both within 2x of 5% target. Per-request state isolation verified; state cleaned up on completion.
- **M7**: `ObservableFn` Protocol extended with optional `hidden_state_deltas` parameter for future backend compatibility. `CusumWatchLogger` now accepts injectable `ObservableFn` via dependency injection. `degrade_to_logprob_only=False` without an injected alternative raises `NotImplementedError`. **This project currently only supports logprob-only monitoring because no available CPU-only backend exposes hidden states.** The pluggable `ObservableFn` interface exists so a future backend swap doesn't require touching `stats/` or `proxy/`.
- `metrics/server.py`: FastAPI `/metrics` endpoint with Prometheus exposition format. Three metrics: `cusum_watch_alarms_total` (Counter by model+direction), `cusum_watch_mean_time_to_detect_tokens` (Histogram of triggered_at_step), `cusum_watch_calibration_drift` (Gauge of KS-test stat between null model and rolling window of live combined values). In-memory only, no persistence across restarts.
- dashboards/cusum-watch.json: Grafana dashboard with alarm rate, time-to-detect histogram, calibration drift gauge, and per-model breakdown. Calibration drift warning threshold at 0.2 (KS-stat) based on M8 test numbers.
- docs/observability.md and docs/DEPLOYMENT.md: metrics documentation and deployment walkthrough. Metrics server now has __main__ block for standalone execution (python -m cusum_watch.metrics.server).
- cli/main.py: cusum-watch CLI with calibrate, inspect, serve-metrics subcommands. calibrate runs full pipeline (M1-M4) with independent positive/negative threshold calibration. inspect displays calibration reports. serve-metrics wraps M9 server entry point.
- CI: added scheduled slow tests job (weekly/manual), clean-install smoke test, verified missing-dep detection works.
- tests/test_adversarial_validation.py: adversarial validation suite. Part A: sequential data FAR (positive=0.230, negative=0.130) vs bootstrap prediction (positive=0.090, negative=0.140) — positive direction shows 2.6x divergence, negative is 0.9x. Part B: repetitive-from-start, short generations, bimodal fit, KS p-value design finding (computed but ignored downstream), 50-concurrent-request isolation. Part C: OWASP LLM Top 10 taxonomy mapping.
- calibrate_threshold: added KS p-value guard (threshold 0.05) — rejects null models with statistically implausible fits. Bimodal calibration set now raises ValueError instead of silently proceeding.
## 0.2.0

### Changed
- INTERFACES.md reconciled against actual code (§2 hidden_state_deltas param, §4 CusumAlert.direction + request_id/direction on ECusum.update, §5 calibrate_threshold extra params + ks_pvalue guard, §6 CusumWatchLogger 3-arg init + MonitorConfig.alt_shift + CusumWatchAlert.direction, §7 MetricsRegistry + create_app).
- calibrate_threshold: rejects null models with KS p-value < 0.05.

### Added
- scripts/check_api_compat.py: public API compatibility checker. Introspects package signatures, compares against api_baseline.json, exits 1 on breaking changes.
- api_baseline.json: checked-in baseline for API compat checking.
