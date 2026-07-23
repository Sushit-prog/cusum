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
