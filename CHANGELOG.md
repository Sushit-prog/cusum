# Changelog

## 0.2.0

### Added
- `proxy/litellm_hook.py`: two-sided CUSUM drift monitor (`CusumWatchLogger`, `MonitorConfig`, `TwoSidedCusumState`). Two independently calibrated thresholds for positive (entropy-spike) and negative (repetition-collapse) directions. Default alt_shift=0.002 gives FAR ~2%/8% (positive/negative).
- `drift_injection/inject.py`: synthetic drift-injection framework with three failure modes (repetition collapse, entropy spike, degenerate flattening as negative control).
- `metrics/server.py`: FastAPI `/metrics` with Prometheus metrics (alarms_total, time-to-detect histogram, calibration_drift gauge).
- `dashboards/cusum-watch.json`: Grafana dashboard with alarm rate, time-to-detect, calibration drift panels.
- `cli/main.py`: `cusum-watch` CLI with calibrate, inspect, serve-metrics subcommands.
- `scripts/check_api_compat.py`: public API compatibility checker against `api_baseline.json`.
- CI: scheduled slow tests, clean-install smoke test, missing-dependency detection.
- Adversarial validation suite (sequential FAR, edge cases, OWASP taxonomy mapping).

### Changed
- `calibrate_threshold`: rejects null models with KS p-value < 0.05 (prevents calibrating against poor fits).
- `ObservableFn` Protocol extended with optional `hidden_state_deltas` for future backend compatibility.
- `CusumWatchLogger` accepts injectable `ObservableFn` via dependency injection.
- INTERFACES.md reconciled against actual code (M13 audit).

### Known Limitations
- Logprob-only monitoring (no hidden-state backend available).
- i.i.d. bootstrap calibration underestimates positive-direction FAR on sequential data (2.6x divergence in M12).
- In-memory metrics only — reset on proxy restart.
- alt_shift must be <= 0.02 (0.2x null std) for calibration to work; larger values break the threshold.

## 0.1.0 (initial)

### Added
- Package scaffold with `calibration/`, `observable/`, `stats/`, `proxy/`, `metrics/`, `cli/` module layout.
- `calibration/generate.py`: CPU-only INT4 GGUF calibration-set generator via llama-cpp-python.
- `observable/compute.py`: quantization-robust observable (`StepObservable`) — entropy ratio + margin ratio, scale-invariant to uniform logprob shifts.
- `stats/null_model.py`: null-distribution fitting with KS-test candidate selection, `null_loglik_ratio`, `combined_values_from_calibration_set` helper.
- `stats/cusum.py`: e-CUSUM engine with reset-at-zero recursion.
- `calibration/threshold.py`: conformal-style threshold calibration with bootstrap simulation and held-out validation.
- `scripts/fetch_reference_model.py`: downloads Qwen2.5-1.5B-Instruct Q4_K_M GGUF.
- GitHub Actions CI with fast/slow test split.
