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
