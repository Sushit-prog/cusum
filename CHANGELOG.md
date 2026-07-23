# Changelog

## Unreleased

### Added
- Package scaffold with `calibration/`, `observable/`, `stats/`, `proxy/`, `metrics/`, `cli/` module layout.
- `calibration/generate.py`: calibration-set generator using llama-cpp-python for CPU-only INT4 GGUF inference. Extracts real per-token top-k logprobs. `hidden_state_deltas` set to `None` (llama-cpp-python does not expose hidden states).
- `scripts/fetch_reference_model.py`: downloads Qwen2.5-1.5B-Instruct Q4_K_M GGUF from HuggingFace.
- Unit, adversarial, and integration tests for calibration-set generation.
- GitHub Actions CI workflow for unit and adversarial tests.
