# cusum-watch

Decoding-time drift monitor for quantized LLMs. Uses a quantization-robust observable derived from per-token log-probabilities to detect anomalous generation patterns (repetition collapse, degenerate looping) via e-CUSUM, without relying on raw logprob thresholding.

**Current limitation**: This project only supports logprob-only monitoring because no available CPU-only inference backend (llama-cpp-python) exposes hidden states. The `ObservableFn` interface is designed so a future backend swap — e.g. one that provides hidden-state deltas — can be plugged in without modifying `stats/` or `proxy/` modules. Set `degrade_to_logprob_only=True` (the only supported mode) or inject a custom `ObservableFn` into `CusumWatchLogger`.
