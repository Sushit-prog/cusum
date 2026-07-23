# cusum-watch

Decoding-time drift monitor for quantized LLMs. Uses a quantization-robust observable derived from per-token log-probabilities to detect anomalous generation patterns (repetition collapse, degenerate looping) via e-CUSUM, without relying on raw logprob thresholding.
