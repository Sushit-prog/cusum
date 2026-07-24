"""Calibration-set generator for cusum-watch.

Uses llama-cpp-python for CPU-only INT4 GGUF inference to extract per-token
log-probabilities from a configurable reference model.

NOTE: hidden_state_deltas is None for all samples because llama-cpp-python
does not expose intermediate hidden states at the Python API level. This means
M7's degradation path (logprob-only observable) is the default path from day
one, not the exception. To get hidden-state-based observables, a different
backend (e.g. TransformerLens, or a custom llama.cpp fork) would be needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CalibrationSample:
    prompt: str
    tokens: list[str]
    logprobs: list[float]  # raw logprob of chosen token per step
    topk_logprobs: list[list[float]]  # top-k logprob per step, k configurable
    hidden_state_deltas: list[float] | None  # None — model doesn't expose hidden states


def generate_calibration_set(
    model_path: str,
    prompts: list[str],
    k: int = 10,
    max_new_tokens: int = 256,
    n_ctx: int = 512,
) -> list[CalibrationSample]:
    """Generate a calibration set by running each prompt through the model.

    Uses llama-cpp-python with ``logits_all=True`` and ``logprobs=k`` to
    extract per-token log-probabilities.

    Parameters
    ----------
    model_path:
        Path to a GGUF model file.
    prompts:
        List of prompt strings. Empty list returns ``[]`` without loading
        the model.
    k:
        Number of top log-probabilities to record per token step.
    max_new_tokens:
        Maximum tokens to generate per prompt.
    n_ctx:
        Context window size. Default 512 is sufficient for typical
        calibration prompts + max_new_tokens. Larger values increase
        memory usage proportionally (n_ctx * vocab_size * 4 bytes).
    """
    if not prompts:
        return []

    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python is required for calibration. "
            "Install with: pip install cusum-watch[calibration]"
        )

    llm = Llama(model_path=model_path, logits_all=True, n_ctx=n_ctx, verbose=False)

    samples = []
    for prompt in prompts:
        output = llm(prompt, max_tokens=max_new_tokens, logprobs=k, echo=False)
        choice = output["choices"][0]
        logprobs_data = choice.get("logprobs")

        if logprobs_data is None or not logprobs_data["tokens"]:
            samples.append(
                CalibrationSample(
                    prompt=prompt,
                    tokens=[],
                    logprobs=[],
                    topk_logprobs=[],
                    hidden_state_deltas=None,
                )
            )
            continue

        tokens = logprobs_data["tokens"]
        raw_logprobs = logprobs_data["token_logprobs"]
        top_logprobs = logprobs_data["top_logprobs"]  # list of dicts {token: logprob}

        # Extract top-k logprob values per step (sorted descending by value)
        topk: list[list[float]] = []
        for step_top in top_logprobs:
            sorted_vals = sorted(step_top.values(), reverse=True)[:k]
            topk.append(sorted_vals)

        samples.append(
            CalibrationSample(
                prompt=prompt,
                tokens=tokens,
                logprobs=raw_logprobs,
                topk_logprobs=topk,
                hidden_state_deltas=None,
            )
        )

    return samples


def save_calibration_set(samples: list[CalibrationSample], path: str) -> None:
    """Serialize calibration samples to JSON."""
    data = [asdict(s) for s in samples]
    Path(path).write_text(json.dumps(data, indent=2))


def load_calibration_set(path: str) -> list[CalibrationSample]:
    """Deserialize calibration samples from JSON."""
    data = json.loads(Path(path).read_text())
    return [CalibrationSample(**d) for d in data]
