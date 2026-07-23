"""Integration test: real GGUF model end-to-end (TEST_TAXONOMY cat 1+6).

Skipped in CI when the model file is not present.
"""

from pathlib import Path

import pytest

from cusum_watch.calibration.generate import generate_calibration_set

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

requires_model = pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason=f"GGUF model not found at {MODEL_FILE} — run: python scripts/fetch_reference_model.py",
)


@requires_model
def test_generate_single_sample():
    """Load real model, generate one sample, verify structure."""
    samples = generate_calibration_set(
        model_path=str(MODEL_FILE),
        prompts=["The capital of France is"],
        k=5,
        max_new_tokens=32,
    )

    assert len(samples) == 1
    sample = samples[0]

    assert sample.prompt == "The capital of France is"
    assert len(sample.tokens) > 0, "should generate at least one token"
    assert len(sample.logprobs) == len(sample.tokens)
    assert len(sample.topk_logprobs) == len(sample.tokens)

    for step_topk in sample.topk_logprobs:
        assert len(step_topk) <= 5
        # Logprobs should be non-positive
        assert all(lp <= 0 for lp in step_topk)

    # llama-cpp-python doesn't expose hidden states
    assert sample.hidden_state_deltas is None


@requires_model
def test_generate_multiple_prompts():
    """Generate calibration set from multiple prompts."""
    prompts = ["Hello", "The number 1 2 3 is"]
    samples = generate_calibration_set(
        model_path=str(MODEL_FILE),
        prompts=prompts,
        k=3,
        max_new_tokens=16,
    )

    assert len(samples) == len(prompts)
    for sample, expected_prompt in zip(samples, prompts):
        assert sample.prompt == expected_prompt
        assert len(sample.tokens) > 0
        assert len(sample.logprobs) == len(sample.tokens)
        assert sample.hidden_state_deltas is None
