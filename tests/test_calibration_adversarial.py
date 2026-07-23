"""Adversarial / edge-case tests for calibration generation (TEST_TAXONOMY cat 6)."""

from unittest.mock import MagicMock, patch

from cusum_watch.calibration.generate import generate_calibration_set


def test_empty_prompts_returns_empty_list():
    """generate_calibration_set with empty prompts returns [] without loading model."""
    result = generate_calibration_set(model_path="dummy.gguf", prompts=[])
    assert result == []


def test_max_new_tokens_zero_handled_cleanly():
    """max_new_tokens=0 produces a sample with empty generated tokens."""
    mock_llm = MagicMock()
    mock_llm.return_value = {
        "choices": [
            {
                "logprobs": {
                    "tokens": [],
                    "token_logprobs": [],
                    "top_logprobs": [],
                }
            }
        ]
    }

    with patch("cusum_watch.calibration.generate.Llama", return_value=mock_llm):
        result = generate_calibration_set(
            model_path="dummy.gguf",
            prompts=["Hello"],
            max_new_tokens=0,
        )

    assert len(result) == 1
    sample = result[0]
    assert sample.prompt == "Hello"
    assert sample.tokens == []
    assert sample.logprobs == []
    assert sample.topk_logprobs == []
    assert sample.hidden_state_deltas is None


def test_single_token_generation():
    """Model returning a single token is handled correctly."""
    mock_llm = MagicMock()
    mock_llm.return_value = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["Hi"],
                    "token_logprobs": [-0.42],
                    "top_logprobs": [{"Hi": -0.42, "Hello": -1.2, "Hey": -2.0}],
                }
            }
        ]
    }

    with patch("cusum_watch.calibration.generate.Llama", return_value=mock_llm):
        result = generate_calibration_set(
            model_path="dummy.gguf",
            prompts=["Greet"],
            k=3,
            max_new_tokens=1,
        )

    assert len(result) == 1
    sample = result[0]
    assert sample.tokens == ["Hi"]
    assert sample.logprobs == [-0.42]
    assert len(sample.topk_logprobs) == 1
    assert sample.topk_logprobs[0] == [-0.42, -1.2, -2.0]
    assert sample.hidden_state_deltas is None


def test_logprobs_none_from_model():
    """When model returns None for logprobs, sample has empty lists."""
    mock_llm = MagicMock()
    mock_llm.return_value = {
        "choices": [{"logprobs": None}]
    }

    with patch("cusum_watch.calibration.generate.Llama", return_value=mock_llm):
        result = generate_calibration_set(
            model_path="dummy.gguf",
            prompts=["Test"],
        )

    assert len(result) == 1
    assert result[0].tokens == []
    assert result[0].logprobs == []
    assert result[0].topk_logprobs == []
