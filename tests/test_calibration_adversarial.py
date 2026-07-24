"""Adversarial / edge-case tests for calibration generation (TEST_TAXONOMY cat 6)."""

import sys
from unittest.mock import MagicMock, patch

from cusum_watch.calibration.generate import generate_calibration_set


def test_empty_prompts_returns_empty_list():
    result = generate_calibration_set(model_path="dummy.gguf", prompts=[])
    assert result == []


def test_max_new_tokens_zero_handled_cleanly():
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"logprobs": {"tokens": [], "token_logprobs": [], "top_logprobs": []}}]}
    fake = MagicMock()
    fake.Llama = MagicMock(return_value=mock_llm)
    with patch.dict(sys.modules, {"llama_cpp": fake}):
        result = generate_calibration_set(model_path="dummy.gguf", prompts=["Hello"], max_new_tokens=0)
    assert len(result) == 1
    assert result[0].prompt == "Hello"
    assert result[0].tokens == []
    assert result[0].hidden_state_deltas is None


def test_single_token_generation():
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"logprobs": {"tokens": ["Hi"], "token_logprobs": [-0.42], "top_logprobs": [{"Hi": -0.42, "Hello": -1.2, "Hey": -2.0}]}}]}
    fake = MagicMock()
    fake.Llama = MagicMock(return_value=mock_llm)
    with patch.dict(sys.modules, {"llama_cpp": fake}):
        result = generate_calibration_set(model_path="dummy.gguf", prompts=["Greet"], k=3, max_new_tokens=1)
    assert len(result) == 1
    assert result[0].tokens == ["Hi"]
    assert result[0].logprobs == [-0.42]
    assert result[0].topk_logprobs[0] == [-0.42, -1.2, -2.0]
    assert result[0].hidden_state_deltas is None


def test_logprobs_none_from_model():
    mock_llm = MagicMock()
    mock_llm.return_value = {"choices": [{"logprobs": None}]}
    fake = MagicMock()
    fake.Llama = MagicMock(return_value=mock_llm)
    with patch.dict(sys.modules, {"llama_cpp": fake}):
        result = generate_calibration_set(model_path="dummy.gguf", prompts=["Test"])
    assert len(result) == 1
    assert result[0].tokens == []
    assert result[0].logprobs == []
    assert result[0].topk_logprobs == []
