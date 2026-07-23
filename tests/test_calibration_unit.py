"""Unit tests for calibration save/load round-trip (TEST_TAXONOMY cat 1)."""

from cusum_watch.calibration.generate import (
    CalibrationSample,
    load_calibration_set,
    save_calibration_set,
)


def test_save_load_round_trip(tmp_path):
    """Round-trip: hand-constructed samples survive save -> load."""
    samples = [
        CalibrationSample(
            prompt="Hello",
            tokens=["World", "!"],
            logprobs=[-0.5, -0.1],
            topk_logprobs=[[-0.5, -1.2, -2.0], [-0.1, -0.8, -1.5]],
            hidden_state_deltas=None,
        ),
        CalibrationSample(
            prompt="What is 2+2?",
            tokens=["4", "."],
            logprobs=[-0.01, -0.3],
            topk_logprobs=[[-0.01, -3.0, -4.5], [-0.3, -1.1, -2.2]],
            hidden_state_deltas=None,
        ),
        CalibrationSample(
            prompt="Short",
            tokens=["OK"],
            logprobs=[-0.2],
            topk_logprobs=[[-0.2, -1.8]],
            hidden_state_deltas=None,
        ),
    ]

    path = str(tmp_path / "calibration.json")
    save_calibration_set(samples, path)
    loaded = load_calibration_set(path)

    assert len(loaded) == len(samples)
    for orig, got in zip(samples, loaded):
        assert got.prompt == orig.prompt
        assert got.tokens == orig.tokens
        assert got.logprobs == orig.logprobs
        assert got.topk_logprobs == orig.topk_logprobs
        assert got.hidden_state_deltas == orig.hidden_state_deltas


def test_save_load_empty_list(tmp_path):
    """Empty calibration set round-trips correctly."""
    path = str(tmp_path / "empty.json")
    save_calibration_set([], path)
    loaded = load_calibration_set(path)
    assert loaded == []


def test_save_load_hidden_state_deltas_none(tmp_path):
    """hidden_state_deltas=None is preserved through round-trip."""
    samples = [
        CalibrationSample(
            prompt="test",
            tokens=["a"],
            logprobs=[-1.0],
            topk_logprobs=[[-1.0, -2.0]],
            hidden_state_deltas=None,
        ),
    ]
    path = str(tmp_path / "cal.json")
    save_calibration_set(samples, path)
    loaded = load_calibration_set(path)
    assert loaded[0].hidden_state_deltas is None
