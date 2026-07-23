"""Test that dashboard JSON is valid and references correct metric names."""

import json
from pathlib import Path


def test_dashboard_json_is_valid():
    """Dashboard JSON parses without error."""
    path = Path(__file__).resolve().parent.parent / "dashboards" / "cusum-watch.json"
    assert path.exists(), f"Dashboard file not found: {path}"
    with open(path) as f:
        data = json.load(f)
    assert data["title"] == "cusum-watch Drift Monitor"
    assert len(data["panels"]) > 0


def test_dashboard_references_correct_metric_names():
    """Dashboard references the exact metric names from M8."""
    path = Path(__file__).resolve().parent.parent / "dashboards" / "cusum-watch.json"
    with open(path) as f:
        text = f.read()

    required_metrics = [
        "cusum_watch_alarms_total",
        "cusum_watch_mean_time_to_detect_tokens",
        "cusum_watch_calibration_drift",
    ]
    for metric in required_metrics:
        assert metric in text, f"Dashboard missing metric: {metric}"


def test_dashboard_has_model_variable():
    """Dashboard has a model template variable for per-model breakdown."""
    path = Path(__file__).resolve().parent.parent / "dashboards" / "cusum-watch.json"
    with open(path) as f:
        data = json.load(f)

    template_names = [t["name"] for t in data["templating"]["list"]]
    assert "model" in template_names, "Dashboard missing model template variable"
