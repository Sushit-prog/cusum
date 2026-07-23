"""cusum-watch CLI.

Subcommands:
  calibrate    End-to-end calibration pipeline
  inspect      Load and display a calibration output file
  serve-metrics Start the Prometheus metrics server
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


def _default_prompts() -> list[str]:
    """Small built-in prompt set for quick calibration without a file."""
    return [
        "The capital of France is",
        "Explain quantum computing in one sentence.",
        "Write a Python function to sort a list.",
        "What is the meaning of life?",
        "Summarize the theory of relativity.",
        "Write a haiku about debugging.",
        "Translate 'hello world' to French.",
        "What are the three laws of thermodynamics?",
        "Explain machine learning to a five-year-old.",
        "Write a SQL query to find the top 10 users.",
    ]


@click.group()
@click.version_option(package_name="cusum-watch")
def cli():
    """cusum-watch: decoding-time drift monitor for quantized LLMs."""


@cli.command()
@click.option("--model-path", required=True, help="Path to GGUF model file")
@click.option("--prompts-file", type=click.Path(exists=True), default=None,
              help="Text file with one prompt per line (default: built-in set)")
@click.option("--target-far", default=0.05, help="Target false-alarm rate (default: 0.05)")
@click.option("--alt-shift", default=0.002, help="Alt shift magnitude for both directions (default: 0.002)")
@click.option("--alt-shift-positive", default=None, type=float,
              help="Override alt_shift for positive direction only")
@click.option("--alt-shift-negative", default=None, type=float,
              help="Override alt_shift for negative direction only")
@click.option("--output", "-o", default="calibration.json", help="Output file path (default: calibration.json)")
def calibrate(model_path, prompts_file, target_far, alt_shift, alt_shift_positive, alt_shift_negative, output):
    """Run end-to-end calibration pipeline."""
    from cusum_watch.calibration.generate import generate_calibration_set
    from cusum_watch.calibration.threshold import calibrate_threshold
    from cusum_watch.stats.null_model import (
        NullModel,
        combined_values_from_calibration_set,
        fit_null,
    )

    # Load prompts
    if prompts_file:
        prompts = [line.strip() for line in Path(prompts_file).read_text().splitlines() if line.strip()]
    else:
        prompts = _default_prompts()

    click.echo(f"Generating calibration set from {len(prompts)} prompts...")
    samples = generate_calibration_set(model_path, prompts)

    if not samples:
        click.echo("Error: no calibration samples generated", err=True)
        sys.exit(1)

    click.echo(f"Generated {len(samples)} calibration samples")

    # Extract combined values
    combined = combined_values_from_calibration_set(samples)
    click.echo(f"Extracted {len(combined)} combined observable values")

    # Fit null model
    click.echo("Fitting null distribution...")
    null_model = fit_null(combined)
    click.echo(f"Null model: {null_model.distribution} with KS-stat {null_model.fit_diagnostics.get('ks_statistic', 'N/A')}")

    # Determine alt_shifts per direction
    shift_pos = alt_shift_positive if alt_shift_positive is not None else alt_shift
    shift_neg = alt_shift_negative if alt_shift_negative is not None else alt_shift

    # Calibrate positive direction
    click.echo(f"Calibrating threshold (positive, shift={shift_pos})...")
    thresh_pos, report_pos = calibrate_threshold(
        combined, target_far, null_model, alt_shift=shift_pos,
    )
    click.echo(f"  Threshold: {thresh_pos:.4f}, Empirical FAR: {report_pos['empirical_false_alarm_rate']:.3f}")

    # Calibrate negative direction
    click.echo(f"Calibrating threshold (negative, shift={shift_neg})...")
    thresh_neg, report_neg = calibrate_threshold(
        combined, target_far, null_model, alt_shift=-shift_neg,
    )
    click.echo(f"  Threshold: {thresh_neg:.4f}, Empirical FAR: {report_neg['empirical_false_alarm_rate']:.3f}")

    # Save output
    result = {
        "null_model": {
            "distribution": null_model.distribution,
            "params": null_model.params,
            "fit_diagnostics": null_model.fit_diagnostics,
        },
        "threshold_positive": thresh_pos,
        "threshold_negative": thresh_neg,
        "calibration_report_positive": report_pos,
        "calibration_report_negative": report_neg,
        "alt_shift_positive": shift_pos,
        "alt_shift_negative": shift_neg,
        "target_far": target_far,
        "calibration_set_size": len(combined),
    }

    Path(output).write_text(json.dumps(result, indent=2))
    click.echo(f"\nCalibration saved to {output}")


@cli.command()
@click.argument("calibration_file", type=click.Path(exists=True))
def inspect(calibration_file):
    """Display a calibration output file."""
    data = json.loads(Path(calibration_file).read_text())

    click.echo("=== cusum-watch Calibration Report ===\n")

    nm = data["null_model"]
    click.echo(f"Null model: {nm['distribution']}")
    click.echo(f"  Params: {nm['params']}")
    diagnostics = nm.get("fit_diagnostics", {})
    click.echo(f"  KS-stat: {diagnostics.get('ks_statistic', 'N/A')}")
    click.echo(f"  KS p-value: {diagnostics.get('ks_pvalue', 'N/A')}")
    click.echo(f"  Sample size: {diagnostics.get('sample_size', 'N/A')}")

    click.echo(f"\nCalibration set size: {data['calibration_set_size']}")
    click.echo(f"Target FAR: {data['target_far']}")

    click.echo(f"\nPositive direction (alt_shift={data['alt_shift_positive']}):")
    click.echo(f"  Threshold: {data['threshold_positive']:.4f}")
    rp = data["calibration_report_positive"]
    click.echo(f"  Empirical FAR: {rp['empirical_false_alarm_rate']:.3f}")
    click.echo(f"  Simulated sequences: {rp['num_simulated_sequences']}")

    click.echo(f"\nNegative direction (alt_shift={data['alt_shift_negative']}):")
    click.echo(f"  Threshold: {data['threshold_negative']:.4f}")
    rn = data["calibration_report_negative"]
    click.echo(f"  Empirical FAR: {rn['empirical_false_alarm_rate']:.3f}")
    click.echo(f"  Simulated sequences: {rn['num_simulated_sequences']}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
@click.option("--port", default=9090, help="Port to listen on (default: 9090)")
def serve_metrics(host, port):
    """Start the Prometheus metrics server."""
    import uvicorn

    from cusum_watch.metrics.server import create_app

    app, _ = create_app()
    click.echo(f"Starting metrics server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
