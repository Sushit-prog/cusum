"""Conformal-style threshold calibration for cusum-watch.

Simulates CUSUM over in-distribution sequences to find a threshold that
achieves a target false-alarm rate.
"""

from __future__ import annotations

import numpy as np

from cusum_watch.stats.cusum import CusumState, ECusum
from cusum_watch.stats.null_model import NullModel


def calibrate_threshold(
    null_observables: list[float],
    target_false_alarm_rate: float,
    null_model: NullModel,
    alt_shift: float = 0.1,
    num_simulations: int = 500,
    sequence_length: int = 100,
    rng_seed: int = 42,
) -> tuple[float, dict]:
    """Find a CUSUM threshold that achieves a target false-alarm rate.

    Splits null_observables into threshold-pick and validation halves.
    Simulates CUSUM sequences over the threshold-pick half to find the
    quantile-based threshold, then validates on the held-out half.

    Parameters
    ----------
    null_observables:
        Combined observable values from in-distribution calibration data.
    target_false_alarm_rate:
        Desired false-alarm rate, e.g. 0.01 for 1%.
    null_model:
        Fitted null distribution.
    alt_shift:
        Shift parameter for the alternative hypothesis.
    num_simulations:
        Number of bootstrap sequences to simulate for threshold picking.
    sequence_length:
        Length of each simulated CUSUM sequence.
    rng_seed:
        Random seed for reproducibility.

    Returns
    -------
    (threshold, calibration_report) where calibration_report includes
    the empirical false-alarm rate measured on held-out data.
    """
    if not (0 < target_false_alarm_rate < 1):
        raise ValueError(
            f"target_false_alarm_rate must be in (0, 1), got {target_false_alarm_rate}"
        )

    if len(null_observables) < 100:
        raise ValueError(
            f"Need at least 100 null observables for stable calibration, "
            f"got {len(null_observables)}"
        )

    rng = np.random.default_rng(rng_seed)
    arr = np.array(null_observables)

    # Split into threshold-pick (first half) and validation (second half)
    mid = len(arr) // 2
    pick_data = arr[:mid]
    val_data = arr[mid:]

    # --- Pick threshold from pick_data ---
    cusum = ECusum(null=null_model, threshold=float("inf"), alt_shift=alt_shift)
    max_cums = []

    for _ in range(num_simulations):
        seq = rng.choice(pick_data, size=sequence_length, replace=True)
        state = CusumState()
        for obs in seq:
            state, _ = cusum.update(state, float(obs))
        max_cums.append(state.cumulative)

    max_cums_arr = np.array(max_cums)
    # Threshold = (1 - target_rate) quantile of max-cumulative under null
    threshold = float(np.quantile(max_cums_arr, 1.0 - target_false_alarm_rate))

    # --- Validate on val_data ---
    cusum_val = ECusum(null=null_model, threshold=threshold, alt_shift=alt_shift)
    num_alarms = 0
    val_sims = min(num_simulations, 200)

    for _ in range(val_sims):
        seq = rng.choice(val_data, size=sequence_length, replace=True)
        state = CusumState()
        for obs in seq:
            state, alert = cusum_val.update(state, float(obs))
            if alert is not None:
                num_alarms += 1
                break

    empirical_rate = num_alarms / val_sims

    calibration_report = {
        "target_false_alarm_rate": target_false_alarm_rate,
        "empirical_false_alarm_rate": empirical_rate,
        "num_simulated_sequences": num_simulations,
        "num_validation_sequences": val_sims,
        "sequence_length": sequence_length,
        "threshold": threshold,
    }

    return threshold, calibration_report
