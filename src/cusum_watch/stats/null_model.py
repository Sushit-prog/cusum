"""Null-distribution fitting for cusum-watch.

Fits a null distribution to calibration-set observables and computes
per-step log-likelihood ratios for the CUSUM engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from cusum_watch.calibration.generate import CalibrationSample
from cusum_watch.observable.compute import default_observable


@dataclass
class NullModel:
    distribution: str  # scipy.stats distribution name
    params: dict[str, float]  # fitted parameters
    fit_diagnostics: dict  # KS stat, p-value, sample size, candidates tried, selected, reason


def combined_values_from_calibration_set(samples: list[CalibrationSample]) -> list[float]:
    """Extract all combined observable values from a calibration set.

    For each sample, for each step's topk_logprobs, computes
    default_observable and collects the .combined value.
    """
    values: list[float] = []
    for sample in samples:
        for topk in sample.topk_logprobs:
            obs = default_observable(topk)
            values.append(obs.combined)
    return values


def fit_null(observables: list[float]) -> NullModel:
    """Fit a null distribution to a list of combined observable values.

    Tries beta and norm distributions from scipy.stats, selects the one
    with the higher KS-test p-value.

    Raises ValueError if fewer than 30 observables or near-zero variance.
    """
    if len(observables) < 30:
        raise ValueError(
            f"Need at least 30 observables to fit a stable null model, got {len(observables)}"
        )

    arr = np.array(observables, dtype=float)
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        raise ValueError("observables contains NaN or Inf values")

    std = float(np.std(arr))
    if std < 1e-10:
        raise ValueError(
            f"Near-zero variance in observables (std={std:.2e}), "
            "cannot fit a meaningful null distribution"
        )

    candidates: list[tuple[str, dict[str, float], float, float]] = []

    # Candidate 1: beta distribution
    # Clip to (0, 1) for beta support
    eps = 1e-6
    clipped = np.clip(arr, eps, 1.0 - eps)
    try:
        a, b, loc, scale = stats.beta.fit(clipped)
        ks_stat, ks_pval = stats.kstest(clipped, "beta", args=(a, b, loc, scale))
        params = {"a": float(a), "b": float(b), "loc": float(loc), "scale": float(scale)}
        candidates.append(("beta", params, float(ks_stat), float(ks_pval)))
    except Exception:
        pass

    # Candidate 2: normal distribution
    try:
        mu, sigma = stats.norm.fit(arr)
        ks_stat, ks_pval = stats.kstest(arr, "norm", args=(mu, sigma))
        params = {"loc": float(mu), "scale": float(sigma)}
        candidates.append(("norm", params, float(ks_stat), float(ks_pval)))
    except Exception:
        pass

    if not candidates:
        raise ValueError("All candidate distributions failed to fit")

    # Select by highest KS p-value
    candidates.sort(key=lambda c: c[3], reverse=True)
    selected_name, selected_params, selected_ks, selected_pval = candidates[0]

    fit_diagnostics = {
        "candidates_tried": [
            {"distribution": name, "ks_statistic": ks, "ks_pvalue": pval}
            for name, _, ks, pval in candidates
        ],
        "selected": selected_name,
        "reason": "higher KS p-value",
        "sample_size": len(observables),
        "ks_statistic": selected_ks,
        "ks_pvalue": selected_pval,
    }

    return NullModel(
        distribution=selected_name,
        params=selected_params,
        fit_diagnostics=fit_diagnostics,
    )


def null_loglik_ratio(x: float, null: NullModel, alt_shift: float) -> float:
    """Compute log-likelihood ratio of x under shifted alternative vs null.

    The shift model: the alternative hypothesis is that the null
    distribution's location parameter is shifted by `alt_shift` in the
    direction of higher combined values (higher combined = more entropy /
    less margin = more uncertainty = potential drift).

    Sign convention: positive when x looks more like drift than null,
    which is what M4's CUSUM needs (accumulates upward on evidence of
    change).
    """
    dist_name = null.distribution
    params = null.params

    if dist_name == "norm":
        null_dist = stats.norm(loc=params["loc"], scale=params["scale"])
        alt_dist = stats.norm(loc=params["loc"] + alt_shift, scale=params["scale"])
    elif dist_name == "beta":
        # Shift the location parameter of the beta distribution
        null_dist = stats.beta(
            a=params["a"], b=params["b"], loc=params["loc"], scale=params["scale"]
        )
        alt_dist = stats.beta(
            a=params["a"],
            b=params["b"],
            loc=params["loc"] + alt_shift,
            scale=params["scale"],
        )
    else:
        raise ValueError(f"Unsupported distribution: {dist_name}")

    null_pdf = null_dist.pdf(x)
    alt_pdf = alt_dist.pdf(x)

    # Handle edge cases where pdf is 0 (x outside distribution support)
    _EPS_PDF = 1e-300
    if null_pdf <= 0 and alt_pdf <= 0:
        return 0.0
    if null_pdf <= 0:
        return 500.0  # large positive: x looks like drift, not null
    if alt_pdf <= 0:
        return -500.0  # large negative: x looks like null, not drift

    return float(np.log(alt_pdf) - np.log(null_pdf))
