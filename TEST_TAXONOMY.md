# TEST_TAXONOMY.md — cusum-watch

Every milestone's tests must be classifiable into one of these categories.
A milestone prompt that only adds happy-path unit tests without touching the
categories relevant to that milestone is incomplete — flag it in the drift check.

## 1. Unit — math correctness
Pure functions in `observable/`, `stats/null_model.py`, `stats/cusum.py`.
No I/O, no model loading. Includes hand-computed CUSUM trajectories against a
known reference sequence (verify the recursion, not just "it runs").

## 2. Calibration validity
Does `calibrate_threshold` actually deliver the false-alarm rate it claims?
Run the calibrated threshold against held-out in-distribution data (not the
calibration set itself) and empirically measure the alarm rate over N trials;
assert it's within tolerance of the target. This is the test category that
justifies the word "calibrated" — without it, the threshold is just tuned.

## 3. Synthetic drift-injection
Perturb logits/hidden-state deltas of a real generation to simulate an
off-distribution failure (repetition collapse, degenerate looping, injected
noise at a known step), feed through the full pipeline, assert detection
within N tokens of the injected perturbation. This is what the "median <15
tokens" resume bullet must be backed by — a real assertion, not a demo run.

## 4. Integration — litellm hook
`CusumWatchLogger` against a mocked litellm call sequence: per-request state
isolation (two concurrent requests don't share CusumState), correct alert
payload shape, correct behavior when a request ends without triggering.

## 5. Degradation-path
Explicit test that `degrade_to_logprob_only=True` produces a valid observable
computation with `hidden_state_deltas=None`, and that this is logged/flagged
distinctly from full-signal mode (never silently identical output).

## 6. Adversarial / edge cases
Empty generations, single-token generations, prompts that never trigger,
NaN/Inf in logits (must not crash or silently produce a spurious alarm or a
silent non-alarm), calibration set too small to fit a stable null model
(must raise, not silently fit garbage).

## 7. Performance (CPU-aware)
No hard latency assertions in CI (hardware varies), but a regression guard:
per-token CUSUM update must stay O(1) — assert it doesn't scale with
sequence length, since the whole design point is a lightweight statistical
layer, not a second model.

## 8. Metrics/observability
`/metrics` endpoint returns valid Prometheus exposition format; counters
increment correctly across a scripted sequence of alarm/no-alarm requests.

## 9. API-compat (from M13 onward)
`check_api_compat.py` diffs the current public API (per INTERFACES.md)
against the previous release's signature set; must catch an intentionally
introduced breaking change (prove this once, as SOPVM did, then revert).
