# VERSIONING_POLICY.md — cusum-watch

Pre-1.0 semver, same rule as sop-runtime: before 1.0.0, a MINOR version bump
MAY include breaking changes to the public API (everything enumerated in
INTERFACES.md). PATCH is reserved for behavior-preserving fixes only —
including calibration/threshold recomputation that doesn't change a function
signature.

## What counts as public API
Every class/function signature listed in INTERFACES.md §1-8. Internal helpers
not listed there are not covered and may change freely within a milestone.

## Release checklist (applies from M13/M15 onward)
1. `check_api_compat.py` run clean against previous tag.
2. CHANGELOG.md entry stating what changed and why, including any accuracy/
   calibration numbers that changed (e.g. a re-fit null model shifting the
   achieved false-alarm rate).
3. Clean-venv install + `cusum-watch --version` verification before tagging,
   not after — same lesson as sop-runtime's PyPI Trusted Publishing failures.
4. No stray debug artifacts or `.orig` files in the tree (`*.orig` stays in
   .gitignore from M1).

## First release target
0.1.0 at M15 unless a milestone before then replaces a public API surface
introduced in an earlier milestone (e.g. if M6's litellm hook signature
changes materially after M9's dashboard work reveals a gap) — in which case
jump straight to 0.2.0 with the reason documented, same as sop-runtime.
