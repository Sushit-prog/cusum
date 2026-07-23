# Contributing to cusum-watch

## Running Tests

```bash
# Fast tests (~30s) — runs on every push/PR
pytest tests/ -v -m "not slow"

# Slow tests (weekly/manual) — calibration simulation-heavy
pytest tests/ -v -m slow

# All tests
pytest tests/ -v
```

## CI Structure

- **test-fast**: Runs on every push/PR. Python 3.10/3.11/3.12 matrix. Runs `pytest -m "not slow"`.
- **test-slow**: Runs weekly (Monday 6am UTC) or on manual trigger. Runs `pytest -m slow`.
- **smoke-test**: Runs on every push/PR. Fresh install + imports all key modules + CLI --help.

## API Compatibility

After any intentional change to a public signature in INTERFACES.md:

```bash
python scripts/check_api_compat.py --update-baseline
git add api_baseline.json
git commit -m "Update API baseline: <reason>"
```

The CI smoke test and local `python scripts/check_api_compat.py` will catch accidental regressions.

## Fetching the Reference Model

For model-dependent tests (integration tests):

```bash
python scripts/fetch_reference_model.py
```

Downloads Qwen2.5-1.5B-Instruct Q4_K_M (~1.2 GB) into `models/` (gitignored).

## Project Layout

```
src/cusum_watch/
  calibration/    M1: generate, M3: threshold, M5: (future)
  observable/     M2: compute
  stats/          M3: null_model, M4: cusum
  proxy/          M6: litellm_hook
  metrics/        M8: server
  cli/            M10: main
tests/            M1-M14 test files
dashboards/       M9: Grafana JSON
docs/             M8-M9: observability, deployment
scripts/          M1: fetch model, M13: check_api_compat
```
