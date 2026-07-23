# Contributing to cusum-watch

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/cusum-watch.git
cd cusum-watch

# Install in development mode
pip install -e ".[dev]"

# For calibration features (requires llama-cpp-python)
pip install -e ".[calibration]"
```

## Running Tests

### Fast Tests (CI runs these on every push)

```bash
pytest tests/ -v -m "not slow"
```

These tests run in ~30 seconds and don't require a model file.

### Slow Tests (weekly/manual)

```bash
pytest tests/ -v -m slow
```

These tests require the reference GGUF model and take several minutes.

### All Tests

```bash
pytest tests/ -v
```

## CI Structure

GitHub Actions runs:
1. **Fast tests** on every push and PR
2. **Slow tests** weekly (or manually triggered)
3. **Clean-install smoke test** to verify package installs correctly
4. **API compatibility check** to detect breaking changes

## API Compatibility Checking

After making intentional API changes, update the baseline:

```bash
python scripts/check_api_compat.py --update
```

This updates `api_baseline.json` with the new API signatures. The CI will verify that future changes don't break backward compatibility.

## Fetching the Reference Model

For calibration tests, download the reference GGUF model:

```bash
pip install huggingface_hub  # if not installed
python scripts/fetch_reference_model.py
```

This downloads Qwen2.5-1.5B-Instruct Q4_K_M (~1.12 GB) to the `models/` directory.

## Code Style

- Follow existing patterns in the codebase
- Keep functions focused and small
- Use type hints consistently
- Write docstrings for public APIs

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run the fast test suite: `pytest tests/ -v -m "not slow"`
4. If you changed public APIs, run `python scripts/check_api_compat.py --update`
5. Submit a PR with a clear description of changes

## Reporting Issues

Open an issue on GitHub with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)
