# Contributing to mdrouter

Thanks for contributing.

## Development setup

1. Create and activate a virtual environment.
2. Install project dependencies in editable mode.
3. Run tests before opening a pull request.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files
pytest
```

## Branch and commit style

- Use short feature branches: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.
- Keep commits small and single-purpose.
- Use clear commit messages in imperative form.
- Open a pull request for `main` (direct pushes to `main` are blocked).

## Pull request checklist

- Describe the problem and solution clearly.
- Mention any env vars or config changes.
- Add tests for behavior changes.
- Update README for user-facing changes.
- Include log samples or command output for ops-related changes when useful.

## Code guidelines

- Prefer readability over cleverness.
- Keep provider-specific behavior isolated.
- Avoid introducing dependencies for small utility tasks.
- Never commit secrets or real API keys.

## Local verification

```bash
python3 -m mdrouter --config config/providers.json
curl http://127.0.0.1:11435/
pytest
```

## CI and required checks

CI runs automatically for pull requests targeting `main` and for pushes to `main`.
The pipeline includes:
- lint + formatting checks
- secret scanning
- dependency audit
- unit tests on Python 3.11, 3.12, and 3.13

If any check fails, the PR should not be merged until fixed.

## Security and safety

- Do not log sensitive request headers.
- Keep request and response body logging opt-in.
- Validate all external inputs before using them in shell commands or file operations.
