# mdrouter — OpenAI/Ollama/Anthropic-compatible multi-provider LLM router

## Project

Python 3.11+ async LLM router using FastAPI + httpx + Pydantic. Routes chat/generate requests across multiple providers (Novita, Go, Anthropic, Ark, Fireworks, DeepSeek, Zen) via provider-prefixed model aliases (e.g. `go/qwen3.5-plus`). Exposes Ollama-compatible (`/api/*`), OpenAI-compatible (`/v1/chat/completions`), and Anthropic-compatible (`/v1/messages`) endpoints with streaming support. Includes an operator CLI (`mdrouterctl`) for request/cache/token visibility from JSONL logs.

Entry points:
- `mdrouter.main:main` → FastAPI server on `0.0.0.0:11435` (default)
- `mdrouter.ops:main` → CLI for status, stats, cache inspection

## Commands

| Action | Command |
|---|---|
| Setup | `make setup` (venv + pip install -e ".[dev]") |
| Run server | `make run` (or `python -m mdrouter --config config/providers.json`) |
| Run tests | `make test` (or `python -m pytest -q`) |
| Lint + format | `make precommit` (ruff check+format, pip-audit, pytest) |
| Lint only | `ruff check .` |
| Format only | `ruff format .` |
| Audit deps | `make audit` (pip-audit) |

Most commands use `.venv/bin/python` under the hood.

## Architecture

```
main.py          — FastAPI app, route handlers, request/response marshalling (67 KB, largest file)
router.py        — ModelRouter: provider selection, model alias resolution, retry/failover logic
runtime.py       — Request lifecycle: logging, caching, streaming orchestration
config.py        — AppConfig + ServerConfig/ProviderConfig/ModelConfig via pydantic-settings + JSON
models.py        — Pydantic request/response models (Ollama, OpenAI, Anthropic shapes)
adapters/
  base.py        — ProviderAdapter ABC (chat_once, chat_stream)
  openai_compat.py  — OpenAI-compatible provider implementation (most providers)
  anthropic_compat.py — Anthropic-compatible wire format adapter
ops.py           — mdrouterctl CLI: status, stats, cachestatus commands (JSONL reader)
```

Config split: `config/providers.json` (routing + file references) → provider-specific files in `config/providers/*.json`. Env vars via `ROUTER_*` prefix (pydantic-settings).

Key env vars: `ROUTER_ENABLED_PROVIDERS`, `ROUTER_HOST`, `ROUTER_PORT`, `ROUTER_CACHE_*`, `ROUTER_AUTO_*`, per-provider API key vars.

## Conventions

- **Imports**: `from __future__ import annotations` in every module. Standard lib → third-party → local, groups separated by blank line.
- **Types**: Pydantic `BaseModel` for all data shapes. `dict[str, Any]` for dynamic payloads. Generous use of `Literal` and `Field(default_factory=...)`.
- **Async**: All I/O is async (`async def`, `AsyncIterator`). Tests use `pytest-asyncio` with `asyncio_mode = auto`.
- **HTTP mocking**: `respx` for upstream HTTP mocking in tests.
- **Private helpers**: `_` prefix for module-internal functions (e.g. `_env_bool`, `_read_jsonl`).
- **Error handling**: FastAPI `HTTPException` for API errors. Provider errors handled in adapters. `raise NotImplementedError` in abstract base.
- **Testing**: Functions (unit) and integration tests in `tests/`. Pytest with `-q`. Use test helper `_config()` to build minimal `AppConfig` fixtures.
- **Style**: Ruff + ruff-format (pre-commit enforced). Spaces, not tabs (default Python).
- **Language**: All agent responses must be in English. Code identifiers, file paths, shell commands, and technical terms remain in their original language.

## Security

- **`.env` contains live API keys** — `.gitignore`, `.claudeignore`, and `.cursorignore` all block it. Never read `.env` directly; use `.env.example` as the reference for available env vars.
- **`data/mcp.db` is PROTECTED** — contains irreplaceable crawled documentation. Never delete, overwrite, or DROP TABLE on this file without explicit user approval, even in YOLO mode. Use temp paths for testing. The `data/` directory itself must not be `rm -rf`'d.

## Notes

<!-- Quick-add space for per-session notes -->
