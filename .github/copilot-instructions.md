# mdrouter — Copilot Instructions

## Identity
OpenAI/Ollama-compatible multi-provider LLM router. Python 3.11+, FastAPI, httpx, pydantic, redis.

## Architecture (key files)
| Path | Role |
|---|---|
| `mdrouter/main.py` | FastAPI app, endpoints, request models |
| `mdrouter/router.py` | ModelRouter — alias resolution, routing, caching, fallback |
| `mdrouter/config.py` | AppConfig, ModelConfig, env-var loading |
| `mdrouter/models.py` | Pydantic request/response models |
| `mdrouter/runtime.py` | RequestLogger, ResponseCache, RuntimeSettings |
| `mdrouter/adapters/base.py` | ProviderAdapter ABC |
| `mdrouter/adapters/openai_compat.py` | OpenAI-compatible adapter + provider quirks |
| `config/providers.json` | Global routing config + provider file refs |
| `config/providers/*.json` | Per-provider model definitions |

## Rules
### Must
- Python 3.11+ only; minimal deps (justify new ones in PRs)
- Explicit typing; small focused functions; no hidden magic
- Surface behavior via env vars and logs
- Provider-specific logic → adapters or dedicated helpers
- Logs: JSONL, parse-friendly, include event names, never log secrets
- Tests: add/update for changed behavior; cache changes need +ve and -ve cases
- Update README for new env vars, commands, or API behavior changes
- Preserve backward compat unless README has a migration note

### Must Not
- Cache tool-call payloads as final assistant text
- Log API keys, auth headers, or raw credentials
- Add provider logic outside adapters/helpers

## Known Quirks (adapter-level)
- `require_reasoning_content_for_tool_calls` — inject non-empty `reasoning_content` on assistant messages with `tool_calls` (empty string → 502 on some providers)
- `normalize_multimodal_content` — ensure dict parts in `messages[*].content` have explicit `type`; wrap single-dict content into a list

## Config
- `ROUTER_ENABLED_PROVIDERS` — comma-list of active providers
- `ROUTER_CACHE_*` / `ROUTER_REDIS_*` — cache tuning (disabled by default)
- `ROUTER_LOG_*` — JSONL logging controls
- `ROUTER_HOST` / `ROUTER_PORT` / `ROUTER_REQUEST_TIMEOUT` — server settings
- Default endpoint: `http://127.0.0.1:11435`

## Testing
```bash
pip install -e ".[dev]" && pytest
```
Async tests auto-detected (`asyncio_mode = auto`).
