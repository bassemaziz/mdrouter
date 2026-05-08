# mdrouter

mdrouter is an OpenAI/Ollama/Anthropic-compatible multi-provider LLM router focused on predictable operations, lower cost, and low-latency routing.

## Documentation

- Product roadmap: [ROADMAP.md](ROADMAP.md)
- Contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## Key Features

- Ollama-compatible APIs: `/api/tags`, `/api/chat`, `/api/generate`, `/api/version`
- OpenAI-compatible API: `/v1/chat/completions`
- Anthropic-compatible API: `/v1/messages` (for Claude Code, etc.)
- Provider-agnostic alias routing (`provider/model`)
- Model metadata and visible IDs use provider-prefixed aliases to avoid cross-provider name collisions
- Smart virtual alias: `mdrouter/auto`
- JSONL request logging with operational metrics
- Operator CLI for request, cache, and token/cost visibility

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python3 -m mdrouter --config config/providers.json
```

Default endpoint: `http://127.0.0.1:11435`

## Configuration Model

Provider configuration is split for clarity:

- `config/providers.json`: global routing + provider file references
- `config/providers/novita.json`: Novita provider + models
- `config/providers/go.json`: Go provider + models
- `config/providers/anthropic.json.example`: Anthropic provider + models (copy to `providers/anthropic.json` to enable)

Enable runtime providers with:

```bash
ROUTER_ENABLED_PROVIDERS=novita,go
```

Required API keys:

- `NOVITA_API_KEY`
- `OPENCODE_GO_API_KEY`
- `ANTHROPIC_API_KEY` (when using Anthropic provider)

## Runtime Environment Variables

### Server

```bash
ROUTER_HOST=127.0.0.1
ROUTER_PORT=11435
ROUTER_REQUEST_TIMEOUT=90
ROUTER_BIND_LOCALHOST_ONLY=true
```

### Logging

```bash
ROUTER_LOG_ENABLED=true
ROUTER_LOG_FILE=logs/router_requests.jsonl
ROUTER_LOG_REQUEST_BODY=false
ROUTER_LOG_RESPONSE_BODY=false
```

### Router Cache (Temporarily Disabled by Default)

```bash
ROUTER_CACHE_FORCE_OFF=true
ROUTER_CACHE_ENABLED=false
ROUTER_CACHE_BACKEND=redis
ROUTER_CACHE_PROFILE=production
ROUTER_CACHE_TTL_SEC=3600
ROUTER_CACHE_MAX_ENTRIES=5000
ROUTER_REDIS_URL=redis://127.0.0.1:6385/0
ROUTER_REDIS_HOST=127.0.0.1
ROUTER_REDIS_PORT=6385
ROUTER_REDIS_DB=0
ROUTER_REDIS_PREFIX=mdrouter_cache
```

Notes:
- `ROUTER_CACHE_FORCE_OFF=true` forces router response cache off (exact + semantic).
- Set `ROUTER_CACHE_FORCE_OFF=false` only when you explicitly want to re-enable cache behavior.
- `ROUTER_REDIS_URL` takes precedence over host/port/db fields.

### Semantic Cache Controls

```bash
ROUTER_SEM_CACHE_ENABLED=false
ROUTER_SEM_CACHE_THRESHOLD=0.93
ROUTER_SEM_CACHE_LATEST_USER_THRESHOLD=0.30
ROUTER_SEM_CACHE_MAX_TURNS=3
ROUTER_SEM_CACHE_INCLUDE_ASSISTANT=false
ROUTER_SEM_CACHE_SINGLE_TURN_ONLY=false
```

### Provider Prompt Cache Hints

```bash
ROUTER_PROMPT_CACHE_KEY_ENABLED=true
ROUTER_PROMPT_CACHE_RETENTION=
ROUTER_ALIBABA_EXPLICIT_CACHE=false
```

## Smart Routing with mdrouter/auto

`mdrouter/auto` is a virtual alias that classifies requests and selects a concrete configured alias.

Request classes:

- `default_coding`
- `heavy_refactor`
- `long_context`
- `tool_heavy`

Common controls:

```bash
ROUTER_AUTO_POLICY=cost_first
ROUTER_AUTO_FREE_STRATEGY=round_robin
ROUTER_AUTO_FREE_CANDIDATES=go/gpt-5-nano-free,go/big-pickle-free
ROUTER_AUTO_DEFAULT_CANDIDATES=go/qwen3.5-plus
ROUTER_AUTO_HEAVY_REFACTOR_CANDIDATES=go/qwen3.6-plus,go/deepseek-v4-pro
ROUTER_AUTO_LONG_CONTEXT_CANDIDATES=go/mimo-v2.5-pro,go/deepseek-v4-pro
ROUTER_AUTO_TOOL_HEAVY_CANDIDATES=go/glm-5.1,go/kimi-k2.6
ROUTER_AUTO_CONTEXT_LENGTH=1048576
```

## Operations

Use either `mdrouterctl` or `mdrouter` command aliases:

```bash
mdrouterctl status --hours 24 --log-file logs/router_requests.jsonl
mdrouterctl stats --hours 24 --log-file logs/router_requests.jsonl
mdrouterctl cachestatus --hours 24 --log-file logs/router_requests.jsonl
```

With optional pricing file:

```bash
mdrouterctl status --hours 24 --pricing config/pricing.example.json
```

## Systemd Deployment

A starter unit is included at `systemd/mdrouter@.service`.

The unit includes a self-healing `ExecStartPre` step that terminates stale `python -m mdrouter`
listeners on port `11435` before startup, preventing restart loops caused by `address already in use`.

```bash
sudo cp systemd/mdrouter@.service /etc/systemd/system/
sudo chmod +x systemd/mdrouterctl
sudo systemctl daemon-reload
sudo systemctl enable mdrouter@${USER}.service
sudo systemctl start mdrouter@${USER}.service

# If the unit file already exists, re-copy it before reload/restart:
sudo cp systemd/mdrouter@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl restart mdrouter@${USER}.service
```

## Anthropic-compatible API (`POST /v1/messages`)

mdrouter exposes an Anthropic-compatible Messages endpoint for tools that speak the Anthropic protocol, including Claude Code.

```bash
curl -s http://127.0.0.1:11435/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic/claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 256
  }' | jq .
```

Supported for both streaming (`stream: true`) and non-streaming requests.

## Claude Code with mdrouter

Use this when you want Claude Code to talk to mdrouter while mdrouter routes to your configured providers.

### 1) Start mdrouter

```bash
python3 -m mdrouter --config config/providers.json
```

### 2) Configure Claude Code environment

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:11435"
export ANTHROPIC_AUTH_TOKEN="mdrouter-local-token"
unset ANTHROPIC_API_KEY
```

Notes:
- Use exactly one auth mechanism in your shell session for Claude Code.
- Prefer `ANTHROPIC_AUTH_TOKEN` for local mdrouter.
- If `ANTHROPIC_API_KEY` is set at the same time, some clients can pick the wrong credential path.

### 3) Run Claude Code against a routed alias

```bash
claude --model anthropic/claude-sonnet-4-6
```

Best practices:
- Use provider-prefixed aliases (`provider/model`) to avoid ambiguity.
- Keep `strict_provider_prefix` enabled in routing config for deterministic model resolution.
- Verify available routed models with:

```bash
curl -s http://127.0.0.1:11435/v1/models | jq .
```

### 4) Troubleshooting

```bash
# Service status
mdrouterctl status --hours 1 --log-file logs/router_requests.jsonl

# Recent request events
tail -n 50 logs/router_requests.jsonl
```

Common issues:
- 401/403 from client: check token env vars and clear conflicting auth vars.
- Unknown model: use a configured alias from `/v1/models`.
- Tool-call errors: confirm the selected model has `tools` capability in provider config.

## Development

```bash
pytest
make precommit
```

## Compatibility

The canonical package name is `mdrouter`. Legacy aliases remain for backward compatibility.

## License

MIT. See [LICENSE](LICENSE).
