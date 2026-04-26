# mdrouter

mdrouter is an OpenAI/Ollama-compatible multi-provider LLM router focused on lower cost, lower latency, and operational clarity.

## Project docs

- Roadmap: [ROADMAP.md](ROADMAP.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)

## Core capabilities

- Ollama-compatible endpoints:
  - /api/tags
  - /api/chat
  - /api/generate
  - /api/version
- OpenAI-compatible endpoint for editors and tools:
  - /v1/chat/completions
- Multi-provider routing using provider-prefixed aliases.
- Exact + semantic response cache with memory or Redis backend.
- JSONL logging with request lifecycle events.
- Operations command for traffic, cache, and cost summary.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
python3 -m mdrouter --config config/providers.json
```

Default server:
- http://127.0.0.1:11435

## Configure models and providers

Use a central file at config/providers.json to select provider config files and enabled providers.

Current pattern:
- config/providers.json: server, routing, provider_files, enabled_providers
- config/providers/novita.json: novita provider + novita models
- config/providers/go.json: go provider + go models

To enable or disable providers, edit enabled_providers in config/providers.json.

Set API keys in .env:

- NOVITA_API_KEY
- OPENCODE_GO_API_KEY

Model aliases must use provider/model format, for example:
- novita/deepseek-r1
- go/glm-5.1

Each model can optionally define context_length (tokens), for example:

```json
"go/glm-5.1": {
  "provider": "go",
  "upstream_model": "glm-5.1",
  "capabilities": ["chat", "stream", "tools", "vision"],
  "context_length": 32768,
  "extra": {}
}
```

## Runtime env vars (important)

### Logging

```bash
ROUTER_LOG_ENABLED=true
ROUTER_LOG_FILE=logs/router_requests.jsonl
ROUTER_LOG_REQUEST_BODY=true
ROUTER_LOG_RESPONSE_BODY=true
```

### Router cache

```bash
ROUTER_CACHE_ENABLED=true
ROUTER_CACHE_BACKEND=redis
ROUTER_CACHE_TTL_SEC=300
ROUTER_CACHE_MAX_ENTRIES=1000
ROUTER_REDIS_URL=redis://127.0.0.1:6379/0
ROUTER_REDIS_PREFIX=mdrouter_cache
```

### Semantic cache

```bash
ROUTER_SEM_CACHE_ENABLED=true
ROUTER_SEM_CACHE_THRESHOLD=0.93
ROUTER_SEM_CACHE_MAX_TURNS=3
ROUTER_SEM_CACHE_INCLUDE_ASSISTANT=false
ROUTER_SEM_CACHE_SINGLE_TURN_ONLY=false
```

### Provider prompt caching hints

```bash
ROUTER_PROMPT_CACHE_KEY_ENABLED=true
ROUTER_PROMPT_CACHE_RETENTION=
ROUTER_ALIBABA_EXPLICIT_CACHE=false
```

## Operations: traffic + cache + cost status

Install in editable mode, then use:

```bash
mdrouterctl status --hours 24 --log-file logs/router_requests.jsonl
```

With pricing config for estimated cost:

```bash
mdrouterctl status --hours 24 --pricing config/pricing.example.json
```

This reports:
- total requests
- prompt/completion/cached tokens
- cache hit breakdown (exact, semantic, miss)
- per-model token and estimated cost view

## Systemd deployment

A starter unit is provided at systemd/mdrouter.service.

Example install flow:

```bash
sudo cp systemd/mdrouter.service /etc/systemd/system/
sudo chmod +x systemd/mdrouterctl
sudo systemctl daemon-reload
sudo systemctl enable mdrouter@${USER}.service
sudo systemctl start mdrouter@${USER}.service
```

Operate with helper script:

```bash
./systemd/mdrouterctl status
./systemd/mdrouterctl logs
./systemd/mdrouterctl cost 24
```

Note:
- Adjust paths in the service file for your installation location.
- Ensure /opt/mdrouter/logs is writable by the runtime user.

## Cost-saving and smart-router strategy

The project roadmap is based on practical guidance from provider and cache docs:
- OpenAI prompt caching: stable prefix + consistent prompt cache keys.
- Anthropic prompt caching: explicit breakpoints, invalidation awareness, usage tracking.
- Redis semantic/embedding cache patterns: TTL, batch operations, async cache access.

See [ROADMAP.md](ROADMAP.md) for planned capabilities such as policy routing, context compaction, and tool gateway support.

## Development

```bash
pytest
```

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE).

## Compatibility note

The canonical Python package is now mdrouter.
Legacy import and command aliases remain available for backward compatibility.
