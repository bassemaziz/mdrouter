# mdrouter

Multi-provider LLM router + MCP documentation server for AI coding agents.

## Overview

mdrouter does two things:

1. **LLM Router** — OpenAI/Ollama/Anthropic-compatible HTTP API that routes chat requests across multiple providers (DeepSeek, Novita, Go, Anthropic, Ark, Fireworks, Zen) using provider-prefixed model aliases.
2. **MCP Doc Server** — Pluggable MCP server that gives AI coding agents (Claude Code, Cursor, Codex) live documentation crawling, full-text search, and on-demand summarization.

## Quick Start

```bash
# Setup
make setup-mcp              # venv + dev + MCP dependencies

# Router (HTTP API on :11435)
make run                    # python -m mdrouter --config config/providers.json

# MCP Server (stdio — for AI coding tools)
make run-mcp                # python -m mdrouter.mcp

# MCP Server (HTTP — for systemd)
make run-mcp-http           # python -m mdrouter.mcp --transport streamable-http
```

Default endpoints: router on `127.0.0.1:11435`, MCP on `127.0.0.1:11436`.

## LLM Router

### Supported APIs

| Endpoint | Protocol |
|----------|----------|
| `/v1/chat/completions` | OpenAI-compatible |
| `/v1/messages` | Anthropic-compatible (Claude Code, etc.) |
| `/api/chat`, `/api/generate`, `/api/tags` | Ollama-compatible |

### Model Routing

Models use provider-prefixed aliases: `deepseek/deepseek-v4-flash`, `anthropic/claude-sonnet-4-6`, `go/qwen3.5-plus`.

```bash
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek/deepseek-v4-flash", "messages": [{"role": "user", "content": "Hello"}]}'
```

`mdrouter/auto` is a virtual alias that classifies requests (`default_coding`, `heavy_refactor`, `long_context`, `tool_heavy`) and picks the best configured model automatically.

### Configuration

Split across `config/providers.json` (routing) and `config/providers/*.json` (per-provider model definitions). Runtime controlled via `ROUTER_*` env vars:

```bash
ROUTER_ENABLED_PROVIDERS=deepseek,go
ROUTER_HOST=127.0.0.1
ROUTER_PORT=11435
```

Required API keys: `DEEPSEEK_API_KEY`, `NOVITA_API_KEY`, `OPENCODE_GO_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### Operations CLI

```bash
mdrouterctl status --hours 24          # Traffic, cache, cost summary
mdrouterctl stats --hours 24           # Per-model statistics
mdrouterctl cachestatus --hours 24     # Cache hit-rate analysis
mdrouterctl start|stop|restart|logs    # Service management
```

## MCP Documentation Server

A pluggable MCP server that gives AI coding agents live, versioned documentation
with **verbatim code examples** and **LLM-generated prose summaries**.
Returns Context7-shaped markdown — code extracted from source HTML, prose by the
shared router. No new dependencies beyond the existing stack.

### AI Agent Tools (v3 — Context7-shaped)

| Tool | Purpose | Key Params |
|------|---------|------------|
| `search_docs` | Search code + info snippets, returns markdown | `query`, `source`, `max_tokens=1000`, `snippet_type` |
| `crawl_docs` | Crawl a site, extract code, auto-summarize | `url`, `name`, `version`, `max_pages` |
| `get_doc_page` | Full page summary (code + info) | `url`, `max_tokens=1000` |
| `snippets_docs` | Code-only search (verbatim from HTML) | `query`, `source`, `language` |
| `list_doc_sources` | List all indexed sources + versions | — |
| `refresh_docs` | Re-crawl unversioned sources | `name`, `force` |
| `resolve_library` | Resolve library name → doc URL | `name`, `version` |
| `init_docs` | **Prompt** — auto-discover project deps + crawl | `project_path` |

All content tools return **markdown**. Status tools return JSON. Every content
tool accepts `max_tokens` (default **1000**) — the AI agent controls context
usage. Token budget: 60% code, 40% info.

### Quick Integration

**1. Add to your MCP client config** (Cursor, Claude Code, Codex, etc.):

```json
{
  "mcpServers": {
    "mdrouter": {
      "command": "python",
      "args": ["-m", "mdrouter.mcp"],
      "cwd": "/path/to/mdrouter"
    }
  }
}
```

**2. Initialize docs for your project** (in the AI agent):
```
Call init_docs for this project
```
The agent auto-discovers dependencies from `pyproject.toml`/`package.json`,
resolves doc URLs, and crawls each one.

**3. Or crawl manually:**
```
crawl_docs(url="https://fastapi.tiangolo.com/", name="fastapi")
crawl_docs(url="https://docs.pydantic.dev/latest/", name="pydantic", version="2.10")
```

**4. Search at query time:**
```
search_docs(query="how to define middleware", source="fastapi")
snippets_docs(query="Depends", source="fastapi", language="python")
```

### How It Works

```
                         ┌── crawler ──► pages (SQLite)
                         │      │
crawl_docs(url, name) ───┤      ├── code_extractor ──► code_snippets_json
                         │      │    (html.parser,     (verbatim from HTML)
                         │      │     no new deps)
                         │      │
                         │      └── summarizer ──────► info_snippets_json
                         │           (prose-only LLM,  (Overview, Concepts, Notes)
                         │            no code gen)
                         │
search_docs(query) ──────┤
                         │      FTS5 search_fts ──────► response_builder
                         │      (code + info indexed)   (60/40 token budget)
                         │                              │
                         └──────────────────────────────┘
                                                    markdown
                                                 (Context7-shaped)
```

**Version handling:**
- `crawl_docs(name="fastapi")` → unversioned, re-crawled periodically for changes
- `crawl_docs(name="fastapi", version="0.115.0")` → **immutable**, never re-crawled
- `refresh_docs(force=True)` → override to re-crawl a versioned source

### Cost-Saving Design

- **Code from HTML, not LLM** — code blocks extracted with stdlib `html.parser` (free, deterministic, verbatim)
- **Prose-only summarization** — LLM never reproduces code (code is always exact)
- **Content hash dedup** — skip unchanged pages entirely
- **Cache-first info snippets** — reuse LLM summaries when content hasn't changed
- **Daily token budget** — hard cap on LLM spend (`max_tokens_per_day`)
- **FTS5 zero-cost search** — no embedding API, instant keyword matching
- **llms.txt discovery** — 1 HTTP request replaces hundreds
- **Scheduler jitter** — prevents thundering herd on re-crawls
- **Versioned = skip** — immutable sources never re-crawled by scheduler

### Environment Variables

All env vars are optional — reasonable defaults are set in `config/mcp.json`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_MCP_DB_PATH` | `~/.local/share/mdrouter/mcp.db` | Centralized shared DB (XDG) |
| `ROUTER_MCP_CACHE_TTL` | `60` | Search cache TTL (seconds, 0=disabled) |
| `ROUTER_MCP_MAX_RESPONSE_TOKENS` | `1000` | Default token cap for responses |
| `ROUTER_MCP_CONFIG` | `config/mcp.json` | MCP config path |
| `CONTEXT7_API_KEY` | — | Optional: Context7 API fallback for resolve_library |

### Systemd Deployment (zero env vars)

```bash
# Copy service files
sudo cp systemd/mdrouter-mcp.service /etc/systemd/system/
sudo cp systemd/mdrouter@.service /etc/systemd/system/

# Or for user session (no sudo):
mkdir -p ~/.config/systemd/user/
cp systemd/user/mdrouter-mcp.service ~/.config/systemd/user/

# Start
sudo systemctl daemon-reload
sudo systemctl enable --now mdrouter-mcp@${USER}

# Check status
systemctl status mdrouter-mcp@${USER}

# Manage from CLI
python -m mdrouter.mcp --crawl fastapi
python -m mdrouter.mcp --search "routing"
python -m mdrouter.mcp --sources
```

No `Environment=` or `EnvironmentFile=` needed — the server reads `config/mcp.json`
and resolves paths via XDG natively. DB location: `~/.local/share/mdrouter/mcp.db`
(shared across all projects).

### Performance Tuning

| Setting | Default | Tune |
|---------|---------|------|
| `max_concurrent_requests` | 5 | Increase for large sites (up to 10) |
| `request_delay_seconds` | 0.5 | Lower for permissive sites (0.1) |
| `max_pages_per_site` | 500 | Cap for very large docs (1000+) |
| `cache_ttl_seconds` | 60 | Increase for stable docs (300+) |
| `max_response_tokens` | 1000 | Agent controls this per-call |
| `LimitNOFILE` (systemd) | 8192 | Socket limit for concurrent crawls |
| `MemoryMax` (systemd) | 512M | SQLite works in surprisingly little RAM |

### Architecture

```
mdrouter/mcp/capabilities/docs/
    __init__.py          DocsCapability — lifecycle, tool registration (375 lines)
    _tools.py            Tool implementations — pure async functions (316 lines)
    crawler.py           Async crawler (llms.txt + parallel sitemap + trafilatura)
    code_extractor.py    HTML code block extractor (stdlib html.parser)
    store.py             SQLite + FTS5 (sources, pages, search_fts, migrations)
    summarizer.py        Prose-only LLM summarization via shared ModelRouter
    response_builder.py  Context7-shaped markdown with 60/40 token budget
    resolver.py          Library name → doc URL (45+ known mappings)
mdrouter/mcp/framework/
    capability.py        Capability ABC + register_prompts support
    store.py             Namespaced SQLiteStore with FTS5 helpers
    scheduler.py         Recurring task runner with jitter + backoff
    config.py            MCPConfig + SummarizationConfig (XDG-aware)
```

Adding a new capability (e.g. Git, Jira) requires one module implementing the
`Capability` ABC — zero server changes.

### Configuration (config/mcp.json)

```json
{
  "enabled_capabilities": ["docs"],
  "db_path": "~/.local/share/mdrouter/mcp.db",
  "transport": "stdio",
  "cache_ttl_seconds": 60,
  "max_response_tokens": 1000,
  "summarization": {
    "enabled": true,
    "model": "deepseek/deepseek-v4-flash",
    "max_tokens_per_day": 200000
  },
  "capabilities": {
    "docs": {
      "crawl_interval_hours": 24,
      "max_pages_per_site": 500
    }
  }
}
```

## Development

```bash
make test           # Full suite (128 tests)
make precommit      # Lint + format + audit + test
```

## License

MIT. See [LICENSE](LICENSE).
