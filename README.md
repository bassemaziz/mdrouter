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

A pluggable MCP server that lets AI coding agents search and crawl documentation live. Inspired by Context7, powered by the existing router for LLM summarization.

### Tools for AI Agents

| Tool | Purpose |
|------|---------|
| `doc_search(query, source?, limit)` | Full-text search crawled docs with ranked snippets |
| `doc_crawl(url, name)` | Crawl a documentation site and index it |
| `doc_sources()` | List all indexed sources with page counts and status |
| `doc_refresh(name?)` | Re-crawl one or all sources for latest content |
| `doc_page(url)` | Retrieve full page content + summaries by URL |

### Quick Example

```bash
# Crawl a site via CLI
python -m mdrouter.mcp --crawl nextjs

# Search from CLI
python -m mdrouter.mcp --search "server components"

# List all sources
python -m mdrouter.mcp --sources
```

When the MCP server is connected to an AI coding agent, these are available as native tools. The crawler auto-discovers pages via `llms.txt` (used by Next.js, Astro, Mintlify) or `sitemap.xml`, extracts content, and indexes it in SQLite with FTS5 full-text search.

### Architecture

```
AI Coding Agent
    │ STDIO or HTTP (MCP protocol)
    ▼
mdrouter/mcp/server.py         FastMCP — tools, resources, lifecycle
mdrouter/mcp/capabilities/     Pluggable capability modules
    docs/crawler.py             Async crawler (llms.txt + sitemap + trafilatura)
    docs/store.py               SQLite + FTS5 (zero-dependency search)
    docs/summarizer.py          LLM summarization via shared ModelRouter
mdrouter/mcp/framework/        Reusable infrastructure
    capability.py               Capability ABC — add new modules in one file
    store.py                    Namespaced SQLiteStore with FTS5 helpers
    scheduler.py                Recurring task runner with jitter + backoff
    config.py                   MCPConfig + SummarizationConfig
```

Adding a new capability (e.g. Git, Jira) requires one module implementing the `Capability` ABC — no server changes.

### Cost-Saving Design

- **Content hash dedup** — never re-process or re-store unchanged pages
- **Cache-first summarization** — skip chunks that already have summaries
- **Daily token budget** — hard cap on LLM summarization spend, resets daily
- **Summarization off toggle** — set `enabled: false` to eliminate all LLM costs
- **Chunk truncation** — cap tokens per LLM call
- **FTS5 zero-cost search** — no embedding API needed
- **llms.txt discovery** — 1 HTTP request replaces hundreds of discovery requests
- **Scheduler jitter** — prevents thundering herd on re-crawls

### Systemd Deployment

```bash
sudo cp systemd/mdrouter@.service /etc/systemd/system/
sudo cp systemd/mdrouter-mcp@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now mdrouter@${USER}.service
sudo systemctl enable --now mdrouter-mcp@${USER}.service

# Or use the wrapper
mdrouterctl start           # Router
mdrouterctl mcp start       # MCP server
mdrouterctl mcp crawl nextjs  # Trigger re-crawl
mdrouterctl mcp search "async" # Search docs
```

### Configuration (config/mcp.json)

```json
{
  "enabled_capabilities": ["docs"],
  "transport": "stdio",
  "host": "127.0.0.1",
  "port": 11436,
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
