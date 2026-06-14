---
name: docs-first-coding
description: >
  Before writing any code, first consult mdrouter-mcp for external docs AND
  codegraph for codebase context. Both are required — prevents guessing,
  deprecated patterns, hallucinated APIs, and reinventing existing patterns.
---

# Docs-First Coding — Consult Both mdrouter-mcp + codegraph Before Writing Code

## RULE

**No code that touches external APIs/libraries/sdks without first checking both external docs AND codebase context.**

Whenever the task involves writing, modifying, or debugging code that interacts with an external dependency, you **must** use **both** before writing:

| Tool | What it provides | When to use |
|---|---|---|
| **mdrouter-mcp** (`doc_search`/`doc_page`/`doc_crawl`) | Official API docs, function signatures, parameters, best practices | Any external library, framework, SDK, or API |
| **codegraph** (`codegraph_explore`/`codegraph_search`/`codegraph_callers`/`codegraph_impact`) | Local code patterns, existing implementations, call chains, refactor impact | Every code task — always |

For purely internal changes (no external deps), codegraph is still required; mdrouter-mcp may be skipped.

## Workflow

### Phase A: Understand the external API (mdrouter-mcp)

**Step A1:** Check what's already indexed
```
doc_sources
```

**Step A2:** If the library isn't indexed, crawl it
```
doc_crawl url="https://<official-docs-url>" name="<short-name>"
```

**Step A3:** Search for the specific topic
```
doc_search query="<what-you-need-to-do>" source="<source-name>"
doc_page url="<url-from-results>"
```

**Recommended docs to crawl for this project:**

| Library/Dependency | Docs URL | Reason |
|---|---|---|
| LiveKit Agents SDK | `https://docs.livekit.io/agents/` | Voice agent pipeline |
| Deepgram | `https://developers.deepgram.com/docs/` | STT |
| Groq | `https://console.groq.com/docs/` | LLM inference |
| ElevenLabs | `https://elevenlabs.io/docs/` | TTS |
| OpenAI | `https://platform.openai.com/docs/` | LLM + TTS fallback |
| Celery | `https://docs.celeryq.dev/en/stable/` | Background tasks |
| ClickHouse | `https://clickhouse.com/docs/` | Analytics/observability |
| Django | `https://docs.djangoproject.com/en/stable/` | Core backend |
| DRF | `https://www.django-rest-framework.org/` | REST API layer |
| FastAPI | `https://fastapi.tiangolo.com/` | Widget service |
| Pydantic | `https://docs.pydantic.dev/latest/` | Config & validation |

### Phase B: Understand the codebase (codegraph)

**Step B1:** Survey the area — one call gives you everything
```
codegraph_explore "<feature or area>"
```
This is your primary codegraph tool. A single call returns relevant symbols + their source grouped by file.

**Step B2:** Dig deeper only if needed

| If you need to... | Use |
|---|---|
| Find where a symbol is | `codegraph_search "<name>"` |
| See what calls a function | `codegraph_callers "<name>"` |
| See what a function calls | `codegraph_callees "<name>"` |
| See what breaking a symbol affects | `codegraph_impact "<name>"` |
| Get a symbol's full body | `codegraph_node "<name>" includeCode=true` |
| Survey project structure | `codegraph_files` |

**Do NOT grep for symbols codegraph already knows.** Codegraph's results come from a full AST parse and are more accurate.

### Phase C: Implement

Write code using:
- Correct function signatures from official docs (Phase A)
- Established project patterns from codegraph (Phase B)
- Layers & conventions from the codebase (services/selectors, feature-first, etc.)
- Proper error handling as recommended by both

### Phase D: Validate

- Run relevant tests (`pytest`, `pnpm test`, etc.)
- Run linting (`ruff`, `eslint`)
- Check for errors (`get_errors`)
- Verify against docs one more time if anything felt uncertain

## Decision Flow

```
Task received
  │
  ├─ Does it touch an external dependency? ──yes──> Phase A (mdrouter-mcp)
  │                                                    │
  │   Otherwise skip Phase A                           │
  │                                                    │
  └────────────────────────────────────────────────────┘
                              │
                              ▼
                   Phase B (codegraph) ←──── ALWAYS
                              │
                              ▼
                   Phase C (Implement)
                              │
                              ▼
                   Phase D (Validate)
```

## Exceptions

**mdrouter-mcp may be skipped** when:
- The code is purely internal (stdlib, no external API/SDK call)
- Docs already consulted in this same conversation
- The external dependency has no public docs site to crawl

**codegraph may be skipped** only when:
- The workspace has no `.codegraph/` index (brand new project)

## Verification Checklist

Before marking done, verify:
- [ ] Did I check official docs before using any external API/endpoint/parameter?
- [ ] Did I check codegraph for existing patterns before writing new code?
- [ ] Is the code following project conventions (not reinventing)?
- [ ] Did I run validation (test/lint/check)?
