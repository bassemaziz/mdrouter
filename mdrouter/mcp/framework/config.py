"""MCP configuration models (pydantic).

Cost-saving baked in:
- summarization.model uses cheapest capable provider by default (deepseek)
- summarization.max_tokens_per_day enforces a soft budget
- summarization.enabled can be toggled off to skip LLM costs entirely
- crawl_interval_hours defaults to 24 to avoid excessive re-crawling
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _resolve_db_path(raw: str, config_dir: str | Path | None = None) -> str:
    """Resolve db_path to an absolute path.

    - Env var ROUTER_MCP_DB_PATH overrides everything.
    - ~-paths: expanded to home directory.
    - Relative paths: resolved against config_dir (if provided), else CWD.

    For centralized storage across projects, set db_path to an
    absolute path like ~/.local/share/mdrouter/mcp.db.
    """
    env_path = os.getenv("ROUTER_MCP_DB_PATH")
    if env_path:
        return str(Path(env_path).expanduser().resolve())
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p.resolve())
    # Relative: resolve against config_dir or CWD
    base = Path(config_dir) if config_dir else Path()
    return str((base / p).resolve())


class SummarizationConfig(BaseModel):
    """Controls LLM-based summarization of crawled content.

    COST MODEL:
    - enabled=false (DEFAULT): Crawl stores raw content + extracted code
      blocks. Search/snippets work on raw text. Zero LLM cost.
    - enabled=true: After each crawl, an LLM call is made per page chunk
      to generate clean prose summaries. Use max_tokens_per_day to cap
      spend. Use a summarize_source tool call to upgrade selectively.

    The default is off because crawling 100+ pages across many libraries
    would otherwise run up large LLM bills. With it off, users can still
    get useful results (extracted code + first-N-paragraph excerpts).
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Auto-summarize after every crawl. Default false (zero LLM cost). "
            "Use the summarize_source tool to upgrade a source on demand."
        ),
    )
    model: str = "deepseek/deepseek-v4-flash"
    max_concurrent: int = Field(default=3, ge=1, le=10)
    max_tokens_per_day: int = Field(
        default=200_000,
        description="Soft daily token budget for summarization. 0 = unlimited.",
    )
    prompt: str = Field(
        default=(
            "Summarize this documentation page into 3-5 key points. "
            "Focus on APIs, parameters, return types, and usage examples. "
            "Be concise — each point one sentence."
        ),
        description="System prompt for the summarization LLM call.",
    )
    max_chunk_tokens: int = Field(
        default=4000,
        description="Max tokens per chunk sent to the LLM. Chunks exceeding "
        "this are truncated before summarization to control costs.",
    )


class SchedulerConfig(BaseModel):
    """Scheduler for recurring background tasks (re-crawls, etc.)."""

    enabled: bool = True
    default_interval_hours: int = Field(default=24, ge=1, le=168)
    jitter_seconds: int = Field(
        default=300,
        description="Random jitter added to intervals to avoid thundering herd.",
    )


class CapabilitySettings(BaseModel):
    """Per-capability configuration section. Capabilities receive their own TypedDict.

    The framework passes the raw dict to the capability's initialize() method.
    Common keys like 'crawl_interval_hours' can be read by any capability.
    """

    model_config = {"extra": "allow"}


class MCPConfig(BaseModel):
    """Root MCP configuration loaded from config/mcp.json."""

    enabled_capabilities: list[str] = Field(default_factory=lambda: ["docs"])
    db_path: str = "data/mcp.db"
    transport: str = Field(
        default="stdio",
        description="Transport mode: 'stdio' for AI coding tools, "
        "'streamable-http' for persistent systemd service.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Host to bind when transport='streamable-http'.",
    )
    port: int = Field(
        default=11436,
        description="Port to bind when transport='streamable-http'.",
    )
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    capabilities: dict[str, CapabilitySettings] = Field(default_factory=dict)
    cache_ttl_seconds: int = Field(
        default=60,
        description="TTL for search result cache in seconds. 0 = disabled.",
    )
    max_response_tokens: int = Field(
        default=1000,
        description="Default max tokens for content-returning tool responses.",
    )
