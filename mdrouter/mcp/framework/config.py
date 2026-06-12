"""MCP configuration models (pydantic).

Cost-saving baked in:
- summarization.model uses cheapest capable provider by default (deepseek)
- summarization.max_tokens_per_day enforces a soft budget
- summarization.enabled can be toggled off to skip LLM costs entirely
- crawl_interval_hours defaults to 24 to avoid excessive re-crawling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SummarizationConfig(BaseModel):
    """Controls LLM-based summarization of crawled content.

    COST NOTE: Set enabled=false to eliminate ALL LLM costs.
    When enabled, max_tokens_per_day caps daily spend.
    """

    enabled: bool = True
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
