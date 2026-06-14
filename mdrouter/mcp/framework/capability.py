"""Capability ABC — the pluggable contract for every MCP capability module.

Design:
- Each capability is a self-contained module under capabilities/<name>/
- A capability registers tools + resources on the FastMCP server
- The framework injects shared dependencies via CapabilityContext
- Adding a new capability requires zero changes to server.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from mdrouter.mcp.framework.config import MCPConfig

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from mdrouter.mcp.framework.scheduler import Scheduler
    from mdrouter.mcp.framework.store import SQLiteStore
    from mdrouter.router import ModelRouter


@dataclass
class ScheduledTask:
    """A recurring task the scheduler should run.

    Attributes:
        name: Unique task identifier within the capability.
        interval_hours: How often to run.
        coroutine: Async callable (no arguments) to execute.
        run_on_startup: Whether to run once immediately at server start.
    """

    name: str
    interval_hours: int
    coroutine: Callable[[], Any]
    run_on_startup: bool = False


@dataclass
class CapabilityContext:
    """Shared dependencies injected into every capability.

    COST NOTE: The router is shared across all capabilities. Cost tracking
    and model selection happen at the router level, so capabilities don't
    need to manage their own LLM budget.
    """

    router: "ModelRouter"
    store_factory: Callable[[str], "SQLiteStore"]
    scheduler: "Scheduler"
    config: MCPConfig
    capability_config: dict[str, Any] = field(default_factory=dict)


class Capability(ABC):
    """A pluggable module that contributes tools + resources to the MCP server.

    Subclasses must implement:
    - name: str identifier matching the config key (e.g. "docs")
    - description: human-readable one-liner
    - register_tools(mcp): add @mcp.tool() decorated functions
    - register_resources(mcp): add @mcp.resource() decorated functions
    - initialize(ctx): set up stores, warm caches

    Optional:
    - scheduled_tasks(): return recurring tasks for the scheduler
    - shutdown(): clean up connections
    """

    name: str
    description: str

    def __init__(self, ctx: CapabilityContext) -> None:
        self.ctx = ctx

    async def initialize(self) -> None:
        """Called at server startup. Set up stores, warm caches, etc."""

    @abstractmethod
    def register_tools(self, mcp: "FastMCP") -> None:
        """Register this capability's tools on the FastMCP instance."""

    @abstractmethod
    def register_resources(self, mcp: "FastMCP") -> None:
        """Register this capability's resources on the FastMCP instance."""

    def scheduled_tasks(self) -> list[ScheduledTask]:
        """Return recurring tasks the scheduler should run."""
        return []

    def register_prompts(self, mcp: "FastMCP") -> None:
        """Register this capability's prompts on the FastMCP instance.

        Prompts are templates that guide AI agents through multi-step
        workflows. Unlike tools, they are not called directly — they
        provide instructions the agent should follow.
        """

    async def shutdown(self) -> None:
        """Called at server shutdown. Clean up connections."""
