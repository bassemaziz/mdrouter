"""FastMCP server factory for mdrouter-mcp.

Creates a fully configured FastMCP instance with all enabled capabilities,
shared ModelRouter, Scheduler, and SQLite stores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mdrouter.mcp.capabilities import CAPABILITY_MAP as _CAPABILITY_MODULES
from mdrouter.mcp.framework.config import _resolve_db_path

logger = logging.getLogger("mdrouter.mcp")


def _load_capability_class(module_path: str) -> Any:
    """Import a capability class by dotted path."""
    mod_name, cls_name = module_path.split(":")
    import importlib

    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


async def create_server(
    mcp_config_path: str | Path = "config/mcp.json",
    router_config_path: str | Path = "config/providers.json",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:
    """Build a FastMCP server with all enabled capabilities.

    Args:
        mcp_config_path: Path to config/mcp.json.
        router_config_path: Path to the existing providers config.
            Used to initialize the shared ModelRouter for LLM-based
            capabilities (summarization, etc.).
        host: Host for streamable-http transport (default: 127.0.0.1).
        port: Port for streamable-http transport (default: 8000).

    Returns:
        A configured FastMCP instance ready for .run(transport="...").
    """
    from mcp.server.fastmcp import FastMCP
    from contextlib import asynccontextmanager
    from collections.abc import AsyncIterator

    from mdrouter.mcp.framework import (
        CapabilityContext,
        MCPConfig,
        Scheduler,
        SQLiteStore,
    )
    from mdrouter.config import AppConfig
    from mdrouter.router import ModelRouter
    from mdrouter.runtime import RuntimeSettings

    # Load MCP config
    mcp_config_path = Path(mcp_config_path).expanduser().resolve()
    with open(mcp_config_path, "r", encoding="utf-8") as f:
        mcp_config = MCPConfig.model_validate(json.load(f))
    # Resolve db_path relative to config file directory
    mcp_config.db_path = _resolve_db_path(
        mcp_config.db_path, config_dir=mcp_config_path.parent
    )

    # Load router config (for LLM summarization)
    router_config_path = Path(router_config_path).expanduser().resolve()
    if router_config_path.exists():
        app_config = AppConfig.from_file(router_config_path)
    else:
        logger.warning(
            "Router config not found at %s, LLM features disabled",
            router_config_path,
        )
        app_config = AppConfig(providers={}, models={})

    runtime = RuntimeSettings.from_env()
    router = ModelRouter(config=app_config, runtime=runtime)

    # Shared infrastructure
    scheduler = Scheduler(config=mcp_config.scheduler)

    def _store_factory(namespace: str) -> SQLiteStore:
        return SQLiteStore(namespace=namespace, db_path=mcp_config.db_path)

    # --- Lifespan: initialize all capabilities ---
    @asynccontextmanager
    async def _lifespan(server: FastMCP) -> AsyncIterator[list[Any]]:
        logger.info(
            "Starting mdrouter MCP server with capabilities: %s",
            mcp_config.enabled_capabilities,
        )

        capabilities: list[Any] = []
        for cap_name in mcp_config.enabled_capabilities:
            if cap_name not in _CAPABILITY_MODULES:
                logger.warning("Unknown capability '%s', skipping", cap_name)
                continue

            cap_cls = _load_capability_class(_CAPABILITY_MODULES[cap_name])
            cap_config = (mcp_config.capabilities or {}).get(cap_name)
            cap_config_dict = cap_config.model_dump() if cap_config else {}

            ctx = CapabilityContext(
                router=router,
                store_factory=_store_factory,
                scheduler=scheduler,
                config=mcp_config,
                capability_config=cap_config_dict,
            )
            capability = cap_cls(ctx)

            logger.info("Initializing capability: %s", cap_name)
            await capability.initialize()

            capability.register_tools(server)
            capability.register_resources(server)
            capability.register_prompts(server)

            for task in capability.scheduled_tasks():
                scheduler.register(
                    name=f"{cap_name}/{task.name}",
                    coroutine=task.coroutine,
                    interval_hours=task.interval_hours,
                    run_on_startup=task.run_on_startup,
                )

            capabilities.append(capability)
            logger.info("Capability '%s' ready", cap_name)

        await scheduler.start()

        try:
            yield capabilities
        finally:
            await scheduler.stop()
            for cap in capabilities:
                await cap.shutdown()

    # Create FastMCP server
    mcp = FastMCP(
        "mdrouter", json_response=True, lifespan=_lifespan, host=host, port=port
    )

    # Attach config so callers can read transport settings
    mcp._mcp_config = mcp_config
    return mcp
