"""mdrouter MCP server — pluggable capabilities for AI coding agents.

Entry points:
- mdrouter-mcp CLI command (stdio transport)
- create_server() for programmatic use
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("mdrouter.mcp")

from mdrouter.mcp.capabilities import CAPABILITY_MAP as _CAPABILITY_MODULES


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

    # Load router config (for LLM summarization)
    router_config_path = Path(router_config_path).expanduser().resolve()
    if router_config_path.exists():
        app_config = AppConfig.from_file(router_config_path)
    else:
        logger.warning("Router config not found at %s, LLM features disabled", router_config_path)
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
        logger.info("Starting mdrouter MCP server with capabilities: %s",
                     mcp_config.enabled_capabilities)

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
    mcp = FastMCP("mdrouter", json_response=True, lifespan=_lifespan, host=host, port=port)

    # Attach config so callers can read transport settings
    mcp._mcp_config = mcp_config
    return mcp


def main() -> None:
    """Entry point: mdrouter-mcp CLI command.

    Server mode (default):
      python -m mdrouter.mcp [--transport stdio|streamable-http] [--host HOST] [--port PORT]

    Operational commands (talk to the store directly, no server needed):
      python -m mdrouter.mcp --crawl <source-name>
      python -m mdrouter.mcp --search <query> [--source <name>]
      python -m mdrouter.mcp --sources

    Env vars: ROUTER_MCP_CONFIG, ROUTER_CONFIG, ROUTER_MCP_TRANSPORT,
              ROUTER_MCP_HOST, ROUTER_MCP_PORT
    """
    import argparse
    import os
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="mdrouter MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport mode (default: from config, or stdio).",
    )
    parser.add_argument("--host", default=None, help="Host for HTTP transport.")
    parser.add_argument("--port", type=int, default=None, help="Port for HTTP transport.")
    # Operational subcommands (no server)
    parser.add_argument("--crawl", default=None, metavar="NAME", help="Trigger a re-crawl of a doc source.")
    parser.add_argument("--search", default=None, metavar="QUERY", help="Search crawled docs.")
    parser.add_argument("--source", default=None, help="Scope --search to a source.")
    parser.add_argument("--sources", action="store_true", help="List doc sources.")
    args = parser.parse_args()

    mcp_config_path = os.getenv("ROUTER_MCP_CONFIG", "config/mcp.json")
    router_config_path = os.getenv("ROUTER_CONFIG", "config/providers.json")

    # ── operational commands (no server) ───────────────────────
    if args.crawl or args.search or args.sources:
        asyncio.run(_run_command(args, mcp_config_path))
        return

    # ── server mode ────────────────────────────────────────────
    async def _run() -> None:
        # Resolve transport: CLI > env var > config > default (stdio)
        transport = (
            args.transport
            or os.getenv("ROUTER_MCP_TRANSPORT")
            or "stdio"
        )
        host = args.host or os.getenv("ROUTER_MCP_HOST") or "127.0.0.1"
        port = args.port or int(os.getenv("ROUTER_MCP_PORT", "0")) or 8000

        server = await create_server(
            mcp_config_path=mcp_config_path,
            router_config_path=router_config_path,
            host=host,
            port=port,
        )

        if transport == "streamable-http":
            logger.info("Starting MCP server on %s:%d (streamable-http)", host, port)
            await server.run_streamable_http_async()
        else:
            logger.info("Starting MCP server on stdio")
            await server.run_stdio_async()

    asyncio.run(_run())


async def _run_command(args: Any, mcp_config_path: str) -> None:
    """Execute an operational command against the store directly."""
    import json
    import os
    import sys
    from pathlib import Path

    from mdrouter.mcp.framework import MCPConfig, SQLiteStore
    from mdrouter.mcp.capabilities.docs.crawler import DocCrawler
    from mdrouter.mcp.capabilities.docs.store import DocStore

    # Load config
    mcp_config_path_obj = Path(mcp_config_path).expanduser().resolve()
    with open(mcp_config_path_obj, "r", encoding="utf-8") as f:
        mcp_config = MCPConfig.model_validate(json.load(f))

    # Init store
    sql_store = SQLiteStore(namespace="docs", db_path=mcp_config.db_path)
    await sql_store.init()
    doc_store = DocStore(sql_store)
    await doc_store.init()  # Run migrations

    try:
        if args.sources:
            sources = await doc_store.list_sources()
            print(json.dumps({"sources": [
                {"name": s["name"], "base_url": s["base_url"],
                 "page_count": s.get("page_count", 0),
                 "last_crawl": s.get("last_crawl", "never"),
                 "status": s.get("status", "unknown")}
                for s in sources
            ]}, indent=2))

        elif args.crawl:
            cfg = mcp_config.capabilities.get("docs")
            cfg_dict = cfg.model_dump() if cfg else {}
            crawler = DocCrawler(
                user_agent=cfg_dict.get("user_agent", "mdrouter-docbot/1.0"),
                max_concurrent=cfg_dict.get("max_concurrent_requests", 5),
                request_delay=cfg_dict.get("request_delay_seconds", 0.5),
                max_pages=cfg_dict.get("max_pages_per_site", 500),
            )
            source = await doc_store.get_source(args.crawl)
            if not source:
                print(f"Error: source '{args.crawl}' not found. Use --sources to list.")
                sys.exit(1)
            result = await crawler.crawl_site(
                base_url=source["base_url"],
                source_name=args.crawl,
                doc_store=doc_store,
            )
            print(json.dumps({
                "source": args.crawl,
                "pages_found": result.pages_found,
                "pages_new": result.pages_new,
                "pages_updated": result.pages_updated,
                "pages_skipped": result.pages_skipped,
                "duration_seconds": round(result.duration_seconds, 1),
                "errors": result.errors[:5],
            }, indent=2))

        elif args.search:
            results = await doc_store.search(
                query=args.search,
                source_name=args.source,
                limit=10,
            )
            print(json.dumps({"results": [
                {"title": r.get("title", ""), "url": r.get("url", ""),
                 "source": r.get("source_name", ""),
                 "snippet": r.get("snippet", ""),
                 "relevance": r.get("_fts_rank", 0)}
                for r in results
            ], "total": len(results)}, indent=2))

    finally:
        await sql_store.close()


if __name__ == "__main__":
    main()
