"""CLI entry point for mdrouter-mcp.

Handles server startup (stdio or streamable-http) and operational
subcommands (--crawl, --search, --sources).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("mdrouter.mcp")


def main() -> None:
    """Entry point: mdrouter-mcp CLI command.

    Server mode (default):
      python -m mdrouter.mcp [--transport stdio|streamable-http] [--host HOST] [--port PORT]

    Operational commands (talk to the store directly, no server needed):
      python -m mdrouter.mcp --init-schema [--reset]
      python -m mdrouter.mcp --crawl <source-name>
      python -m mdrouter.mcp --search <query> [--source <name>]
      python -m mdrouter.mcp --sources

    Env vars: ROUTER_MCP_CONFIG, ROUTER_CONFIG, ROUTER_MCP_TRANSPORT,
              ROUTER_MCP_HOST, ROUTER_MCP_PORT
    """
    import argparse

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
    parser.add_argument(
        "--port", type=int, default=None, help="Port for HTTP transport."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config/mcp.json (default: auto-detect from binary or CWD).",
    )
    parser.add_argument(
        "--router-config",
        default=None,
        metavar="PATH",
        help="Path to config/providers.json (default: auto-detect from binary or CWD).",
    )
    parser.add_argument(
        "--init-schema",
        action="store_true",
        help=(
            "Create the docs database file and initialize its schema. "
            "Safe to run on an existing DB (idempotent). Use --reset to wipe data first."
        ),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="With --init-schema: delete the existing DB file before recreating it.",
    )
    # Operational subcommands (no server)
    parser.add_argument(
        "--crawl",
        default=None,
        metavar="NAME",
        help="Trigger a re-crawl of a doc source.",
    )
    parser.add_argument(
        "--search", default=None, metavar="QUERY", help="Search crawled docs."
    )
    parser.add_argument("--source", default=None, help="Scope --search to a source.")
    parser.add_argument("--sources", action="store_true", help="List doc sources.")
    args = parser.parse_args()

    from mdrouter.mcp._detect import detect_project_root

    mcp_config_path = os.getenv("ROUTER_MCP_CONFIG", "config/mcp.json")
    router_config_path = os.getenv("ROUTER_CONFIG", "config/providers.json")

    # Auto-detect project root from binary path (pip install -e . places
    # the binary in .venv/bin/ — project root is two levels up).
    _project_root = detect_project_root()

    def _resolve_config(path: str, arg: str | None) -> str:
        """Resolve config path: CLI arg > env var > auto-detect > CWD."""
        if arg:
            return arg
        p = Path(path)
        if p.is_absolute():
            return path
        if Path(path).exists():
            return path
        if _project_root:
            candidate = _project_root / path
            if candidate.exists():
                return str(candidate)
        return path

    mcp_config_path = _resolve_config(mcp_config_path, args.config)
    router_config_path = _resolve_config(router_config_path, args.router_config)

    # ── operational commands (no server) ───────────────────────
    if args.init_schema or args.crawl or args.search or args.sources:
        asyncio.run(_run_command(args, mcp_config_path))
        return

    # ── server mode ────────────────────────────────────────────
    from mdrouter.mcp._server import create_server

    async def _run() -> None:
        # Resolve transport: CLI > env var > config > default (stdio)
        transport = args.transport or os.getenv("ROUTER_MCP_TRANSPORT") or "stdio"
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
    from pathlib import Path

    from mdrouter.mcp.framework import MCPConfig, SQLiteStore
    from mdrouter.mcp.capabilities.docs.crawler import DocCrawler
    from mdrouter.mcp.capabilities.docs.store import DocStore

    # Load config
    from mdrouter.mcp.framework.config import _resolve_db_path

    mcp_config_path_obj = Path(mcp_config_path).expanduser().resolve()
    with open(mcp_config_path_obj, "r", encoding="utf-8") as f:
        mcp_config = MCPConfig.model_validate(json.load(f))
    mcp_config.db_path = _resolve_db_path(
        mcp_config.db_path, config_dir=mcp_config_path_obj.parent
    )

    # Init store
    sql_store = SQLiteStore(namespace="docs", db_path=mcp_config.db_path)
    await sql_store.init()
    doc_store = DocStore(sql_store)
    await doc_store.init()  # Run migrations

    try:
        if args.init_schema:
            db_file = Path(mcp_config.db_path)
            existed = db_file.exists()
            if args.reset:
                await sql_store.close()
                db_file.unlink(missing_ok=True)
                logger.info("Removed existing DB: %s", db_file)
                sql_store = SQLiteStore(namespace="docs", db_path=mcp_config.db_path)
                await sql_store.init()
                doc_store = DocStore(sql_store)
                await doc_store.init()
            # Verify by counting tables
            tables = await sql_store.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name",
                (f"{sql_store.namespace}%",),
            )
            sources = await doc_store.list_sources()
            print(
                json.dumps(
                    {
                        "db_path": str(db_file),
                        "db_existed": existed,
                        "reset": args.reset,
                        "tables_created": [t["name"] for t in tables],
                        "sources_count": len(sources),
                    },
                    indent=2,
                )
            )
            return

        if args.sources:
            sources = await doc_store.list_sources()
            print(
                json.dumps(
                    {
                        "sources": [
                            {
                                "name": s["name"],
                                "base_url": s["base_url"],
                                "page_count": s.get("page_count", 0),
                                "last_crawl": s.get("last_crawl", "never"),
                                "status": s.get("status", "unknown"),
                            }
                            for s in sources
                        ]
                    },
                    indent=2,
                )
            )

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
            print(
                json.dumps(
                    {
                        "source": args.crawl,
                        "pages_found": result.pages_found,
                        "pages_new": result.pages_new,
                        "pages_updated": result.pages_updated,
                        "pages_skipped": result.pages_skipped,
                        "duration_seconds": round(result.duration_seconds, 1),
                        "errors": result.errors[:5],
                    },
                    indent=2,
                )
            )

        elif args.search:
            results = await doc_store.search(
                query=args.search,
                source_name=args.source,
                limit=10,
            )
            print(
                json.dumps(
                    {
                        "results": [
                            {
                                "title": r.get("title", ""),
                                "url": r.get("url", ""),
                                "source": r.get("source_name", ""),
                                "snippet": r.get("snippet", ""),
                                "relevance": r.get("_fts_rank", 0),
                            }
                            for r in results
                        ],
                        "total": len(results),
                    },
                    indent=2,
                )
            )

    finally:
        await sql_store.close()
