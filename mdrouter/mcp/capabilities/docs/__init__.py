"""DocsCapability — documentation search and crawling for AI coding agents.

Registers tools:
- doc_search: full-text search across crawled docs
- doc_crawl: crawl a new documentation site
- doc_sources: list all indexed sources
- doc_refresh: force re-crawl one or all sources
- doc_page: get full page content by URL

Registers resources:
- doc://sources/{name}/tree: browse page tree for a source
- doc://sources/{name}/pages/{page_id}: full page content
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from mdrouter.mcp.framework.capability import Capability, ScheduledTask
from mdrouter.mcp.framework.capability import CapabilityContext
from mdrouter.mcp.capabilities.docs.store import DocStore
from mdrouter.mcp.capabilities.docs.crawler import DocCrawler
from mdrouter.mcp.capabilities.docs.summarizer import DocSummarizer

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context, FastMCP

logger = logging.getLogger("mdrouter.mcp.docs")


class DocsCapability(Capability):
    """Documentation search and crawling capability."""

    name = "docs"
    description = "Search and crawl documentation sites for AI coding agents"

    def __init__(self, ctx: CapabilityContext) -> None:
        super().__init__(ctx)
        self._store: DocStore | None = None
        self._crawler: DocCrawler | None = None
        self._summarizer: DocSummarizer | None = None

    # ── lifecycle ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """Set up the doc store, crawler, and summarizer."""
        cfg = self.ctx.capability_config

        # Store
        sql_store = self.ctx.store_factory("docs")
        await sql_store.init()
        self._store = DocStore(sql_store)
        await self._store.init()  # Run DocStore migrations

        # Crawler
        self._crawler = DocCrawler(
            user_agent=cfg.get("user_agent", "mdrouter-docbot/1.0"),
            max_concurrent=cfg.get("max_concurrent_requests", 5),
            request_delay=cfg.get("request_delay_seconds", 0.5),
            max_pages=cfg.get("max_pages_per_site", 500),
        )

        # Summarizer (cost-saving: only if enabled)
        summ_cfg = self.ctx.config.summarization
        if summ_cfg.enabled:
            self._summarizer = DocSummarizer(
                router=self.ctx.router,
                model=summ_cfg.model,
                max_concurrent=summ_cfg.max_concurrent,
                max_tokens_per_day=summ_cfg.max_tokens_per_day,
                max_chunk_tokens=summ_cfg.max_chunk_tokens,
                prompt=summ_cfg.prompt,
            )
        else:
            logger.info("Summarization disabled — LLM costs are zero")

        logger.info("DocsCapability initialized")

    async def shutdown(self) -> None:
        if self._store:
            await self._store._store.close()

    # ── scheduled tasks ─────────────────────────────────────────

    def scheduled_tasks(self) -> list[ScheduledTask]:
        cfg = self.ctx.capability_config
        interval = cfg.get("crawl_interval_hours", 24)

        async def _re_crawl_all() -> None:
            if not self._store:
                return
            sources = await self._store.list_sources()
            for source in sources:
                if source.get("status") == "active":
                    await self._refresh_source(source["name"])

        return [
            ScheduledTask(
                name="re_crawl",
                interval_hours=interval,
                coroutine=_re_crawl_all,
                run_on_startup=False,
            )
        ]

    # ── tools ───────────────────────────────────────────────────

    def register_tools(self, mcp: "FastMCP") -> None:
        self._mcp = mcp

        @mcp.tool()
        async def doc_search(
            query: str,
            source: str | None = None,
            limit: int = 10,
        ) -> str:
            """Search crawled documentation by keyword.

            Returns ranked results with page titles, URLs, and relevant snippets.
            Optionally scope to a specific source (e.g. 'fastapi', 'pydantic').

            Args:
                query: Search query string.
                source: Optional source name to scope the search.
                limit: Maximum results to return (default 10, max 20).
            """
            if not self._store:
                return json.dumps({"error": "DocStore not initialized"})

            limit = min(limit, 20)
            results = await self._store.search(
                query=query, source_name=source, limit=limit
            )

            formatted = [
                {
                    "title": r.get("title", "Untitled"),
                    "url": r.get("url", ""),
                    "source": r.get("source_name", ""),
                    "snippet": r.get("snippet", ""),
                    "relevance": r.get("_fts_rank", 0),
                }
                for r in results
            ]
            return json.dumps({"results": formatted, "total": len(formatted)})

        @mcp.tool()
        async def doc_sources() -> str:
            """List all documentation sources that have been crawled.

            Returns source name, base URL, page count, last crawl time, and status.
            """
            if not self._store:
                return json.dumps({"error": "DocStore not initialized"})

            sources = await self._store.list_sources()
            formatted = [
                {
                    "name": s["name"],
                    "base_url": s["base_url"],
                    "page_count": s.get("page_count", 0),
                    "last_crawl": s.get("last_crawl", "never"),
                    "status": s.get("status", "unknown"),
                }
                for s in sources
            ]
            return json.dumps({"sources": formatted, "total": len(formatted)})

        @mcp.tool()
        async def doc_crawl(
            url: str,
            name: str,
            max_pages: int | None = None,
        ) -> str:
            """Crawl a documentation site and index it for search.

            Provide the base URL of the documentation site and a short name
            (e.g. 'fastapi', 'pydantic'). The crawler respects robots.txt
            and sitemaps. Pages are stored for full-text search.

            Args:
                url: Base URL of the documentation site.
                name: Short name to identify this source.
                max_pages: Optional page limit for this crawl.
            """
            if not self._store or not self._crawler:
                return json.dumps({"error": "Crawler not initialized"})

            result = await self._crawler.crawl_site(
                base_url=url,
                source_name=name,
                doc_store=self._store,
                max_pages_override=max_pages,
            )

            # Auto-summarize after crawl if summarizer is enabled (cost-saving: only new pages)
            if self._summarizer and result.pages_changed > 0:
                await self._summarizer.summarize_source(
                    source_name=name, doc_store=self._store
                )

            return json.dumps({
                "source": name,
                "url": url,
                "pages_found": result.pages_found,
                "pages_new": result.pages_new,
                "pages_updated": result.pages_updated,
                "pages_skipped": result.pages_skipped,
                "duration_seconds": round(result.duration_seconds, 1),
                "errors": result.errors[:5],  # Only first 5 errors
            })

        @mcp.tool()
        async def doc_refresh(name: str | None = None) -> str:
            """Re-crawl one or all documentation sources to get the latest content.

            Args:
                name: Source name to refresh, or None to refresh all active sources.
            """
            if not self._store or not self._crawler:
                return json.dumps({"error": "Crawler not initialized"})

            if name:
                source = await self._store.get_source(name)
                if not source:
                    return json.dumps({"error": f"Source '{name}' not found"})
                result = await self._crawler.crawl_site(
                    base_url=source["base_url"],
                    source_name=name,
                    doc_store=self._store,
                )
                if self._summarizer and result.pages_changed > 0:
                    await self._summarizer.summarize_source(
                        source_name=name, doc_store=self._store
                    )
                return json.dumps({
                    "source": name,
                    "pages_new": result.pages_new,
                    "pages_updated": result.pages_updated,
                    "pages_skipped": result.pages_skipped,
                    "duration_seconds": round(result.duration_seconds, 1),
                })

            # Refresh all active sources
            sources = await self._store.list_sources()
            results = {}
            for source in sources:
                if source.get("status") != "active":
                    continue
                result = await self._crawler.crawl_site(
                    base_url=source["base_url"],
                    source_name=source["name"],
                    doc_store=self._store,
                )
                if self._summarizer and result.pages_changed > 0:
                    await self._summarizer.summarize_source(
                        source_name=source["name"],
                        doc_store=self._store,
                    )
                results[source["name"]] = {
                    "pages_new": result.pages_new,
                    "pages_updated": result.pages_updated,
                    "pages_skipped": result.pages_skipped,
                }

            return json.dumps({"refreshed": results})

        @mcp.tool()
        async def doc_page(url: str) -> str:
            """Retrieve the full crawled content of a specific documentation page by URL.

            Args:
                url: Exact URL of the page to retrieve.
            """
            if not self._store:
                return json.dumps({"error": "DocStore not initialized"})

            page = await self._store.get_page_by_url(url)
            if not page:
                return json.dumps({"error": f"Page not found: {url}"})

            summaries = await self._store.get_summaries(page["id"])
            return json.dumps({
                "title": page.get("title", ""),
                "url": page.get("url", ""),
                "content": page.get("content", ""),
                "summaries": [
                    {"chunk": s["chunk_index"], "summary": s["summary"]}
                    for s in summaries
                ],
                "crawled_at": page.get("crawled_at", ""),
            })

    # ── resources ───────────────────────────────────────────────

    def register_resources(self, mcp: "FastMCP") -> None:

        @mcp.resource("doc://sources/{name}/tree")
        async def source_tree(name: str) -> str:
            """Browse the page tree for a documentation source."""
            if not self._store:
                return json.dumps({"error": "DocStore not initialized"})

            source = await self._store.get_source(name)
            if not source:
                return json.dumps({"error": f"Source '{name}' not found"})

            pages = await self._store.list_pages(source["id"], limit=200)
            return json.dumps({
                "source": name,
                "base_url": source["base_url"],
                "page_count": len(pages),
                "pages": [
                    {"id": p["id"], "title": p.get("title", ""), "url": p["url"]}
                    for p in pages
                ],
            })

        @mcp.resource("doc://sources/{name}/pages/{page_id}")
        async def page_content(name: str, page_id: int) -> str:
            """Get the full content of a specific page."""
            if not self._store:
                return json.dumps({"error": "DocStore not initialized"})

            page = await self._store.get_page(page_id)
            if not page:
                return json.dumps({"error": f"Page {page_id} not found"})

            summaries = await self._store.get_summaries(page_id)
            return json.dumps({
                "title": page.get("title", ""),
                "url": page.get("url", ""),
                "content": page.get("content", ""),
                "summaries": [
                    {"chunk": s["chunk_index"], "summary": s["summary"]}
                    for s in summaries
                ],
            })

    # ── helpers ─────────────────────────────────────────────────

    async def _refresh_source(self, name: str) -> None:
        """Refresh a single source (used by scheduler)."""
        if not self._store or not self._crawler:
            return
        source = await self._store.get_source(name)
        if not source or source.get("status") != "active":
            return
        result = await self._crawler.crawl_site(
            base_url=source["base_url"],
            source_name=name,
            doc_store=self._store,
        )
        if self._summarizer and result.pages_changed > 0:
            await self._summarizer.summarize_source(
                source_name=name, doc_store=self._store
            )
