"""DocsCapability — Context7-style documentation tools for AI coding agents.

Tools (verb-first, AI-discoverable):
- search_docs: search pre-computed code + info snippets, Context7-shaped markdown
- crawl_docs: crawl + auto-extract code + auto-summarize a doc site
- get_doc_page: get a page's full code + info summary
- list_doc_sources: list all indexed sources
- refresh_docs: re-crawl + re-extract + re-summarize
- snippets_docs: code-only search and extraction

Prompts:
- init_docs: guide agent through auto-discovering + crawling project docs

Resources:
- doc://sources/{name}/tree: browse page tree for a source
- doc://sources/{name}/pages/{page_id}: full page content

Implementation lives in _tools.py — this file is registration + docstrings only.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal, TYPE_CHECKING

from mdrouter.mcp.framework.capability import Capability, ScheduledTask
from mdrouter.mcp.framework.capability import CapabilityContext
from mdrouter.mcp.capabilities.docs.store import DocStore
from mdrouter.mcp.capabilities.docs.crawler import DocCrawler
from mdrouter.mcp.capabilities.docs.summarizer import DocSummarizer
from mdrouter.mcp.capabilities.docs.heuristic_summarizer import HeuristicSummarizer

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

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
        self._max_response_tokens: int = 1000

    # ── lifecycle ──────────────────────────────────────────────

    async def initialize(self) -> None:
        """Set up the doc store, crawler, and summarizer."""
        cfg = self.ctx.capability_config

        sql_store = self.ctx.store_factory("docs")
        await sql_store.init()
        self._store = DocStore(sql_store)
        await self._store.init()

        self._crawler = DocCrawler(
            user_agent=cfg.get("user_agent", "mdrouter-docbot/1.0"),
            max_concurrent=cfg.get("max_concurrent_requests", 5),
            request_delay=cfg.get("request_delay_seconds", 0.5),
            max_pages=cfg.get("max_pages_per_site", 500),
        )

        summ_cfg = self.ctx.config.summarization
        self._max_response_tokens = cfg.get(
            "max_response_tokens",
            self.ctx.config.max_response_tokens,
        )
        # Two summarizer paths:
        #  - _heuristic_summarizer: always available, zero-cost, runs after every crawl
        #  - _summarizer: LLM-based, only created when enabled in config
        # The LLM path is opt-in per source (via summarize_source tool) and
        # never runs on auto-crawl by default — keeps 100+ page crawls free.
        self._heuristic_summarizer = HeuristicSummarizer()
        if summ_cfg.enabled:
            self._summarizer = DocSummarizer(
                router=self.ctx.router,
                model=summ_cfg.model,
                max_concurrent=summ_cfg.max_concurrent,
                max_tokens_per_day=summ_cfg.max_tokens_per_day,
                max_chunk_tokens=summ_cfg.max_chunk_tokens,
                prompt=summ_cfg.prompt,
                max_response_tokens=self._max_response_tokens,
            )
            logger.info(
                "Summarization enabled (LLM: %s) — auto-summarizing on crawl",
                summ_cfg.model,
            )
        else:
            self._summarizer = None
            logger.info(
                "LLM summarization off — heuristic summaries only "
                "(use summarize_source tool to upgrade a source with LLM)"
            )

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
            active = [
                s
                for s in sources
                if s.get("status") == "active" and not s.get("version")
            ]
            if not active:
                logger.debug("No unversioned active sources to re-crawl")
                return
            sem = asyncio.Semaphore(2)

            async def _refresh_one(name: str) -> None:
                async with sem:
                    await self._refresh_source(name)

            await asyncio.gather(
                *[_refresh_one(s["name"]) for s in active],
                return_exceptions=True,
            )

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
        from mdrouter.mcp.capabilities.docs import _tools

        # ── Primary tools ──────────────────────────────────────

        @mcp.tool()
        async def search_docs(
            query: str,
            source: str | None = None,
            limit: int = 10,
            max_tokens: int = 1000,
            offset: int = 0,
            snippet_type: Literal["all", "code", "info"] = "all",
        ) -> str:
            """Search documentation for code examples and explanations.

            **When to use:** You need to look up an API, find code examples,
            or understand how a library works. Searches both code snippets
            (verbatim from docs) and prose summaries.

            **Parameters:**
            - query (str): Natural language or code search. E.g. "how to define routes", "Depends"
            - source (str, optional): Scope to a source. E.g. "fastapi", "pydantic"
            - limit (int, default=10): Max results (1-20)
            - max_tokens (int, default=1000): Token budget per result (60% code, 40% info)
            - offset (int, default=0): Pagination offset
            - snippet_type (str, default="all"): "all", "code", or "info"

            **Returns:** Markdown — each result has `### Code Examples` with
            language-tagged fences, and `### Info` with LLM prose.
            Results separated by `---`.
            """
            return await _tools.search_docs(
                self,
                query=query,
                source=source,
                limit=limit,
                max_tokens=max_tokens,
                offset=offset,
                snippet_type=snippet_type,
            )

        @mcp.tool()
        async def crawl_docs(
            url: str,
            name: str,
            version: str | None = None,
            max_pages: int | None = None,
        ) -> str:
            """Crawl a documentation site and index it for search.

            **When to use:** You want to index a library's docs for future
            lookups. Crawls the site, extracts code blocks verbatim, and
            auto-generates prose summaries. Versioned sources are immutable
            (never re-crawled automatically).

            **Parameters:**
            - url (str): Base URL of the doc site. E.g. "https://fastapi.tiangolo.com/"
            - name (str): Short name for this source. E.g. "fastapi"
            - version (str, optional): Pin to a version. E.g. "0.115.0"
            - max_pages (int, optional): Page limit

            **Returns:** JSON status with pages_found, pages_new, pages_skipped, duration_seconds.
            """
            return await _tools.crawl_docs(
                self,
                url=url,
                name=name,
                version=version,
                max_pages=max_pages,
            )

        @mcp.tool()
        async def get_doc_page(
            url: str,
            max_tokens: int = 1000,
            include_code: bool = True,
        ) -> str:
            """Get a page's Context7-shaped summary (code + info).

            **When to use:** You want the full summary of a specific doc page.
            Returns verbatim code snippets plus LLM-generated prose.

            **Parameters:**
            - url (str): Exact URL of the page.
            - max_tokens (int, default=1000): Token cap. 0 = no limit.
            - include_code (bool, default=True): Include code blocks.

            **Returns:** Markdown with `### Code Examples` and prose.
            """
            return await _tools.get_doc_page(
                self,
                url=url,
                max_tokens=max_tokens,
                include_code=include_code,
            )

        @mcp.tool()
        async def list_doc_sources() -> str:
            """List all indexed documentation sources.

            **When to use:** Check what documentation is available, when it
            was last crawled, and whether it's versioned (immutable) or
            unversioned (periodically refreshed).

            **Returns:** JSON array of sources with name, base_url, version,
            page_count, last_crawl, status.
            """
            return await _tools.list_doc_sources(self)

        @mcp.tool()
        async def refresh_docs(
            name: str | None = None,
            force: bool = False,
        ) -> str:
            """Re-crawl one or all documentation sources.

            **When to use:** Update docs to pick up changes in unversioned
            ("latest") sources. Versioned sources are skipped unless force=True.

            **Parameters:**
            - name (str, optional): Source to refresh, or None for all active.
            - force (bool, default=False): Refresh even versioned/immutable sources.

            **Returns:** JSON with per-source page counts.
            """
            return await _tools.refresh_docs(self, name=name, force=force)

        @mcp.tool()
        async def snippets_docs(
            query: str,
            source: str | None = None,
            limit: int = 10,
            max_tokens: int = 1000,
            language: str | None = None,
        ) -> str:
            """Search and return only code snippets (no prose).

            **When to use:** You need working code examples for an API,
            not explanations. Code is extracted verbatim from source HTML.

            **Parameters:**
            - query (str): Search query matching code identifiers.
            - source (str, optional): Scope to a source.
            - limit (int, default=10): Max code snippets.
            - max_tokens (int, default=1000): Total token cap.
            - language (str, optional): Filter by language. E.g. "python", "javascript".

            **Returns:** Markdown — language-tagged code fences with source context.
            """
            return await _tools.snippets_docs(
                self,
                query=query,
                source=source,
                limit=limit,
                max_tokens=max_tokens,
                language=language,
            )

        @mcp.tool()
        async def summarize_source(
            name: str,
            method: Literal["heuristic", "llm"] = "heuristic",
            max_pages: int | None = None,
        ) -> str:
            """Generate (or regenerate) info snippets for a doc source.

            **When to use:** The default crawl uses the heuristic summarizer
            (free, no LLM). Call this tool to upgrade a source to higher
            quality via LLM summarization, or to re-run summaries after
            upgrading the source content.

            **Cost model:**
            - method="heuristic" (default): zero cost. Uses section headings
              and parameter patterns from the raw page text.
            - method="llm": one LLM call per page chunk. Capped by
              `summarization.max_tokens_per_day`. Replaces any existing
              heuristic summaries for the source.

            **Parameters:**
            - name (str): Source name (e.g. "fastapi", "react").
            - method (str, default="heuristic"): "heuristic" or "llm".
            - max_pages (int, optional): Limit pages to summarize.

            **Returns:** JSON with pages_processed, pages_skipped, duration.
            """
            return await _tools.summarize_source(
                self,
                name=name,
                method=method,
                max_pages=max_pages,
            )

        @mcp.tool()
        async def resolve_library(
            name: str,
            version: str | None = None,
        ) -> str:
            """Resolve a library name (or URL) to a documentation URL.

            **When to use:** You know a library name (e.g. "fastapi",
            "react", "click") and need to find its documentation site
            before calling crawl_docs. Optionally pin to a specific version
            to make the source immutable (skipped on auto-refresh).

            **Parameters:**
            - name (str): Library name (e.g. "fastapi") or a docs URL.
            - version (str, optional): Pin to a specific version
              (e.g. "0.115.0") so the source is treated as immutable.

            **Returns:** JSON with library, doc_url, suggested_source_name,
            version, method (e.g. "known_mapping", "readthedocs_pattern").
            """
            from mdrouter.mcp.capabilities.docs.resolver import (
                resolve_library_async,
            )

            result = await resolve_library_async(name, version)
            if result is None:
                return json.dumps(
                    {
                        "error": f"Could not resolve '{name}'. Try passing a full URL instead.",
                    }
                )
            return json.dumps(result)

        # ── Deprecated aliases (backward compat) ─────────────────

        @mcp.tool()
        async def doc_search(
            query: str,
            source: str | None = None,
            limit: int = 10,
        ) -> str:
            """[DEPRECATED] Use search_docs instead."""
            return await search_docs(query=query, source=source, limit=limit)

        @mcp.tool()
        async def doc_sources() -> str:
            """[DEPRECATED] Use list_doc_sources instead."""
            return await list_doc_sources()

        @mcp.tool()
        async def doc_crawl(
            url: str,
            name: str,
            max_pages: int | None = None,
        ) -> str:
            """[DEPRECATED] Use crawl_docs instead."""
            return await crawl_docs(url=url, name=name, max_pages=max_pages)

        @mcp.tool()
        async def doc_refresh(name: str | None = None) -> str:
            """[DEPRECATED] Use refresh_docs instead."""
            return await refresh_docs(name=name)

        @mcp.tool()
        async def doc_page(
            url: str,
            max_tokens: int = 1000,
        ) -> str:
            """[DEPRECATED] Use get_doc_page instead."""
            return await get_doc_page(url=url, max_tokens=max_tokens)

    # ── resources ───────────────────────────────────────────────

    def register_resources(self, mcp: "FastMCP") -> None:
        from mdrouter.mcp.capabilities.docs import _tools

        @mcp.resource("doc://sources/{name}/tree")
        async def source_tree(name: str) -> str:
            """Browse the page tree for a documentation source."""
            return await _tools.source_tree_resource(self, name)

        @mcp.resource("doc://sources/{name}/pages/{page_id}")
        async def page_content(name: str, page_id: int) -> str:
            """Get the full content of a specific page."""
            return await _tools.page_content_resource(self, name, page_id)

    # ── prompts ─────────────────────────────────────────────────

    def register_prompts(self, mcp: "FastMCP") -> None:
        from mdrouter.mcp.capabilities.docs import _tools

        @mcp.prompt()
        async def init_docs(project_path: str = ".") -> str:
            """Initialize mdrouter docs for this project.

            Guides the AI agent through automatic documentation setup:
            1. Read the project's dependency file to discover libraries
            2. For each major dependency, call resolve_library(name, version)
            3. For each resolved URL, call crawl_docs(url, name, version)
            4. After crawling, code examples and summaries are auto-generated
            5. Use search_docs(query, source=<name>) to look up APIs

            **Usage:** The AI agent should invoke this prompt, then follow
            the instructions step by step using the tools above.
            """
            return _tools.init_docs_prompt(project_path)

    # ── helpers ─────────────────────────────────────────────────

    async def _refresh_source(self, name: str) -> None:
        """Refresh a single source: crawl + extract code + summarize."""
        if not self._store or not self._crawler:
            return
        source = await self._store.get_source(name)
        if not source or source.get("status") != "active":
            return
        if source.get("version"):
            logger.debug("Skipping versioned source '%s'", name)
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
