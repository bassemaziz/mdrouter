"""Tool implementations for DocsCapability.

All tool logic lives here — no FastMCP decoration, no closure tricks.
Each function takes `cap` (DocsCapability) as its first argument.
__init__.py handles decoration and docstrings (AI agent needs those
at decoration time).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from mdrouter.mcp.capabilities.docs.response_builder import (
    build_search_response,
    build_page_response,
    build_snippets_response,
    parse_search_result,
    parse_snippets_from_json,
)

logger = logging.getLogger("mdrouter.mcp.docs")


async def search_docs(
    cap: Any,
    query: str,
    source: str | None = None,
    limit: int = 10,
    max_tokens: int = 1000,
    offset: int = 0,
    snippet_type: Literal["all", "code", "info"] = "all",
) -> str:
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    limit = min(limit, 20)
    results, total = await cap._store.search_combined(
        query=query,
        source_name=source,
        limit=limit,
        offset=offset,
        snippet_type=snippet_type,
    )

    # Fallback: if search_fts is empty (pre-migration DB), use old FTS on content
    if not results:
        raw = await cap._store.search(
            query=query,
            source_name=source,
            limit=limit,
        )
        if raw:
            # Convert old-format results to SearchResult shape
            from mdrouter.mcp.capabilities.docs.response_builder import SearchResult

            parsed = [
                SearchResult(
                    title=r.get("title", "Untitled"),
                    url=r.get("url", ""),
                    source=r.get("source_name", ""),
                    relevance=float(r.get("_fts_rank", 0)),
                )
                for r in raw
            ]
            return build_search_response(
                parsed,
                max_tokens=max_tokens,
                total_count=len(raw),
                offset=offset,
                snippet_type=snippet_type,
            )

    parsed = [parse_search_result(r) for r in results]
    return build_search_response(
        parsed,
        max_tokens=max_tokens,
        total_count=total,
        offset=offset,
        snippet_type=snippet_type,
    )


async def crawl_docs(
    cap: Any,
    url: str,
    name: str,
    version: str | None = None,
    max_pages: int | None = None,
) -> str:
    if not cap._store or not cap._crawler:
        return json.dumps({"error": "Crawler not initialized"})

    # Register source with version
    await cap._store.add_source(name, url, version=version)

    result = await cap._crawler.crawl_site(
        base_url=url,
        source_name=name,
        doc_store=cap._store,
        max_pages_override=max_pages,
    )

    # Auto-summarize new/changed pages
    # Default path: heuristic (free, no LLM). LLM path is opt-in via
    # the summarize_source tool or by setting summarization.enabled=true.
    if result.pages_changed > 0:
        if cap._summarizer:
            await cap._summarizer.summarize_source(
                source_name=name, doc_store=cap._store
            )
        elif cap._heuristic_summarizer:
            await cap._heuristic_summarizer.summarize_source(
                source_name=name, doc_store=cap._store
            )

    return json.dumps(
        {
            "source": name,
            "url": url,
            "version": version,
            "pages_found": result.pages_found,
            "pages_new": result.pages_new,
            "pages_updated": result.pages_updated,
            "pages_skipped": result.pages_skipped,
            "duration_seconds": round(result.duration_seconds, 1),
            "errors": result.errors[:5],
        }
    )


async def get_doc_page(
    cap: Any,
    url: str,
    max_tokens: int = 1000,
    include_code: bool = True,
) -> str:
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    page = await cap._store.get_page_by_url(url)
    if not page:
        return json.dumps({"error": f"Page not found: {url}"})

    # Resolve source name from source_id
    source_name = ""
    sid = page.get("source_id")
    if sid:
        ns = cap._store._store.namespace
        src_row = await cap._store._store.fetch_one(
            f"SELECT name FROM {ns}_sources WHERE id=?",
            (sid,),
        )
        if src_row:
            source_name = src_row["name"]

    code_snippets, info_snippets = parse_snippets_from_json(
        page.get("code_snippets_json", "[]")
    )
    _, info2 = parse_snippets_from_json(page.get("info_snippets_json", "[]"))
    if not info_snippets:
        info_snippets = info2

    return build_page_response(
        title=page.get("title", ""),
        url=page.get("url", ""),
        source=source_name,
        code_snippets=code_snippets,
        info_snippets=info_snippets,
        max_tokens=max_tokens,
        include_code=include_code,
    )


async def list_doc_sources(cap: Any) -> str:
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    sources = await cap._store.list_sources()
    formatted = [
        {
            "name": s["name"],
            "base_url": s["base_url"],
            "version": s.get("version"),
            "page_count": s.get("page_count", 0),
            "last_crawl": s.get("last_crawl", "never"),
            "status": s.get("status", "unknown"),
        }
        for s in sources
    ]
    return json.dumps({"sources": formatted, "total": len(formatted)})


async def refresh_docs(
    cap: Any,
    name: str | None = None,
    force: bool = False,
) -> str:
    if not cap._store or not cap._crawler:
        return json.dumps({"error": "Crawler not initialized"})

    if name:
        source = await cap._store.get_source(name)
        if not source:
            return json.dumps({"error": f"Source '{name}' not found"})
        # Skip versioned sources unless forced
        if source.get("version") and not force:
            return json.dumps(
                {
                    "source": name,
                    "version": source["version"],
                    "skipped": True,
                    "reason": "Versioned source (immutable). Use force=True to re-crawl.",
                }
            )
        result = await cap._crawler.crawl_site(
            base_url=source["base_url"],
            source_name=name,
            doc_store=cap._store,
        )
        if result.pages_changed > 0:
            if cap._summarizer:
                await cap._summarizer.summarize_source(
                    source_name=name, doc_store=cap._store
                )
            elif cap._heuristic_summarizer:
                await cap._heuristic_summarizer.summarize_source(
                    source_name=name, doc_store=cap._store
                )
        return json.dumps(
            {
                "source": name,
                "version": source.get("version"),
                "pages_new": result.pages_new,
                "pages_updated": result.pages_updated,
                "pages_skipped": result.pages_skipped,
                "duration_seconds": round(result.duration_seconds, 1),
            }
        )

    # Refresh all active, unversioned sources (or all if force=True)
    sources = await cap._store.list_sources()
    results: dict[str, Any] = {}
    for source in sources:
        if source.get("status") != "active":
            continue
        if source.get("version") and not force:
            results[source["name"]] = {
                "skipped": True,
                "reason": "Versioned source (immutable)",
                "version": source["version"],
            }
            continue
        result = await cap._crawler.crawl_site(
            base_url=source["base_url"],
            source_name=source["name"],
            doc_store=cap._store,
        )
        if result.pages_changed > 0:
            if cap._summarizer:
                await cap._summarizer.summarize_source(
                    source_name=source["name"],
                    doc_store=cap._store,
                )
            elif cap._heuristic_summarizer:
                await cap._heuristic_summarizer.summarize_source(
                    source_name=source["name"],
                    doc_store=cap._store,
                )
        results[source["name"]] = {
            "version": source.get("version"),
            "pages_new": result.pages_new,
            "pages_updated": result.pages_updated,
            "pages_skipped": result.pages_skipped,
        }

    return json.dumps({"refreshed": results})


async def snippets_docs(
    cap: Any,
    query: str,
    source: str | None = None,
    limit: int = 10,
    max_tokens: int = 1000,
    language: str | None = None,
) -> str:
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    limit = min(limit, 20)
    results, _ = await cap._store.search_combined(
        query=query,
        source_name=source,
        limit=limit,
        offset=0,
        snippet_type="code",
    )
    parsed = [parse_search_result(r) for r in results]
    return build_snippets_response(
        parsed,
        max_tokens=max_tokens,
        language=language,
    )


async def summarize_source(
    cap: Any,
    name: str,
    method: str = "heuristic",
    max_pages: int | None = None,
) -> str:
    """Generate info snippets for a source on demand.

    method="heuristic" (default): free, deterministic, no LLM. Replaces
    any existing summaries for the source with heuristic ones.
    method="llm": uses the configured LLM (cost-incurring). Skips
    pages that already have LLM summaries (cache-first).

    Returns JSON with counts and duration.
    """
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    source = await cap._store.get_source(name)
    if not source:
        return json.dumps({"error": f"Source '{name}' not found"})

    import time

    start = time.monotonic()

    if method == "llm":
        if not cap._summarizer:
            return json.dumps(
                {
                    "error": (
                        "LLM summarization is not enabled. Set "
                        "summarization.enabled=true in config/mcp.json, "
                        "or use method='heuristic'."
                    ),
                    "source": name,
                }
            )
        # Reset existing summaries so the LLM re-processes everything
        # (cache-first check inside DocSummarizer would otherwise skip).
        result = await cap._summarizer.summarize_source(
            source_name=name,
            doc_store=cap._store,
            max_pages=max_pages,
        )
        return json.dumps(
            {
                "source": name,
                "method": "llm",
                "model": cap._summarizer.model,
                "pages_processed": result.pages_processed,
                "chunks_summarized": result.chunks_summarized,
                "chunks_skipped": result.chunks_skipped,
                "tokens_used": result.tokens_used,
                "budget_exceeded": result.budget_exceeded,
                "errors": result.errors[:5],
                "duration_seconds": round(result.duration_seconds, 1),
            }
        )

    # Default: heuristic
    if not cap._heuristic_summarizer:
        return json.dumps({"error": "Heuristic summarizer not available"})

    # Clear existing info_snippets_json so heuristic re-processes
    # (cache-first check otherwise skips).
    pages = await cap._store.list_pages(source["id"], limit=9999)
    if max_pages:
        pages = pages[:max_pages]
    ns = cap._store._store.namespace
    await cap._store._store.execute(
        f"UPDATE {ns}_pages SET info_snippets_json='[]', "
        f"summarized_at=NULL, summary_model='' "
        f"WHERE source_id=?",
        (source["id"],),
    )

    result = await cap._heuristic_summarizer.summarize_source(
        source_name=name,
        doc_store=cap._store,
        max_pages=max_pages,
    )
    return json.dumps(
        {
            "source": name,
            "method": "heuristic",
            "model": "heuristic",
            "pages_processed": result.get("pages_processed", 0),
            "pages_skipped": result.get("pages_skipped", 0),
            "cost_usd": 0.0,
            "duration_seconds": round(time.monotonic() - start, 1),
        }
    )


async def source_tree_resource(cap: Any, name: str) -> str:
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    source = await cap._store.get_source(name)
    if not source:
        return json.dumps({"error": f"Source '{name}' not found"})

    pages = await cap._store.list_pages(source["id"], limit=200)
    return json.dumps(
        {
            "source": name,
            "base_url": source["base_url"],
            "page_count": len(pages),
            "pages": [
                {"id": p["id"], "title": p.get("title", ""), "url": p["url"]}
                for p in pages
            ],
        }
    )


async def page_content_resource(cap: Any, name: str, page_id: int) -> str:
    if not cap._store:
        return json.dumps({"error": "DocStore not initialized"})

    page = await cap._store.get_page(page_id)
    if not page:
        return json.dumps({"error": f"Page {page_id} not found"})

    summaries = await cap._store.get_summaries(page_id)
    return json.dumps(
        {
            "title": page.get("title", ""),
            "url": page.get("url", ""),
            "content": page.get("content", ""),
            "summaries": [
                {"chunk": s["chunk_index"], "summary": s["summary"]} for s in summaries
            ],
        }
    )


def init_docs_prompt(project_path: str = ".") -> str:
    return (
        "# Initialize Documentation for This Project\n\n"
        "Follow these steps to index documentation for the project's dependencies.\n\n"
        "## Step 1: Discover Dependencies\n"
        f"Read the dependency file in `{project_path}`. Look for:\n"
        "- `pyproject.toml` or `requirements.txt` (Python)\n"
        "- `package.json` (JavaScript/TypeScript)\n"
        "- `Cargo.toml` (Rust)\n"
        "- `go.mod` (Go)\n\n"
        "## Step 2: Resolve Documentation URLs\n"
        "For each major direct dependency, call:\n"
        "```\n"
        'resolve_library(name="<library>", version="<optional-version>")\n'
        "```\n"
        "It returns a `doc_url` and `suggested_source_name` based on a 45+ library "
        "mapping table, plus readthedocs/docs.rs/pkg.go.dev pattern detection.\n"
        "If resolution fails, fall back to a known docs URL.\n\n"
        "## Step 3: Crawl Each Library\n"
        "For each dependency, call:\n"
        "```\n"
        'crawl_docs(url="<doc_url>", name="<library_name>", version="<version>", max_pages=20)\n'
        "```\n"
        "- Include the version to pin it (versioned sources are immutable — never re-crawled).\n"
        "- Omit version for 'latest' docs (will be auto-refreshed periodically).\n"
        "- `max_pages=20` is a good start; bump to 200+ for heavily-used libs.\n\n"
        "## Cost Note\n"
        "By default, crawl uses the **heuristic summarizer** (zero LLM cost). "
        "It extracts section headings and parameter patterns from the raw page text. "
        "After crawling, you can upgrade a specific source to LLM-quality summaries:\n"
        "```\n"
        'summarize_source(name="<library>", method="llm")\n'
        "```\n"
        "Use this only for libraries you actively work with — it costs tokens per page.\n\n"
        "## Step 4: Verify\n"
        "Call `list_doc_sources()` to see what was indexed.\n\n"
        "## Step 5: Use\n"
        'Call `search_docs(query="<question>", source="<library>")` to look up APIs.\n'
        'Call `snippets_docs(query="<code search>", language="python")` for code examples only.\n'
        'Call `get_doc_page(url="<docs_url>")` for the full summary of a specific page.\n'
    )
