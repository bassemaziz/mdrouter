"""DocStore — documentation storage with FTS5 full-text search.

Extends the framework SQLiteStore with docs-specific tables:
- sources: documentation sources being tracked
- pages: crawled pages with content hash for dedup
- pages_fts: FTS5 index over title + content
- summaries: LLM-generated chunk summaries (cost-saving: cached)
"""

from __future__ import annotations

from typing import Any

from mdrouter.mcp.framework.store import SQLiteStore, _content_hash

_MIGRATIONS = [
    # v0: initial schema
    """
    CREATE TABLE IF NOT EXISTS {ns}sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        base_url TEXT NOT NULL,
        last_crawl TEXT,
        page_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'active'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {ns}pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL REFERENCES {ns}sources(id) ON DELETE CASCADE,
        url TEXT UNIQUE NOT NULL,
        title TEXT DEFAULT '',
        content TEXT NOT NULL DEFAULT '',
        content_hash TEXT NOT NULL DEFAULT '',
        crawled_at TEXT NOT NULL DEFAULT (datetime('now')),
        content_length INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {ns}summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_id INTEGER NOT NULL REFERENCES {ns}pages(id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL DEFAULT 0,
        chunk_text TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        model_used TEXT NOT NULL DEFAULT '',
        tokens_used INTEGER DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    # v1: Context7-shaped code + info snippets, search index, version support
    """
    ALTER TABLE {ns}pages ADD COLUMN code_snippets_json TEXT DEFAULT '[]'
    """,
    """
    ALTER TABLE {ns}pages ADD COLUMN info_snippets_json TEXT DEFAULT '[]'
    """,
    """
    ALTER TABLE {ns}pages ADD COLUMN summary_tokens INTEGER DEFAULT 0
    """,
    """
    ALTER TABLE {ns}pages ADD COLUMN summary_model TEXT DEFAULT ''
    """,
    """
    ALTER TABLE {ns}pages ADD COLUMN summarized_at TEXT
    """,
    """
    ALTER TABLE {ns}sources ADD COLUMN version TEXT
    """,
    # v2: FTS5 search index for combined code + info (contentless, manually synced)
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS {ns}search_fts USING fts5(
        code_text,
        info_text,
        tokenize='porter unicode61'
    )
    """,
]


class DocStore:
    """High-level docs storage API wrapping SQLiteStore."""

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    # ── lifecycle ──────────────────────────────────────────────

    async def init(self) -> None:
        """Run migrations and create FTS index."""
        ns = self._store.namespace
        formatted = [sql.replace("{ns}", f"{ns}_") for sql in _MIGRATIONS]
        await self._store.run_migrations(formatted)
        # Regular FTS5 (not external content) — we insert into both tables
        await self._store.create_fts(
            table=f"{ns}_pages_fts",
            columns=["title", "content"],
        )

    # ── sources ─────────────────────────────────────────────────

    async def add_source(
        self, name: str, base_url: str, version: str | None = None
    ) -> dict[str, Any]:
        """Add or return existing source. Versioned sources are immutable."""
        ns = self._store.namespace
        existing = await self._store.fetch_one(
            f"SELECT * FROM {ns}_sources WHERE name=?", (name,)
        )
        if existing:
            # Update version if newly provided and not yet set
            if version and not existing.get("version"):
                await self._store.execute(
                    f"UPDATE {ns}_sources SET version=? WHERE name=?",
                    (version, name),
                )
                existing["version"] = version
            return existing
        await self._store.execute(
            f"INSERT INTO {ns}_sources (name, base_url, version, status) VALUES (?, ?, ?, 'active')",
            (name, base_url, version),
        )
        return await self._store.fetch_one(
            f"SELECT * FROM {ns}_sources WHERE name=?", (name,)
        ) or {"id": -1, "name": name, "base_url": base_url}

    async def list_sources(self) -> list[dict[str, Any]]:
        ns = self._store.namespace
        return await self._store.fetch_all(f"SELECT * FROM {ns}_sources ORDER BY name")

    async def get_source(self, name: str) -> dict[str, Any] | None:
        ns = self._store.namespace
        return await self._store.fetch_one(
            f"SELECT * FROM {ns}_sources WHERE name=?", (name,)
        )

    async def update_source(self, name: str, **kwargs: Any) -> None:
        ns = self._store.namespace
        if not kwargs:
            return
        sets = ", ".join(f"{k}=?" for k in kwargs)
        values = tuple(kwargs.values()) + (name,)
        await self._store.execute(
            f"UPDATE {ns}_sources SET {sets} WHERE name=?", values
        )

    # ── pages ───────────────────────────────────────────────────

    async def upsert_page(
        self, source_id: int, url: str, title: str, content: str
    ) -> tuple[int, bool]:
        """Insert or update a page. Returns (page_id, was_new).

        Cost-saving: checks content_hash before updating. If the
        content hasn't changed since last crawl, the page is skipped.
        """
        ns = self._store.namespace
        h = _content_hash(content)

        existing = await self._store.fetch_one(
            f"SELECT id, content_hash FROM {ns}_pages WHERE url=?", (url,)
        )
        if existing:
            if existing["content_hash"] == h:
                # Unchanged — but still ensure search_fts is populated
                # (existing DBs may lack FTS entries from before the migration)
                await self._sync_search_fts(existing["id"])
                return existing["id"], False
            await self._store.execute(
                f"UPDATE {ns}_pages SET title=?, content=?, content_hash=?, "
                f"crawled_at=datetime('now'), content_length=? WHERE id=?",
                (title, content, h, len(content), existing["id"]),
            )
            # Update FTS entry
            await self._store.execute(
                f"DELETE FROM {ns}_pages_fts WHERE rowid=?", (existing["id"],)
            )
            await self._store.execute(
                f"INSERT INTO {ns}_pages_fts (rowid, title, content) VALUES (?, ?, ?)",
                (existing["id"], title, content),
            )
            # Sync search_fts with content as fallback
            await self._sync_search_fts(existing["id"])
            return existing["id"], True

        await self._store.execute(
            f"INSERT INTO {ns}_pages (source_id, url, title, content, content_hash, crawled_at, content_length) "
            f"VALUES (?, ?, ?, ?, ?, datetime('now'), ?)",
            (source_id, url, title, content, h, len(content)),
        )
        new_row = await self._store.fetch_one(
            f"SELECT id FROM {ns}_pages WHERE url=?", (url,)
        )
        page_id = new_row["id"] if new_row else -1
        # Also insert into FTS index
        await self._store.execute(
            f"INSERT INTO {ns}_pages_fts (rowid, title, content) VALUES (?, ?, ?)",
            (page_id, title, content),
        )
        # Populate search_fts with content as fallback (enriched later by summarizer)
        await self._sync_search_fts(page_id)
        return page_id, True

    async def get_page(self, page_id: int) -> dict[str, Any] | None:
        ns = self._store.namespace
        return await self._store.fetch_one(
            f"SELECT * FROM {ns}_pages WHERE id=?", (page_id,)
        )

    async def get_page_by_url(self, url: str) -> dict[str, Any] | None:
        ns = self._store.namespace
        return await self._store.fetch_one(
            f"SELECT * FROM {ns}_pages WHERE url=?", (url,)
        )

    async def list_pages(
        self, source_id: int, limit: int = 100
    ) -> list[dict[str, Any]]:
        ns = self._store.namespace
        return await self._store.fetch_all(
            f"SELECT id, source_id, url, title, content, crawled_at, content_length "
            f"FROM {ns}_pages WHERE source_id=? ORDER BY url LIMIT ?",
            (source_id, limit),
        )

    async def count_pages(self, source_id: int) -> int:
        ns = self._store.namespace
        row = await self._store.fetch_one(
            f"SELECT COUNT(*) as cnt FROM {ns}_pages WHERE source_id=?", (source_id,)
        )
        return row["cnt"] if row else 0

    # ── search ──────────────────────────────────────────────────

    async def search(
        self, query: str, source_name: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Full-text search across all pages, optionally scoped to a source."""
        ns = self._store.namespace
        fts_table = f"{ns}_pages_fts"
        pages_table = f"{ns}_pages"
        sources_table = f"{ns}_sources"

        extra_where = ""
        extra_params: tuple[Any, ...] = ()
        if source_name:
            source = await self.get_source(source_name)
            if source:
                extra_where = "AND p.source_id = ?"
                extra_params = (source["id"],)

        # Build a join to get full page info with FTS ranking
        safe_query = self._store._escape_fts_query(query)
        terms = [f'"{t}"*' if " " not in t else f'"{t}"' for t in safe_query.split()]
        fts_query = " AND ".join(terms) if terms else safe_query

        sql = (
            f"SELECT p.id, p.source_id, p.url, p.title, "
            f"snippet({fts_table}, 1, '<mark>', '</mark>', '...', 40) AS snippet, "
            f"s.name AS source_name, rank AS _fts_rank "
            f"FROM {fts_table} "
            f"JOIN {pages_table} p ON p.id = {fts_table}.rowid "
            f"JOIN {sources_table} s ON s.id = p.source_id "
            f"WHERE {fts_table} MATCH ? {extra_where} "
            f"ORDER BY rank LIMIT ?"
        )
        params: tuple[Any, ...] = (fts_query,) + extra_params + (limit,)
        return await self._store.fetch_all(sql, params)

    # ── summaries ───────────────────────────────────────────────

    async def has_summaries(self, page_id: int) -> bool:
        ns = self._store.namespace
        row = await self._store.fetch_one(
            f"SELECT COUNT(*) as cnt FROM {ns}_summaries WHERE page_id=?", (page_id,)
        )
        return (row["cnt"] if row else 0) > 0

    async def save_summary(
        self,
        page_id: int,
        chunk_index: int,
        chunk_text: str,
        summary: str,
        model_used: str = "",
        tokens_used: int = 0,
    ) -> None:
        ns = self._store.namespace
        await self._store.execute(
            f"INSERT INTO {ns}_summaries (page_id, chunk_index, chunk_text, summary, model_used, tokens_used) "
            f"VALUES (?, ?, ?, ?, ?, ?)",
            (page_id, chunk_index, chunk_text, summary, model_used, tokens_used),
        )

    async def get_summaries(self, page_id: int) -> list[dict[str, Any]]:
        ns = self._store.namespace
        return await self._store.fetch_all(
            f"SELECT * FROM {ns}_summaries WHERE page_id=? ORDER BY chunk_index",
            (page_id,),
        )

    async def total_tokens_used(self) -> int:
        """Cost tracking: total tokens consumed by summarization."""
        ns = self._store.namespace
        row = await self._store.fetch_one(
            f"SELECT COALESCE(SUM(tokens_used), 0) as total FROM {ns}_summaries"
        )
        return row["total"] if row else 0

    # ── code snippets (verbatim from HTML) ────────────────────

    async def set_code_snippets(self, page_id: int, snippets_json: str) -> None:
        """Store extracted code snippets as JSON."""
        ns = self._store.namespace
        await self._store.execute(
            f"UPDATE {ns}_pages SET code_snippets_json=? WHERE id=?",
            (snippets_json, page_id),
        )
        await self._sync_search_fts(page_id)

    async def set_info_snippets(
        self,
        page_id: int,
        snippets_json: str,
        tokens_used: int = 0,
        model: str = "",
    ) -> None:
        """Store LLM-generated info snippets as JSON."""
        ns = self._store.namespace
        await self._store.execute(
            f"UPDATE {ns}_pages SET info_snippets_json=?, summary_tokens=?, "
            f"summary_model=?, summarized_at=datetime('now') WHERE id=?",
            (snippets_json, tokens_used, model, page_id),
        )
        await self._sync_search_fts(page_id)

    async def get_code_snippets(self, page_id: int) -> str:
        """Get extracted code snippets JSON for a page."""
        ns = self._store.namespace
        row = await self._store.fetch_one(
            f"SELECT code_snippets_json FROM {ns}_pages WHERE id=?", (page_id,)
        )
        return row["code_snippets_json"] if row else "[]"

    async def get_info_snippets(self, page_id: int) -> str:
        """Get LLM-generated info snippets JSON for a page."""
        ns = self._store.namespace
        row = await self._store.fetch_one(
            f"SELECT info_snippets_json FROM {ns}_pages WHERE id=?", (page_id,)
        )
        return row["info_snippets_json"] if row else "[]"

    # ── FTS5 search over code + info ──────────────────────────

    async def _sync_search_fts(self, page_id: int) -> None:
        """Sync a page's code + info text into the search_fts index.

        Falls back to page content and title if no code/info snippets exist.
        This ensures search works immediately after crawling, even before
        summarization populates the richer data.
        """
        ns = self._store.namespace
        row = await self._store.fetch_one(
            f"SELECT code_snippets_json, info_snippets_json, content, title "
            f"FROM {ns}_pages WHERE id=?",
            (page_id,),
        )
        if not row:
            return
        # Extract text from code/info snippets, or fall back to page content
        code_text = self._extract_text_from_snippets(row["code_snippets_json"] or "[]")
        info_text = self._extract_text_from_snippets(row["info_snippets_json"] or "[]")
        # Fallback: if no snippets yet, use content + title for searchability
        if not code_text and not info_text:
            info_text = (row.get("content") or "") + " " + (row.get("title") or "")
        # Delete old entry if any
        await self._store.execute(
            f"DELETE FROM {ns}_search_fts WHERE rowid=?", (page_id,)
        )
        await self._store.execute(
            f"INSERT INTO {ns}_search_fts (rowid, code_text, info_text) VALUES (?, ?, ?)",
            (page_id, code_text, info_text),
        )

    @staticmethod
    def _extract_text_from_snippets(json_str: str) -> str:
        """Extract searchable text from a JSON array of snippets."""
        import json as _json

        try:
            snippets = _json.loads(json_str)
        except _json.JSONDecodeError:
            return ""
        parts: list[str] = []
        for s in snippets:
            if isinstance(s, dict):
                parts.append(s.get("code", ""))
                parts.append(s.get("content", ""))
                parts.append(s.get("code_title", ""))
        return " ".join(parts)

    async def search_combined(
        self,
        query: str,
        source_name: str | None = None,
        limit: int = 10,
        offset: int = 0,
        snippet_type: str = "all",
    ) -> tuple[list[dict[str, Any]], int]:
        """Search code + info snippets with pagination.

        Returns (results, total_count).
        snippet_type: "all", "code", or "info".
        """
        ns = self._store.namespace
        fts_table = f"{ns}_search_fts"
        pages_table = f"{ns}_pages"
        sources_table = f"{ns}_sources"

        safe_query = self._store._escape_fts_query(query)
        terms = [f'"{t}"*' if " " not in t else f'"{t}"' for t in safe_query.split()]
        fts_query = " AND ".join(terms) if terms else safe_query

        # Build WHERE based on snippet_type
        extra_where = ""
        extra_params: tuple[Any, ...] = ()
        if source_name:
            source = await self.get_source(source_name)
            if source:
                extra_where = "AND p.source_id = ?"
                extra_params = (source["id"],)

        # Count query
        count_sql = (
            f"SELECT COUNT(*) as cnt FROM {fts_table} "
            f"JOIN {pages_table} p ON p.id = {fts_table}.rowid "
            f"WHERE {fts_table} MATCH ? {extra_where}"
        )
        count_row = await self._store.fetch_one(count_sql, (fts_query,) + extra_params)
        total = count_row["cnt"] if count_row else 0

        # Results query
        sql = (
            f"SELECT p.id, p.source_id, p.url, p.title, "
            f"p.code_snippets_json, p.info_snippets_json, "
            f"s.name AS source_name, rank AS _fts_rank "
            f"FROM {fts_table} "
            f"JOIN {pages_table} p ON p.id = {fts_table}.rowid "
            f"JOIN {sources_table} s ON s.id = p.source_id "
            f"WHERE {fts_table} MATCH ? {extra_where} "
            f"ORDER BY rank LIMIT ? OFFSET ?"
        )
        params: tuple[Any, ...] = (fts_query,) + extra_params + (limit, offset)
        results = await self._store.fetch_all(sql, params)

        return results, total
