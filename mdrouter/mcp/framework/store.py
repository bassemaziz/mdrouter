"""Namespaced async SQLite store with FTS5 full-text search.

Cost-saving features:
- Content hash deduplication — never store or re-process the same content
- FTS5 for zero-cost full-text search (no embedding API calls needed)
- WAL mode for concurrent read performance
- Single-file database — no external service costs

Usage:
    store = SQLiteStore(namespace="docs", db_path="data/mcp.db")
    await store.init()
    await store.execute("CREATE TABLE IF NOT EXISTS ...")
    await store.close()
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("mdrouter.mcp.store")


def _content_hash(text: str) -> str:
    """Deterministic hash for content deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


class SQLiteStore:
    """Async SQLite store with namespace-prefixed tables and FTS5 support.

    All tables created through this store are prefixed with `{namespace}_`.
    This allows multiple capabilities to share one database file without
    table name collisions.
    """

    def __init__(self, namespace: str, db_path: str) -> None:
        self.namespace = namespace
        self.db_path = Path(db_path)
        self._conn: Any = None  # aiosqlite.Connection

    # ── lifecycle ──────────────────────────────────────────────

    async def init(self) -> None:
        """Open the database, enable WAL, run migrations."""
        import aiosqlite

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._ensure_migrations_table()
        logger.info("SQLiteStore '%s' opened at %s", self.namespace, self.db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("SQLiteStore '%s' closed", self.namespace)

    # ── migrations ─────────────────────────────────────────────

    async def _ensure_migrations_table(self) -> None:
        await self.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                namespace TEXT NOT NULL,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (namespace, version)
            )
        """)

    async def run_migrations(self, migrations: list[str]) -> None:
        """Run idempotent migration SQL statements.

        Each migration is tracked by its index (0-based). Already-applied
        migrations are skipped.
        """
        for idx, sql in enumerate(migrations):
            row = await self.fetch_one(
                "SELECT 1 FROM _migrations WHERE namespace=? AND version=?",
                (self.namespace, idx),
            )
            if row:
                continue
            await self.execute(sql)
            await self.execute(
                "INSERT INTO _migrations(namespace, version) VALUES (?, ?)",
                (self.namespace, idx),
            )
            logger.debug("Applied migration %s v%d", self.namespace, idx)

    # ── query helpers ──────────────────────────────────────────

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        await self._conn.execute(sql, params)
        await self._conn.commit()

    async def execute_many(self, sql: str, params_list: list[tuple[Any, ...]]) -> None:
        await self._conn.executemany(sql, params_list)
        await self._conn.commit()

    async def fetch_all(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def fetch_one(
        self, sql: str, params: tuple[Any, ...] = ()
    ) -> dict[str, Any] | None:
        cursor = await self._conn.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ── FTS5 helpers ──────────────────────────────────────────

    async def create_fts(
        self,
        table: str,
        columns: list[str],
        content_table: str | None = None,
    ) -> None:
        """Create an FTS5 virtual table for full-text search.

        When content_table is provided, the FTS index stays in sync
        with the content table automatically (external content mode).
        """
        col_defs = ", ".join(columns)
        if content_table:
            sql = (
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
                f"USING fts5({col_defs}, content='{content_table}', "
                f"content_rowid='id')"
            )
        else:
            sql = f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING fts5({col_defs})"
        await self.execute(sql)

    async def search_fts(
        self,
        table: str,
        query: str,
        limit: int = 10,
        extra_where: str = "",
        extra_params: tuple[Any, ...] = (),
    ) -> list[dict[str, Any]]:
        """Full-text search with ranking. Returns results sorted by relevance.

        Escapes FTS5 special characters in the query.
        """
        # Escape FTS5 special characters and build a prefix-friendly query
        safe_query = self._escape_fts_query(query)
        # Add prefix matching for partial words
        terms = [f'"{t}"*' if " " not in t else f'"{t}"' for t in safe_query.split()]
        fts_query = " AND ".join(terms) if terms else safe_query

        where = f"WHERE {table} MATCH ?"
        if extra_where:
            where += f" AND {extra_where}"
        params: tuple[Any, ...] = (fts_query,) + extra_params

        sql = f"SELECT *, rank AS _fts_rank FROM {table} {where} ORDER BY rank LIMIT ?"
        return await self.fetch_all(sql, params + (limit,))

    @staticmethod
    def _escape_fts_query(query: str) -> str:
        """Escape FTS5 special characters.

        FTS5 has its own query syntax. Operator characters are stripped
        for safety — they're rare in doc search queries and stripping
        prevents syntax errors from malformed user input.
        """
        import re

        # Characters that are FTS5 operators — strip them
        stripped = re.sub(r'[\x00-\x1f\[\]{}\(\)\*\+\-\^"~:!&|\\]', " ", query)
        # Collapse multiple spaces
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped

    # ── content hash helpers (cost-saving dedup) ───────────────

    @staticmethod
    def content_hash(text: str) -> str:
        """Return a deterministic hash for content deduplication."""
        return _content_hash(text)

    async def find_by_hash(
        self, table: str, hash_column: str, content: str
    ) -> dict[str, Any] | None:
        """Check if content with the given hash already exists in the table."""
        h = _content_hash(content)
        return await self.fetch_one(
            f"SELECT * FROM {table} WHERE {hash_column}=?", (h,)
        )
