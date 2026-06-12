"""Tests for DocStore — sources, pages, FTS5 search, summaries."""

from __future__ import annotations

import pytest

from mdrouter.mcp.framework.store import SQLiteStore
from mdrouter.mcp.capabilities.docs.store import DocStore


@pytest.fixture
async def doc_store(tmp_path):
    """Create a test DocStore."""
    db_path = tmp_path / "test.db"
    sql_store = SQLiteStore(namespace="docs", db_path=str(db_path))
    await sql_store.init()
    store = DocStore(sql_store)
    await store.init()
    yield store
    await sql_store.close()


# ── sources ────────────────────────────────────────────────────

async def test_add_source(doc_store):
    source = await doc_store.add_source("fastapi", "https://fastapi.tiangolo.com/")
    assert source["name"] == "fastapi"
    assert source["base_url"] == "https://fastapi.tiangolo.com/"
    assert source["status"] == "active"


async def test_add_source_idempotent(doc_store):
    s1 = await doc_store.add_source("fastapi", "https://fastapi.tiangolo.com/")
    s2 = await doc_store.add_source("fastapi", "https://other.com/")  # different URL
    assert s1["id"] == s2["id"]
    assert s2["base_url"] == "https://fastapi.tiangolo.com/"  # Original URL preserved


async def test_list_sources(doc_store):
    await doc_store.add_source("a", "https://a.com")
    await doc_store.add_source("b", "https://b.com")
    sources = await doc_store.list_sources()
    assert len(sources) == 2
    names = [s["name"] for s in sources]
    assert "a" in names
    assert "b" in names


async def test_update_source(doc_store):
    await doc_store.add_source("test", "https://test.com")
    await doc_store.update_source("test", page_count=42, status="paused")
    source = await doc_store.get_source("test")
    assert source["page_count"] == 42
    assert source["status"] == "paused"


# ── pages ──────────────────────────────────────────────────────

async def test_upsert_page_new(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    page_id, was_new = await doc_store.upsert_page(
        source["id"], "https://test.com/page1", "Page One", "Content here"
    )
    assert was_new is True
    assert page_id > 0


async def test_upsert_page_duplicate_skipped(doc_store):
    """Unchanged content should be skipped (cost-saving dedup)."""
    source = await doc_store.add_source("test", "https://test.com")
    page_id1, was_new1 = await doc_store.upsert_page(
        source["id"], "https://test.com/page1", "Page One", "Same content"
    )
    assert was_new1 is True

    page_id2, was_new2 = await doc_store.upsert_page(
        source["id"], "https://test.com/page1", "Updated Title", "Same content"
    )
    assert was_new2 is False  # Skipped — content unchanged
    assert page_id1 == page_id2


async def test_upsert_page_content_changed(doc_store):
    """Changed content should trigger an update."""
    source = await doc_store.add_source("test", "https://test.com")
    page_id1, _ = await doc_store.upsert_page(
        source["id"], "https://test.com/page1", "Page", "Original content"
    )
    page_id2, was_new2 = await doc_store.upsert_page(
        source["id"], "https://test.com/page1", "Page Updated", "New content"
    )
    assert was_new2 is True  # Updated
    assert page_id1 == page_id2


async def test_get_page(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    page_id, _ = await doc_store.upsert_page(
        source["id"], "https://test.com/page1", "The Page", "Content here"
    )
    page = await doc_store.get_page(page_id)
    assert page is not None
    assert page["title"] == "The Page"
    assert page["url"] == "https://test.com/page1"


async def test_count_pages(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    await doc_store.upsert_page(source["id"], "https://test.com/1", "A", "a")
    await doc_store.upsert_page(source["id"], "https://test.com/2", "B", "b")
    count = await doc_store.count_pages(source["id"])
    assert count == 2


# ── search ─────────────────────────────────────────────────────

async def test_search_finds_content(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    page_id, _ = await doc_store.upsert_page(
        source["id"], "https://test.com/1", "Async in Python",
        "Python asyncio provides async/await syntax for concurrent programming."
    )
    # Rebuild FTS
    await doc_store._store.execute("INSERT INTO docs_pages_fts(docs_pages_fts) VALUES('rebuild')")

    results = await doc_store.search("async concurrent", limit=5)
    assert len(results) >= 1
    assert any("Python" in (r.get("title", "")) for r in results)


async def test_search_scoped_to_source(doc_store):
    source1 = await doc_store.add_source("src1", "https://one.com")
    source2 = await doc_store.add_source("src2", "https://two.com")
    await doc_store.upsert_page(source1["id"], "https://one.com/1", "Python Guide", "Python programming")
    await doc_store.upsert_page(source2["id"], "https://two.com/1", "Rust Guide", "Rust programming")
    await doc_store._store.execute("INSERT INTO docs_pages_fts(docs_pages_fts) VALUES('rebuild')")

    results = await doc_store.search("Rust", source_name="src2", limit=5)
    assert len(results) == 1
    assert results[0]["source_name"] == "src2"


# ── summaries ──────────────────────────────────────────────────

async def test_has_summaries_false(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    page_id, _ = await doc_store.upsert_page(source["id"], "https://test.com/1", "P", "c")
    assert await doc_store.has_summaries(page_id) is False


async def test_save_and_get_summaries(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    page_id, _ = await doc_store.upsert_page(source["id"], "https://test.com/1", "P", "c")
    await doc_store.save_summary(page_id, 0, "chunk text", "summary text", "deepseek/deepseek-chat", 150)
    assert await doc_store.has_summaries(page_id) is True
    summaries = await doc_store.get_summaries(page_id)
    assert len(summaries) == 1
    assert summaries[0]["summary"] == "summary text"
    assert summaries[0]["tokens_used"] == 150


async def test_total_tokens_used(doc_store):
    source = await doc_store.add_source("test", "https://test.com")
    page_id, _ = await doc_store.upsert_page(source["id"], "https://test.com/1", "P", "c")
    await doc_store.save_summary(page_id, 0, "c1", "s1", "model", 100)
    await doc_store.save_summary(page_id, 1, "c2", "s2", "model", 200)
    total = await doc_store.total_tokens_used()
    assert total == 300
