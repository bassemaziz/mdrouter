"""Tests for the SQLiteStore — namespacing, FTS5, content hash dedup."""

from __future__ import annotations

import pytest

from mdrouter.mcp.framework.store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    """Create a test store with a unique namespace."""
    db_path = tmp_path / "test.db"
    s = SQLiteStore(namespace="test", db_path=str(db_path))
    await s.init()
    yield s
    await s.close()


async def test_init_creates_db_file(store, tmp_path):
    """Database file should exist after init."""
    assert tmp_path.joinpath("test.db").exists()


async def test_init_enables_wal(store):
    """WAL journal mode should be enabled."""
    rows = await store.fetch_all("PRAGMA journal_mode")
    assert rows[0]["journal_mode"] == "wal"


async def test_execute_and_fetch(store):
    await store.execute("CREATE TABLE test_items (id INTEGER PRIMARY KEY, name TEXT)")
    await store.execute("INSERT INTO test_items (name) VALUES (?)", ("hello",))
    row = await store.fetch_one("SELECT * FROM test_items WHERE name=?", ("hello",))
    assert row is not None
    assert row["name"] == "hello"


async def test_fetch_all(store):
    await store.execute("CREATE TABLE test_items (id INTEGER PRIMARY KEY, name TEXT)")
    await store.execute("INSERT INTO test_items (name) VALUES (?)", ("a",))
    await store.execute("INSERT INTO test_items (name) VALUES (?)", ("b",))
    rows = await store.fetch_all("SELECT * FROM test_items ORDER BY name")
    assert len(rows) == 2
    assert rows[0]["name"] == "a"
    assert rows[1]["name"] == "b"


async def test_namespace_isolation(tmp_path):
    """Two stores with different namespaces should have separate tables."""
    s1 = SQLiteStore(namespace="ns1", db_path=str(tmp_path / "test.db"))
    s2 = SQLiteStore(namespace="ns2", db_path=str(tmp_path / "test.db"))
    await s1.init()
    await s2.init()

    await s1.execute("CREATE TABLE ns1_test (id INTEGER PRIMARY KEY, val TEXT)")
    await s1.execute("INSERT INTO ns1_test (val) VALUES (?)", ("ns1_value",))

    # ns2 should not see ns1's table
    try:
        await s2.fetch_all("SELECT * FROM ns1_test")
        assert False, "Should have raised"
    except Exception:
        pass

    await s1.close()
    await s2.close()


async def test_migrations_tracking(store):
    """Migrations should be tracked and idempotent."""
    await store.run_migrations([
        "CREATE TABLE test_v0 (id INTEGER PRIMARY KEY)",
        "CREATE TABLE test_v1 (id INTEGER PRIMARY KEY, extra TEXT)",
    ])

    # Check migration records
    rows = await store.fetch_all(
        "SELECT * FROM _migrations WHERE namespace=? ORDER BY version", ("test",)
    )
    assert len(rows) == 2
    assert rows[0]["version"] == 0
    assert rows[1]["version"] == 1

    # Running again should be a no-op
    await store.run_migrations([
        "CREATE TABLE test_v0 (id INTEGER PRIMARY KEY)",
        "CREATE TABLE test_v1 (id INTEGER PRIMARY KEY, extra TEXT)",
    ])
    rows = await store.fetch_all(
        "SELECT * FROM _migrations WHERE namespace=? ORDER BY version", ("test",)
    )
    assert len(rows) == 2  # Still only 2


async def test_content_hash_dedup(store):
    """Content hash should be deterministic and findable."""
    text = "This is test content for deduplication."
    h = store.content_hash(text)
    assert len(h) == 32

    # Same content = same hash
    assert store.content_hash(text) == h

    # Different content = different hash
    assert store.content_hash(text + " changed") != h


async def test_fts_create_and_search(store):
    """FTS5 should support creation and search."""
    await store.execute("CREATE TABLE test_pages (id INTEGER PRIMARY KEY, title TEXT, content TEXT)")
    await store.execute("INSERT INTO test_pages (title, content) VALUES (?, ?)", ("Page 1", "Python async programming guide"))
    await store.execute("INSERT INTO test_pages (title, content) VALUES (?, ?)", ("Page 2", "Rust ownership and borrowing"))
    await store.execute("INSERT INTO test_pages (title, content) VALUES (?, ?)", ("Page 3", "Python decorators explained"))

    await store.create_fts("test_fts", ["title", "content"], content_table="test_pages")

    # Rebuild FTS index (external content mode needs it)
    await store.execute("INSERT INTO test_fts(test_fts) VALUES('rebuild')")

    results = await store.search_fts("test_fts", "Python", limit=5)
    assert len(results) == 2
    # Both results should mention Python
    titles = [r["title"] for r in results]
    assert "Page 1" in titles
    assert "Page 3" in titles
