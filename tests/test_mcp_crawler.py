"""Tests for DocCrawler — sitemap parsing, content extraction, dedup."""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response

from mdrouter.mcp.capabilities.docs.crawler import DocCrawler, chunk_text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_single():
    chunks = chunk_text("Short text.", max_words=500)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_chunk_text_multiple_sentences():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_text(text, max_words=2)
    assert len(chunks) >= 2


def test_chunk_text_respects_boundaries():
    """Chunks should split at sentence boundaries, not mid-word."""
    text = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + "."
    chunks = chunk_text(text, max_words=10)
    for chunk in chunks:
        assert chunk.endswith(".")


def test_basic_extract():
    html = "<html><head><title>Test Page</title></head><body><p>Hello world</p></body></html>"
    crawler = DocCrawler()
    title, content = crawler._basic_extract(html)
    assert title == "Test Page"
    assert "Hello world" in content


def test_basic_extract_strips_tags():
    html = "<div><script>evil()</script><p>Clean text</p><nav>Menu</nav></div>"
    crawler = DocCrawler()
    _, content = crawler._basic_extract(html)
    assert "Clean text" in content
    assert "evil()" not in content
    assert "Menu" not in content  # nav is stripped


@pytest.mark.asyncio
async def test_discover_urls_from_sitemap():
    """Should parse sitemap.xml and return URLs."""
    sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
        <url><loc>https://example.com/docs/page1.html</loc></url>
        <url><loc>https://example.com/docs/page2.html</loc></url>
    </urlset>"""

    with respx.mock(assert_all_called=False) as mock:
        # robots.txt
        mock.get("https://example.com/robots.txt").return_value = Response(200, text="User-agent: *\nAllow: /")
        # llms.txt — not found, fall back to sitemap
        mock.get("https://example.com/llms.txt").return_value = Response(404)
        # sitemap candidates
        mock.get("https://example.com/sitemap.xml").return_value = Response(200, text=sitemap_xml)
        mock.get("https://example.com/sitemap_index.xml").return_value = Response(404)
        mock.get("https://example.com/sitemap-index.xml").return_value = Response(404)
        mock.get("https://example.com/sitemap.php").return_value = Response(404)

        crawler = DocCrawler()
        entries = await crawler._discover_urls("https://example.com/docs/")
        urls = [e["url"] for e in entries]
        assert len(urls) == 2
        assert "https://example.com/docs/page1.html" in urls
        assert "https://example.com/docs/page2.html" in urls


@pytest.mark.asyncio
async def test_crawl_single_page():
    """Should fetch and extract a single page."""
    html = """<html><head><title>Async Guide</title></head>
    <body><article><p>Python asyncio provides async/await.</p></article></body></html>"""

    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://example.com/async").return_value = Response(200, text=html)

        crawler = DocCrawler()

        class FakeDocStore:
            async def add_source(self, name, url):
                return {"id": 1, "name": name, "base_url": url}
            async def upsert_page(self, source_id, url, title, content):
                return (1, True)
            async def get_page(self, page_id):
                return {"id": 1, "title": "Async Guide", "url": "https://example.com/async", "content": "Python asyncio"}

        page = await crawler.crawl_single_page("https://example.com/async", FakeDocStore(), source_id=1)
        assert page is not None
        assert "async" in page["content"].lower() or "asyncio" in page["content"].lower()


@pytest.mark.asyncio
async def test_fetch_single_404():
    """Should return None for 404 responses."""
    with respx.mock(assert_all_called=False) as mock:
        mock.get("https://example.com/missing").return_value = Response(404)

        crawler = DocCrawler()
        async with httpx.AsyncClient(timeout=10, headers={"User-Agent": "test"}) as client:
            result = await crawler._fetch_single(client, "https://example.com/missing")
        assert result is None


def test_is_same_domain():
    assert DocCrawler._is_same_domain("https://example.com/page", "https://example.com/")
    assert DocCrawler._is_same_domain("https://docs.example.com/page", "https://docs.example.com/")
    assert not DocCrawler._is_same_domain("https://other.com/page", "https://example.com/")


def test_parse_llms_txt():
    """Should parse llms.txt format into structured entries."""
    llms_txt = """# Next.js Docs
@doc-version: 16.2.9

## [Getting Started](https://nextjs.org/docs/app/getting-started)

- [Installation](https://nextjs.org/docs/app/getting-started/installation): Learn how to create a new Next.js application.
- [Project Structure](https://nextjs.org/docs/app/getting-started/project-structure): Learn the folder and file conventions.

## [API Reference](https://nextjs.org/docs/app/api-reference)

- [Components](https://nextjs.org/docs/app/api-reference/components): API Reference for Next.js built-in components.
"""

    entries = DocCrawler._parse_llms_txt(llms_txt, "https://nextjs.org/")
    assert len(entries) == 5  # 2 section headers + 3 page entries
    
    # Check section headers
    assert entries[0]["title"] == "Getting Started"
    assert entries[0]["url"] == "https://nextjs.org/docs/app/getting-started"

    # Check page entries (Installation is entries[1], not [2])
    assert entries[1]["title"] == "Installation"
    assert entries[1]["description"] == "Learn how to create a new Next.js application."


def test_parse_llms_txt_filters_external_domains():
    """Should only return URLs from the same domain."""
    llms_txt = """- [External](https://other.com/page): External site
- [Local](https://nextjs.org/docs/page): Local page"""

    entries = DocCrawler._parse_llms_txt(llms_txt, "https://nextjs.org/")
    assert len(entries) == 1
    assert entries[0]["url"] == "https://nextjs.org/docs/page"
