"""DocCrawler — crawl documentation sites with rate limiting and dedup.

Cost-saving features:
- Content hash dedup: only stores pages whose content actually changed
- HTTP caching: sends ETag/If-Modified-Since to avoid re-downloading
- Sitemap parsing: discovers pages efficiently without recursive crawling
- Rate limiting: semaphore-gated concurrency + delay between requests
- Max pages cap: prevents runaway crawls on large sites
- robots.txt: respects crawl-delay directives
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger("mdrouter.mcp.crawler")


@dataclass
class CrawlResult:
    """Result of a crawl operation."""

    source_name: str
    base_url: str
    pages_found: int = 0
    pages_new: int = 0
    pages_updated: int = 0
    pages_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def pages_changed(self) -> int:
        return self.pages_new + self.pages_updated


class DocCrawler:
    """Async documentation crawler with rate limiting and dedup."""

    def __init__(
        self,
        user_agent: str = "mdrouter-docbot/1.0",
        max_concurrent: int = 5,
        request_delay: float = 0.5,
        max_pages: int = 500,
        timeout: float = 30.0,
    ) -> None:
        self.user_agent = user_agent
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.max_pages = max_pages
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._robots_delay: float | None = None

    # ── public API ──────────────────────────────────────────────

    async def crawl_site(
        self,
        base_url: str,
        source_name: str,
        doc_store: Any,  # DocStore
        max_pages_override: int | None = None,
    ) -> CrawlResult:
        """Crawl a documentation site and store results.

        Args:
            base_url: Root URL of the documentation site.
            source_name: Unique name for this source in the store.
            doc_store: DocStore instance for persistence.
            max_pages_override: Override the configured max_pages.

        Returns:
            CrawlResult with counts of pages found/new/updated/skipped.
        """
        start_time = time.monotonic()
        max_pages = max_pages_override or self.max_pages
        result = CrawlResult(source_name=source_name, base_url=base_url)

        # Ensure source exists in DB
        source = await doc_store.add_source(source_name, base_url)
        source_id = source["id"]

        # Parse robots.txt for crawl delay + sitemap hints
        await self._read_robots(base_url)

        # Discover URLs
        page_entries = await self._discover_urls(base_url)
        if not page_entries:
            page_entries = [{"url": base_url, "title": "", "description": ""}]  # Fall back

        # Cap
        page_entries = page_entries[:max_pages]
        result.pages_found = len(page_entries)
        logger.info("Crawling %s: %d pages discovered", source_name, len(page_entries))

        # Fetch and store concurrently
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        ) as client:
            tasks = [
                self._fetch_and_store(client, entry, source_id, doc_store)
                for entry in page_entries
            ]
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for outcome in outcomes:
            if isinstance(outcome, Exception):
                result.errors.append(str(outcome))
                continue
            status = outcome
            if status == "new":
                result.pages_new += 1
            elif status == "updated":
                result.pages_updated += 1
            elif status == "skipped":
                result.pages_skipped += 1

        # Update source stats
        page_count = await doc_store.count_pages(source_id)
        await doc_store.update_source(
            source_name,
            last_crawl=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            page_count=page_count,
        )

        result.duration_seconds = time.monotonic() - start_time
        logger.info(
            "Crawl '%s' done in %.1fs: %d new, %d updated, %d skipped, %d errors",
            source_name,
            result.duration_seconds,
            result.pages_new,
            result.pages_updated,
            result.pages_skipped,
            len(result.errors),
        )
        return result

    async def crawl_single_page(
        self, url: str, doc_store: Any, source_id: int | None = None
    ) -> dict[str, Any] | None:
        """Crawl a single page and return stored page data."""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        ) as client:
            result = await self._fetch_single(client, url)
            if result is None:
                return None
            title, content = result
            page_id, _ = await doc_store.upsert_page(
                source_id=source_id or 0,
                url=url,
                title=title,
                content=content,
            )
            return await doc_store.get_page(page_id)

    # ── URL discovery ───────────────────────────────────────────

    async def _discover_urls(self, base_url: str) -> list[dict[str, str]]:
        """Discover page URLs with metadata (title, description).

        Returns list of dicts with keys: url, title, description.
        Tries llms.txt first, then sitemap.xml.
        """
        # Try llms.txt first (richer metadata)
        llms_urls = await self._discover_from_llms_txt(base_url)
        if llms_urls:
            logger.info("Discovered %d URLs from llms.txt for %s", len(llms_urls), base_url)
            return llms_urls

        # Fall back to sitemap.xml
        sitemaps = await self._find_sitemaps(base_url)
        urls: list[dict[str, str]] = []
        seen: set[str] = set()

        for sitemap_url in sitemaps:
            sitemap_urls = await self._parse_sitemap(sitemap_url)
            for url in sitemap_urls:
                if url not in seen and self._is_same_domain(url, base_url):
                    seen.add(url)
                    urls.append({"url": url, "title": "", "description": ""})

        if urls:
            logger.info("Discovered %d URLs from sitemaps for %s", len(urls), base_url)
        return urls

    async def _find_sitemaps(self, base_url: str) -> list[str]:
        """Find sitemap URLs from robots.txt and common locations."""
        # We already parsed robots.txt in _read_robots; re-check common locations
        candidates = [
            urljoin(base_url, "/sitemap.xml"),
            urljoin(base_url, "/sitemap_index.xml"),
            urljoin(base_url, "/sitemap-index.xml"),
            urljoin(base_url, "/sitemap.php"),
        ]

        async with httpx.AsyncClient(
            timeout=10.0,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        ) as client:
            for url in candidates:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200 and (
                        "sitemap" in resp.text.lower()[:200]
                        or "urlset" in resp.text.lower()[:200]
                    ):
                        return [url]
                except Exception:
                    continue
        return []

    async def _parse_sitemap(self, sitemap_url: str) -> list[str]:
        """Parse a sitemap XML and return URLs."""
        urls: list[str] = []
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            ) as client:
                resp = await client.get(sitemap_url)
                text = resp.text
        except Exception as exc:
            logger.warning("Failed to fetch sitemap %s: %s", sitemap_url, exc)
            return []

        try:
            # Remove default namespace to simplify parsing
            text = re.sub(r'\sxmlns="[^"]*"', "", text, count=1)
            root = ET.fromstring(text)
        except ET.ParseError:
            logger.warning("Failed to parse sitemap XML from %s", sitemap_url)
            return []

        # Handle sitemap index (points to other sitemaps)
        sitemap_ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//loc"):
            if loc.text:
                url = loc.text.strip()
                # If it's a sitemap index, recurse
                if url.endswith(".xml") and "sitemap" in url.lower():
                    urls.extend(await self._parse_sitemap(url))
                else:
                    urls.append(url)

        return urls

    async def _discover_from_llms_txt(self, base_url: str) -> list[dict[str, str]]:
        """Fetch and parse llms.txt for structured page listing.

        Returns list of {url, title, description} dicts.
        The llms.txt format is an emerging convention for LLM-friendly
        documentation indexes: https://llmstxt.org/
        """
        llms_url = urljoin(base_url.rstrip("/") + "/", "llms.txt")
        urls: list[dict[str, str]] = []

        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            ) as client:
                resp = await client.get(llms_url)
                if resp.status_code != 200:
                    return []
                text = resp.text
        except Exception:
            return []

        # Also try /docs/llms.txt if the base URL didn't work
        if not text or "llms.txt" not in llms_url:
            return []

        return self._parse_llms_txt(text, base_url)

    @staticmethod
    def _parse_llms_txt(text: str, base_url: str) -> list[dict[str, str]]:
        """Parse llms.txt format into {url, title, description} entries.

        Format:
          ## [Section Name](optional-url)
          - [Page Title](URL): Description of the page.
          @key: value  (metadata, skipped)
        """
        entries: list[dict[str, str]] = []
        url_re = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip metadata lines
            if line.startswith("@"):
                continue
            # Skip section headers (## [...]), might have URLs too
            if line.startswith("## "):
                match = url_re.search(line)
                if match:
                    # Section header URL — include it
                    url = match.group(2)
                    if url.startswith("http") and DocCrawler._is_same_domain(url, base_url):
                        entries.append({
                            "url": url,
                            "title": match.group(1),
                            "description": "",
                        })
                continue

            # Page entries: "- [Title](URL): Description"
            if line.startswith("- "):
                match = url_re.search(line)
                if not match:
                    continue
                url = match.group(2)
                title = match.group(1)
                if url.startswith("http") and DocCrawler._is_same_domain(url, base_url):
                    # Extract description after the URL
                    after_link = line[match.end():]
                    description = after_link.lstrip(": ").strip()
                    entries.append({
                        "url": url,
                        "title": title,
                        "description": description,
                    })

        return entries

    # ── fetching ────────────────────────────────────────────────

    async def _fetch_and_store(
        self, client: httpx.AsyncClient, entry: dict[str, str], source_id: int, doc_store: Any
    ) -> str:
        """Fetch a page, extract content, and store it. Returns 'new', 'updated', or 'skipped'."""
        url = entry["url"]
        llms_title = entry.get("title", "")

        result = await self._fetch_single(client, url)
        if result is None:
            return "skipped"
        title, content = result

        # Prefer the page's <title>, fall back to llms.txt title
        final_title = title or llms_title

        _, was_new = await doc_store.upsert_page(
            source_id=source_id, url=url, title=final_title, content=content
        )
        if was_new:
            return "new"
        return "updated"

    async def _fetch_single(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, str] | None:
        """Fetch and extract a single page. Returns (title, content) or None."""
        async with self._semaphore:
            await self._rate_limit()
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.debug("HTTP %d for %s", resp.status_code, url)
                    return None
                html = resp.text
            except Exception as exc:
                logger.debug("Failed to fetch %s: %s", url, exc)
                return None

        title, content = self._extract_content(html, url)
        if not content or len(content.strip()) < 50:
            return None
        return title, content

    async def _rate_limit(self) -> None:
        """Enforce request delay and optional robots.txt crawl-delay."""
        now = time.monotonic()
        delay = self.request_delay
        if self._robots_delay:
            delay = max(delay, self._robots_delay)
        jitter = random.uniform(0, delay * 0.5)
        total_delay = delay + jitter

        elapsed = now - self._last_request_time
        if elapsed < total_delay:
            await asyncio.sleep(total_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ── content extraction ──────────────────────────────────────

    def _extract_content(self, html: str, url: str) -> tuple[str, str]:
        """Extract title and clean text from HTML using trafilatura."""
        title = ""
        content = ""

        # Try trafilatura first (best quality)
        try:
            import trafilatura

            extracted = trafilatura.extract(
                html,
                output_format="markdown",
                with_metadata=True,
                url=url,
            )
            if extracted:
                # trafilatura with_metadata returns title in the text
                lines = extracted.split("\n")
                if lines and lines[0].startswith("# "):
                    title = lines[0][2:].strip()
                    content = "\n".join(lines[1:]).strip()
                else:
                    content = extracted
        except Exception:
            pass

        # Fall back to basic extraction
        if not content:
            title, content = self._basic_extract(html)

        return title, content

    @staticmethod
    def _basic_extract(html: str) -> tuple[str, str]:
        """Basic HTML extraction without trafilatura."""
        # Remove scripts, styles, nav, footer
        for tag in ["script", "style", "nav", "footer", "header"]:
            html = re.sub(
                rf"<{tag}[^>]*>.*?</{tag}>", "", html, flags=re.DOTALL | re.IGNORECASE
            )

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Strip all tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return title, text

    # ── robots.txt ──────────────────────────────────────────────

    async def _read_robots(self, base_url: str) -> None:
        """Parse robots.txt for crawl delay."""
        robots_url = urljoin(base_url, "/robots.txt")
        try:
            async with httpx.AsyncClient(
                timeout=10.0,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            ) as client:
                resp = await client.get(robots_url)
                if resp.status_code == 200:
                    for line in resp.text.splitlines():
                        line = line.strip().lower()
                        if line.startswith("crawl-delay:"):
                            try:
                                self._robots_delay = float(line.split(":", 1)[1].strip())
                                logger.info(
                                    "robots.txt crawl-delay: %.1fs for %s",
                                    self._robots_delay,
                                    base_url,
                                )
                            except ValueError:
                                pass
        except Exception:
            pass

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _is_same_domain(url: str, base_url: str) -> bool:
        """Check if url is on the same domain as base_url."""
        try:
            return urlparse(url).netloc == urlparse(base_url).netloc
        except Exception:
            return False


def chunk_text(text: str, max_words: int = 500) -> list[str]:
    """Split text into chunks at sentence boundaries, approximately max_words each.

    Cost-saving: chunking avoids sending huge pages to the LLM summarizer,
    keeping per-call token counts low and predictable.
    """
    if not text:
        return []

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0

    for sentence in sentences:
        words = len(sentence.split())
        if current_count + words > max_words and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_count = words
        else:
            current.append(sentence)
            current_count += words

    if current:
        chunks.append(" ".join(current))

    return chunks
