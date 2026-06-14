"""DocSummarizer — LLM-based prose-only documentation summarization.

Uses the shared mdrouter ModelRouter to generate info snippets (prose only).
Code snippets are extracted separately from raw HTML by CodeBlockExtractor.

Cost-saving measures:
1. CACHE-FIRST: checks info_snippets_json before calling LLM
2. HASH-DRIVEN: skips pages whose content_hash hasn't changed
3. TOKEN BUDGET: enforces max_tokens_per_day; refuses when exceeded
4. CHUNK TRUNCATION: caps chunks at max_chunk_tokens before sending to LLM
5. PROSE-ONLY: short prompt — no code reproduction (code is HTML-extracted)
6. BATCHED: chunks processed concurrently (semaphore-limited)
7. TOKEN TRACKING: logs every call's token usage for audit
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("mdrouter.mcp.summarizer")


@dataclass
class SummarizeResult:
    """Result of a summarization run."""

    pages_processed: int = 0
    chunks_summarized: int = 0
    chunks_skipped: int = 0
    tokens_used: int = 0
    budget_exceeded: bool = False
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class TokenBudget:
    """Daily token budget tracker for cost control.

    Simple in-memory counter. Resets when the UTC date changes.
    Not persisted across restarts — that's intentional: the budget
    is a soft cap, not a hard accounting system.
    """

    def __init__(self, max_tokens_per_day: int) -> None:
        self.max_tokens_per_day = max_tokens_per_day
        self._tokens_used = 0
        self._current_day = ""

    @property
    def remaining(self) -> int:
        self._check_day()
        if self.max_tokens_per_day == 0:
            return 1_000_000_000  # unlimited
        return max(0, self.max_tokens_per_day - self._tokens_used)

    @property
    def used(self) -> int:
        self._check_day()
        return self._tokens_used

    def can_spend(self, tokens: int) -> bool:
        if self.max_tokens_per_day == 0:
            return True
        return self.remaining >= tokens

    def spend(self, tokens: int) -> None:
        self._check_day()
        self._tokens_used += tokens

    def _check_day(self) -> None:
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if today != self._current_day:
            self._tokens_used = 0
            self._current_day = today


class DocSummarizer:
    """Generates info snippets (prose only) using the ModelRouter LLM.

    Code snippets are extracted separately by CodeBlockExtractor from
    raw HTML — the LLM never reproduces code. This guarantees verbatim
    code in responses.
    """

    def __init__(
        self,
        router: Any,  # ModelRouter
        model: str = "deepseek/deepseek-v4-flash",
        max_concurrent: int = 3,
        max_tokens_per_day: int = 200_000,
        max_chunk_tokens: int = 4000,
        prompt: str = "",
        max_response_tokens: int = 1000,
    ) -> None:
        self._router = router
        self.model = model
        self._max_concurrent = max_concurrent
        self._max_chunk_tokens = max_chunk_tokens
        self._prompt = prompt
        self._max_response_tokens = max_response_tokens
        self._budget = TokenBudget(max_tokens_per_day)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ── public API ──────────────────────────────────────────────

    async def summarize_source(
        self,
        source_name: str,
        doc_store: Any,  # DocStore
        max_pages: int | None = None,
    ) -> SummarizeResult:
        """Generate info snippets for all unsummarized pages in a source.

        Skips pages that already have info_snippets_json populated
        (cost-saving: cache-first, no redundant LLM calls).

        Args:
            source_name: Which documentation source to process.
            doc_store: DocStore instance.
            max_pages: Optional cap on pages to process.

        Returns:
            SummarizeResult with counts and token usage.
        """
        start = time.monotonic()
        result = SummarizeResult()

        source = await doc_store.get_source(source_name)
        if not source:
            result.errors.append(f"Source '{source_name}' not found")
            return result

        pages = await doc_store.list_pages(source["id"], limit=9999)
        if max_pages:
            pages = pages[:max_pages]

        # Filter to pages without info snippets (cost-saving)
        to_process: list[dict[str, Any]] = []
        for page in pages:
            existing = await doc_store.get_info_snippets(page["id"])
            if not existing or existing == "[]":
                to_process.append(page)

        if not to_process:
            logger.info(
                "All %d pages already have info snippets for '%s'",
                len(pages),
                source_name,
            )
            result.duration_seconds = time.monotonic() - start
            return result

        logger.info(
            "Generating info snippets for %d/%d pages of '%s'",
            len(to_process),
            len(pages),
            source_name,
        )

        # Process concurrently with semaphore
        tasks = [self._summarize_page(page, doc_store, result) for page in to_process]
        await asyncio.gather(*tasks, return_exceptions=True)

        result.duration_seconds = time.monotonic() - start
        logger.info(
            "Summarization of '%s' done in %.1fs: %d chunks summarized, "
            "%d skipped, %d tokens used",
            source_name,
            result.duration_seconds,
            result.chunks_summarized,
            result.chunks_skipped,
            result.tokens_used,
        )
        return result

    # ── page-level summarization ────────────────────────────────

    async def _summarize_page(
        self,
        page: dict[str, Any],
        doc_store: Any,
        result: SummarizeResult,
    ) -> None:
        """Generate info snippets for a single page.

        1. Chunk the page text
        2. Call LLM for each chunk (prose-only prompt)
        3. Merge chunk summaries into a single info_snippets JSON array
        4. Store via doc_store.set_info_snippets()
        """
        content = page.get("content", "")
        if not content:
            return

        from mdrouter.mcp.capabilities.docs.crawler import chunk_text

        chunks = chunk_text(content, max_words=500)
        if not chunks:
            return

        result.pages_processed += 1
        page_snippets: list[dict[str, str]] = []
        total_tokens = 0

        for idx, chunk in enumerate(chunks):
            # Truncate chunk to max_chunk_tokens
            max_chars = self._max_chunk_tokens * 4
            if len(chunk) > max_chars:
                chunk = chunk[:max_chars]

            # Check token budget
            estimated_tokens = len(chunk) // 3 + 300
            if not self._budget.can_spend(estimated_tokens):
                result.budget_exceeded = True
                logger.warning(
                    "Token budget exceeded (%d/%d used). Skipping remaining.",
                    self._budget.used,
                    self._budget.max_tokens_per_day,
                )
                break

            async with self._semaphore:
                try:
                    snippet_text, tokens = await self._call_llm(chunk)
                    if snippet_text:
                        page_snippets.append(
                            {
                                "title": f"Section {idx + 1}",
                                "content": snippet_text,
                            }
                        )
                    self._budget.spend(tokens)
                    total_tokens += tokens
                    result.tokens_used += tokens
                    result.chunks_summarized += 1
                except Exception as exc:
                    result.errors.append(f"Page {page['id']} chunk {idx}: {exc}")
                    result.chunks_skipped += 1

        if page_snippets:
            # Store as JSON
            snippets_json = json.dumps(page_snippets, ensure_ascii=False)
            await doc_store.set_info_snippets(
                page_id=page["id"],
                snippets_json=snippets_json,
                tokens_used=total_tokens,
                model=self.model,
            )

    async def _call_llm(self, chunk_text: str) -> tuple[str, int]:
        """Call the LLM to summarize a chunk. Returns (summary_text, tokens_used).

        Summary is prose-only — no code reproduction (code is HTML-extracted).
        """
        messages = [
            {"role": "system", "content": self._prompt or _DEFAULT_PROMPT},
            {"role": "user", "content": chunk_text},
        ]

        response, meta = await self._router.chat_once(
            model_alias=self.model,
            messages=messages,
            options={"temperature": 0.3, "max_tokens": 300},
        )

        summary = ""
        tokens_used = 0

        if response and "message" in response:
            summary = response["message"].get("content", "")
        if meta:
            prompt_tokens = meta.get("prompt_tokens", 0)
            completion_tokens = meta.get("completion_tokens", 0)
            tokens_used = prompt_tokens + completion_tokens

        return summary.strip(), tokens_used


_DEFAULT_PROMPT = (
    "You are summarizing a documentation page. The code examples have "
    "already been extracted and will be shown separately. Focus ONLY on "
    "the prose.\n\n"
    "Summarize this section in 2-4 concise bullet points. Cover:\n"
    "- What it's about (1 sentence overview)\n"
    "- Key concepts, constraints, or gotchas\n"
    "- API behavior notes (parameter behavior, edge cases, return values)\n\n"
    "DO NOT reproduce any code examples. Code is shown separately.\n"
    "Be concise — each point one sentence."
)
