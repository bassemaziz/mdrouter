"""DocSummarizer — LLM-based documentation chunk summarization.

Uses the shared mdrouter ModelRouter to call DeepSeek Flash (or configured model)
for summarizing crawled documentation chunks.

Cost-saving measures (every one of these matters):
1. CACHE-FIRST: checks DocStore before calling LLM — never re-summarize
2. HASH-DRIVEN: skips pages whose content_hash hasn't changed since last crawl
3. TOKEN BUDGET: enforces max_tokens_per_day; refuses calls when exceeded
4. CHUNK TRUNCATION: chunks are capped at max_chunk_tokens before sending to LLM
5. SHORT PROMPT: uses a concise summarization prompt to minimize input tokens
6. BATCHED: pages with multiple chunks are processed concurrently (but semaphore-limited)
7. TOKEN TRACKING: logs every call's token usage for audit
"""

from __future__ import annotations

import asyncio
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
    """Summarizes documentation chunks using the ModelRouter LLM."""

    def __init__(
        self,
        router: Any,  # ModelRouter
        model: str = "deepseek/deepseek-v4-flash",
        max_concurrent: int = 3,
        max_tokens_per_day: int = 200_000,
        max_chunk_tokens: int = 4000,
        prompt: str = "",
    ) -> None:
        self._router = router
        self.model = model
        self._max_concurrent = max_concurrent
        self._max_chunk_tokens = max_chunk_tokens
        self._prompt = prompt
        self._budget = TokenBudget(max_tokens_per_day)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ── public API ──────────────────────────────────────────────

    async def summarize_source(
        self,
        source_name: str,
        doc_store: Any,  # DocStore
        max_pages: int | None = None,
    ) -> SummarizeResult:
        """Summarize all unsummarized pages for a source.

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

        # Filter to pages without summaries (cost-saving: skip already-done)
        to_process: list[int] = []
        for page in pages:
            if not await doc_store.has_summaries(page["id"]):
                to_process.append(page["id"])

        if not to_process:
            logger.info("All %d pages already summarized for '%s'", len(pages), source_name)
            result.duration_seconds = time.monotonic() - start
            return result

        logger.info(
            "Summarizing %d/%d pages for '%s'",
            len(to_process),
            len(pages),
            source_name,
        )

        # Process concurrently with semaphore
        tasks = [
            self._summarize_page(page_id, doc_store, result)
            for page_id in to_process
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        result.duration_seconds = time.monotonic() - start
        logger.info(
            "Summarization of '%s' done in %.1fs: %d chunks summarized, %d skipped, %d tokens used",
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
        page_id: int,
        doc_store: Any,
        result: SummarizeResult,
    ) -> None:
        """Summarize a single page's chunks."""
        page = await doc_store.get_page(page_id)
        if not page or not page.get("content"):
            return

        from mdrouter.mcp.capabilities.docs.crawler import chunk_text

        chunks = chunk_text(page["content"], max_words=500)
        if not chunks:
            return

        result.pages_processed += 1

        for idx, chunk in enumerate(chunks):
            # Truncate to max tokens (rough estimate: 1 token ≈ 4 chars)
            max_chars = self._max_chunk_tokens * 4
            if len(chunk) > max_chars:
                chunk = chunk[:max_chars]

            # Cost-saving: check budget before each call
            estimated_tokens = len(chunk) // 3 + 200  # rough: input + output estimate
            if not self._budget.can_spend(estimated_tokens):
                result.budget_exceeded = True
                logger.warning(
                    "Token budget exceeded (%d/%d used). Skipping remaining chunks.",
                    self._budget.used,
                    self._budget.max_tokens_per_day,
                )
                return

            async with self._semaphore:
                try:
                    summary, tokens = await self._call_llm(chunk)
                    await doc_store.save_summary(
                        page_id=page_id,
                        chunk_index=idx,
                        chunk_text=chunk,
                        summary=summary,
                        model_used=self.model,
                        tokens_used=tokens,
                    )
                    self._budget.spend(tokens)
                    result.tokens_used += tokens
                    result.chunks_summarized += 1
                except Exception as exc:
                    result.errors.append(f"Page {page_id} chunk {idx}: {exc}")
                    result.chunks_skipped += 1

    async def _call_llm(self, chunk_text: str) -> tuple[str, int]:
        """Call the LLM to summarize a chunk. Returns (summary, tokens_used)."""
        messages = [
            {"role": "system", "content": self._prompt or self._default_prompt()},
            {"role": "user", "content": chunk_text},
        ]

        response, meta = await self._router.chat_once(
            messages=messages,
            model=self.model,
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

    @staticmethod
    def _default_prompt() -> str:
        return (
            "Summarize this documentation page into 3-5 key points. "
            "Focus on APIs, parameters, return types, and usage examples. "
            "Be concise — each point one sentence."
        )
