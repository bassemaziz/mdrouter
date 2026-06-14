"""Tests for DocSummarizer — token budget, caching, cost control."""

from __future__ import annotations

import pytest

from mdrouter.mcp.capabilities.docs.summarizer import (
    DocSummarizer,
    TokenBudget,
    _DEFAULT_PROMPT,
)


class TestTokenBudget:
    def test_unlimited(self):
        budget = TokenBudget(max_tokens_per_day=0)
        assert budget.remaining > 1_000_000
        assert budget.can_spend(999_999)

    def test_limited(self):
        budget = TokenBudget(max_tokens_per_day=1000)
        assert budget.remaining == 1000
        assert budget.can_spend(500)

    def test_spend_reduces_remaining(self):
        budget = TokenBudget(max_tokens_per_day=1000)
        budget.spend(300)
        assert budget.remaining == 700
        assert budget.used == 300

    def test_cannot_exceed_budget(self):
        budget = TokenBudget(max_tokens_per_day=1000)
        budget.spend(900)
        assert not budget.can_spend(200)
        assert budget.can_spend(100)

    def test_multiple_spends(self):
        budget = TokenBudget(max_tokens_per_day=1000)
        budget.spend(100)
        budget.spend(200)
        budget.spend(300)
        assert budget.used == 600
        assert budget.remaining == 400


class TestDocSummarizer:
    def test_constructor_defaults(self):
        # Minimal construction with a mock router
        summ = DocSummarizer(
            router=None,
            model="test/model",
            max_tokens_per_day=5000,
        )
        assert summ.model == "test/model"
        assert summ._budget.max_tokens_per_day == 5000

    def test_default_prompt(self):
        prompt = _DEFAULT_PROMPT
        assert "summariz" in prompt.lower()
        assert "code" in prompt.lower()

    @pytest.mark.asyncio
    async def test_summarize_source_no_pages(self):
        """Should handle empty source gracefully."""

        class FakeDocStore:
            async def get_source(self, name):
                return {"id": 1, "name": name}

            async def list_pages(self, source_id, limit=9999):
                return []

        summ = DocSummarizer(router=None, max_tokens_per_day=5000)
        result = await summ.summarize_source("test", FakeDocStore())
        assert result.pages_processed == 0
        assert result.chunks_summarized == 0

    @pytest.mark.asyncio
    async def test_summarize_skips_already_summarized(self):
        """Should skip pages that already have info snippets (cost-saving)."""

        class FakeDocStore:
            async def get_source(self, name):
                return {"id": 1, "name": name}

            async def list_pages(self, source_id, limit=9999):
                return [
                    {"id": 1, "content": "Some text."},
                    {"id": 2, "content": ""},  # No content to process
                ]

            async def get_info_snippets(self, page_id):
                return "[{}]" if page_id == 1 else "[]"

        summ = DocSummarizer(router=None, max_tokens_per_day=5000)
        result = await summ.summarize_source("test", FakeDocStore())
        assert result.pages_processed == 0  # No actual API calls
        assert result.tokens_used == 0

    @pytest.mark.asyncio
    async def test_budget_exceeded_stops_processing(self):
        """When budget is exceeded, no more chunks should be processed."""
        call_count = 0

        class FakeRouter:
            async def chat_once(self, messages, model, options):
                nonlocal call_count
                call_count += 1
                return (
                    {"message": {"content": "summary"}},
                    {"prompt_tokens": 500, "completion_tokens": 100},
                )

        class FakeDocStore:
            async def set_info_snippets(
                self, page_id, snippets_json, tokens_used=0, model=""
            ):
                pass

        # Budget only enough for 1 chunk
        summ = DocSummarizer(
            router=FakeRouter(),
            model="test/model",
            max_tokens_per_day=600,  # Only ~1 call worth
            max_concurrent=1,
        )

        from mdrouter.mcp.capabilities.docs.summarizer import SummarizeResult

        result = SummarizeResult()
        page = {"id": 1, "content": "Long sentence. " * 200}
        await summ._summarize_page(page, FakeDocStore(), result)

        # Should have made at most 1 call before budget exhausted
        assert call_count <= 1
        assert result.budget_exceeded or call_count <= 1
