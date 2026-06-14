"""Heuristic (LLM-free) page summarizer.

Builds info snippets directly from raw text using deterministic rules:

1. First non-empty paragraph → "Overview" section
2. The first sentence of each <h2> section (split on `\n## ` or similar)
   → "Key Concepts" bullets
3. Parameter-like patterns (`name:`, `param_name=`, `**kwargs`) and
   any "Returns:" / "Raises:" lines → "API Behavior Notes" bullets

No external dependencies, no LLM calls. Cost = 0. Quality is lower than
LLM summarization but better than nothing — and good enough for search
and snippet display.

Used as the default summarizer path so crawling 100+ pages is free.
Users can later run the LLM summarizer (DocSummarizer) on demand to
upgrade specific sources to higher-quality prose.
"""

from __future__ import annotations

import re
from typing import Any

# Heuristics for section detection in plain text
_H2_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_H3_PATTERN = re.compile(r"^###\s+(.+)$", re.MULTILINE)
_API_PATTERNS = [
    re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", re.MULTILINE),
    re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^.\n]{5,80})\.", re.MULTILINE),
    re.compile(
        r"^\s*-\s+\*\*([A-Za-z_][A-Za-z0-9_]*)\*\*\s*[:\-–]\s*([^\n]{5,100})",
        re.MULTILINE,
    ),
    re.compile(r"^\s*Returns?:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Raises?:\s*(.+)$", re.MULTILINE | re.IGNORECASE),
]
_MAX_OVERVIEW_CHARS = 320
_MAX_KEY_CONCEPTS = 6
_MAX_API_NOTES = 8


def _first_paragraph(text: str, max_chars: int) -> str:
    """Return the first non-empty paragraph, truncated to max_chars."""
    if not text:
        return ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para or para.startswith("#") or para.startswith("```"):
            continue
        # Skip if it looks like a parameter block
        if para.startswith("-") or para.startswith("*"):
            continue
        if len(para) > max_chars:
            # Cut at last sentence within 10% margin
            margin = max_chars // 10
            for sep in (". ", ".\n", "! ", "?\n"):
                idx = para.rfind(sep, 0, max_chars)
                if idx > max_chars - margin:
                    return para[: idx + 1].rstrip()
            return para[:max_chars].rstrip() + "…"
        return para
    return ""


def _key_concepts(text: str) -> list[str]:
    """Extract the first sentence of each H2/H3 section as a key concept."""
    concepts: list[str] = []
    seen: set[str] = set()

    for pattern in (_H2_PATTERN, _H3_PATTERN):
        for match in pattern.finditer(text):
            heading = match.group(1).strip()
            if not heading or heading.lower() in seen:
                continue
            seen.add(heading.lower())

            # Get the first sentence after the heading
            start = match.end()
            tail = text[start : start + 600]
            tail = tail.lstrip("\n")
            if tail.startswith("```"):
                continue
            first_sentence = ""
            for sep in (". ", ".\n", "! ", "?\n", "\n\n"):
                idx = tail.find(sep)
                if idx > 0 and idx < 400:
                    first_sentence = tail[: idx + 1].strip()
                    break
            if not first_sentence:
                first_sentence = tail[:120].strip().split("\n")[0]
            if first_sentence:
                concepts.append(f"**{heading}** — {first_sentence[:200]}")
                if len(concepts) >= _MAX_KEY_CONCEPTS:
                    return concepts
    return concepts


def _api_notes(text: str) -> list[str]:
    """Extract parameter/return/raise notes using regex patterns."""
    notes: list[str] = []
    seen: set[str] = set()
    for pattern in _API_PATTERNS:
        for match in pattern.finditer(text):
            groups = [g.strip() for g in match.groups() if g and g.strip()]
            if not groups:
                continue
            note = " ".join(groups).strip()
            note = re.sub(r"\s+", " ", note)
            if note in seen or len(note) < 8 or len(note) > 200:
                continue
            seen.add(note)
            notes.append(note)
            if len(notes) >= _MAX_API_NOTES:
                return notes
    return notes


def heuristic_summarize_page(text: str) -> list[dict[str, str]]:
    """Build info snippets for a page from raw text — no LLM call.

    Returns a list of {title, content} dicts ready to be stored in
    info_snippets_json. Format mirrors LLM output (markdown sections).
    """
    if not text or not text.strip():
        return []

    snippets: list[dict[str, str]] = []

    overview = _first_paragraph(text, _MAX_OVERVIEW_CHARS)
    if overview:
        snippets.append({"title": "Overview", "content": overview})

    concepts = _key_concepts(text)
    if concepts:
        snippets.append(
            {
                "title": "Key Concepts",
                "content": "\n".join(f"- {c}" for c in concepts),
            }
        )

    notes = _api_notes(text)
    if notes:
        snippets.append(
            {
                "title": "API Behavior Notes",
                "content": "\n".join(f"- {n}" for n in notes),
            }
        )

    return snippets


class HeuristicSummarizer:
    """Stateless wrapper around the heuristic summarizer functions.

    Mirrors the DocSummarizer interface minimally so the crawl/refresh
    path can call it the same way (summarize_source returns a result).
    Always free, always available, no model required.
    """

    def __init__(self) -> None:
        self.model = "heuristic"

    async def summarize_source(
        self,
        source_name: str,
        doc_store: Any,
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """Apply heuristic summarization to all pages of a source.

        Cache-first: skips pages that already have info_snippets_json.
        """
        return await heuristic_summarize_source(
            doc_store=doc_store,
            source_name=source_name,
            max_pages=max_pages,
        )


async def heuristic_summarize_source(
    doc_store: Any,
    source_name: str,
    max_pages: int | None = None,
) -> dict[str, int]:
    """Apply heuristic_summarize_page to all pages of a source.

    Cache-first: skips pages that already have info_snippets_json.
    Stores results in info_snippets_json with model="heuristic".

    Returns counts for observability: {pages_processed, pages_skipped}.
    """
    source = await doc_store.get_source(source_name)
    if not source:
        return {"pages_processed": 0, "pages_skipped": 0, "errors": 1}

    pages = await doc_store.list_pages(source["id"], limit=9999)
    if max_pages:
        pages = pages[:max_pages]

    processed = 0
    skipped = 0
    for page in pages:
        existing = await doc_store.get_info_snippets(page["id"])
        if existing and existing != "[]":
            skipped += 1
            continue

        content = page.get("content", "")
        snippets = heuristic_summarize_page(content)
        if snippets:
            import json

            await doc_store.set_info_snippets(
                page_id=page["id"],
                snippets_json=json.dumps(snippets, ensure_ascii=False),
                tokens_used=0,
                model="heuristic",
            )
        processed += 1

    return {
        "pages_processed": processed,
        "pages_skipped": skipped,
        "errors": 0,
    }
