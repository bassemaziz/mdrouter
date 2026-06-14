"""Response builder — formats Context7-shaped markdown from code + info snippets.

Produces agent-friendly markdown with:
- 60% token budget for code snippets (preserved verbatim from HTML)
- 40% token budget for info snippets (LLM-generated prose)
- Truncation at snippet boundaries (never mid-code or mid-sentence)
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field


@dataclass
class InfoSnippet:
    """An LLM-generated prose summary section."""

    title: str = ""
    content: str = ""


@dataclass
class CodeSnippet:
    """A verbatim code block extracted from HTML."""

    code_title: str = ""
    language: str = ""
    code: str = ""
    source_section: str = ""


@dataclass
class SearchResult:
    """A single search result with code + info snippets."""

    title: str
    url: str
    source: str
    code_snippets: list[CodeSnippet] = field(default_factory=list)
    info_snippets: list[InfoSnippet] = field(default_factory=list)
    relevance: float = 0.0


def _estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 characters."""
    return len(text) // 4


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens at a sentence boundary."""
    if max_tokens <= 0:
        return text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # Try to cut at sentence boundary within 10% margin
    margin = max_chars // 10
    end = max_chars
    truncated = text[: end + margin]
    # Find the last sentence-ending punctuation
    for sep in ("\n\n", ". ", ".\n", ".  ", "!\n", "?\n"):
        idx = truncated.rfind(sep, 0, end)
        if idx > end - margin:
            return truncated[: idx + len(sep)].rstrip()
    return truncated[:max_chars].rstrip() + "..."


def build_search_response(
    results: list[SearchResult],
    max_tokens: int = 1000,
    total_count: int = 0,
    offset: int = 0,
    snippet_type: str = "all",
) -> str:
    """Build a markdown search response, Context7-style.

    Args:
        results: Search results with code + info snippets.
        max_tokens: Per-result token budget (default 1000).
        total_count: Total matching results (for pagination).
        offset: Current page offset.
        snippet_type: "all", "code", or "info".

    Returns:
        Markdown text with --- separated results.
    """
    parts: list[str] = []

    for i, r in enumerate(results):
        code_budget = (
            int(max_tokens * 0.6)
            if snippet_type == "all"
            else (max_tokens if snippet_type == "code" else 0)
        )
        info_budget = (
            int(max_tokens * 0.4)
            if snippet_type == "all"
            else (max_tokens if snippet_type == "info" else 0)
        )

        parts.append(f"## {r.title or 'Untitled'}")
        parts.append(
            f"**Source:** {r.source} | **URL:** {r.url} | **Relevance:** {r.relevance:.1f}"
        )
        parts.append("")

        # Code snippets first (higher value for agents)
        if code_budget > 0 and snippet_type in ("all", "code"):
            code = _build_code_section(r.code_snippets, code_budget)
            if code:
                parts.append(code)

        # Info snippets
        if info_budget > 0 and snippet_type in ("all", "info"):
            info = _build_info_section(r.info_snippets, info_budget)
            if info:
                parts.append(info)

        parts.append("---")
        parts.append("")

    # Pagination footer
    if total_count > 0:
        shown = offset + len(results)
        parts.append(
            f"*Showing {offset + 1}–{min(shown, total_count)} of {total_count} results.*"
        )

    return "\n".join(parts).rstrip()


def build_page_response(
    title: str,
    url: str,
    source: str,
    code_snippets: list[CodeSnippet],
    info_snippets: list[InfoSnippet],
    max_tokens: int = 1000,
    include_code: bool = True,
) -> str:
    """Build a markdown page response, Context7-style.

    Args:
        title: Page title.
        url: Page URL.
        source: Source name.
        code_snippets: Extracted code blocks.
        info_snippets: LLM-generated prose.
        max_tokens: Token budget (default 1000).
        include_code: Whether to include code blocks.

    Returns:
        Markdown text.
    """
    code_budget = int(max_tokens * 0.6) if include_code else 0
    info_budget = int(max_tokens * 0.4) if include_code else max_tokens

    parts = [
        f"# {title or 'Untitled'}",
        f"**Source:** {source} | **URL:** {url}",
        "",
    ]

    if include_code and code_budget > 0:
        code = _build_code_section(code_snippets, code_budget)
        if code:
            parts.append(code)

    info = _build_info_section(info_snippets, info_budget)
    if info:
        parts.append(info)

    return "\n".join(parts).rstrip()


def build_snippets_response(
    results: list[SearchResult],
    max_tokens: int = 1000,
    language: str | None = None,
) -> str:
    """Build a code-only markdown response.

    Args:
        results: Search results with code snippets.
        max_tokens: Total token budget.
        language: Optional language filter.

    Returns:
        Markdown text with language-tagged code fences.
    """
    parts: list[str] = []
    tokens_used = 0

    for r in results:
        parts.append(f"## {r.title or 'Untitled'}")
        parts.append(f"**Source:** {r.source} | **URL:** {r.url}")
        parts.append("")

        for cs in r.code_snippets:
            if language and cs.language != language:
                continue
            remaining = max_tokens - tokens_used
            if remaining <= 0:
                parts.append("*…truncated (token limit)*")
                return "\n".join(parts)

            code_block = _format_code_block(cs, remaining)
            tokens_used += _estimate_tokens(code_block)
            parts.append(code_block)

        parts.append("---")
        parts.append("")

    return "\n".join(parts).rstrip()


def _build_code_section(snippets: list[CodeSnippet], budget: int) -> str:
    """Build the '### Code Examples' section within a token budget."""
    if not snippets:
        return ""
    parts = ["### Code Examples", ""]
    tokens_used = 0

    for cs in snippets:
        remaining = budget - tokens_used
        if remaining < 50:  # Too little space for a meaningful code block
            parts.append("*…more code examples (token limit)*")
            break
        block = _format_code_block(cs, remaining)
        tokens_used += _estimate_tokens(block)
        parts.append(block)
        parts.append("")

    return "\n".join(parts)


def _build_info_section(snippets: list[InfoSnippet], budget: int) -> str:
    """Build the info/prose section within a token budget."""
    if not snippets:
        return ""
    tokens_used = 0

    for s in snippets:
        remaining = budget - tokens_used
        if remaining < 30:
            break
        content = _truncate_to_tokens(s.content, remaining)
        tokens_used += _estimate_tokens(content)
        return content  # Info snippets come as individual sections

    return ""


def _format_code_block(cs: CodeSnippet, max_tokens: int) -> str:
    """Format a single code block with language tag and context."""
    code = _truncate_to_tokens(cs.code, max(max_tokens, 100))
    lang = cs.language
    title = cs.code_title
    section = cs.source_section

    lines = []
    if title and section:
        lines.append(f"**{title}** ({section})")
    elif title:
        lines.append(f"**{title}**")

    if lang:
        lines.append(f"```{lang}")
    else:
        lines.append("```")
    lines.append(code)
    lines.append("```")

    return "\n".join(lines)


def parse_snippets_from_json(
    json_str: str,
) -> tuple[list[CodeSnippet], list[InfoSnippet]]:
    """Parse code_snippets_json and info_snippets_json into typed objects.

    Args:
        json_str: A JSON array string of snippet dicts.

    Returns:
        Tuple of (code_snippets, info_snippets).
    """
    code: list[CodeSnippet] = []
    info: list[InfoSnippet] = []

    try:
        items = _json.loads(json_str) if isinstance(json_str, str) else json_str
    except (_json.JSONDecodeError, TypeError):
        return code, info

    for item in items or []:
        if not isinstance(item, dict):
            continue
        if "language" in item or "code" in item:
            code.append(
                CodeSnippet(
                    code_title=item.get("code_title", ""),
                    language=item.get("language", ""),
                    code=item.get("code", ""),
                    source_section=item.get("source_section", ""),
                )
            )
        else:
            info.append(
                InfoSnippet(
                    title=item.get("title", ""),
                    content=item.get("content", ""),
                )
            )

    return code, info


def parse_search_result(row: dict) -> SearchResult:
    """Parse a database row into a SearchResult with typed snippets.

    Args:
        row: Dict from DocStore.search_combined with code_snippets_json
             and info_snippets_json keys.

    Returns:
        SearchResult with typed code_snippets and info_snippets.
    """
    code_snippets, info_snippets = parse_snippets_from_json(
        row.get("code_snippets_json", "[]")
    )
    # Also try info from info_snippets_json
    _, info2 = parse_snippets_from_json(row.get("info_snippets_json", "[]"))
    if not info_snippets:
        info_snippets = info2

    return SearchResult(
        title=row.get("title", "Untitled"),
        url=row.get("url", ""),
        source=row.get("source_name", ""),
        code_snippets=code_snippets,
        info_snippets=info_snippets,
        relevance=float(row.get("_fts_rank", 0.0)),
    )
