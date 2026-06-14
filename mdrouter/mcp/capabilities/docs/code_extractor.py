"""Code block extractor — extracts <pre><code> from HTML using stdlib only.

No BeautifulSoup, no lxml, no new dependencies. Uses Python's built-in
html.parser.HTMLParser with state tracking for headings and code blocks.

Output: list[CodeSnippet] with code_title, language, code, source_section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser


@dataclass
class CodeSnippet:
    """A single code example extracted from a documentation page.

    Fields match the Context7 API shape:
    - code_title: short label (e.g., "Basic usage example")
    - language: detected programming language
    - code: verbatim source text
    - source_section: heading path for context (e.g., "Tutorial > Routing")
    """

    code_title: str = ""
    language: str = ""
    code: str = ""
    source_section: str = ""


# Regex patterns for detecting language from <code> class attributes.
# These match common doc-site conventions (ReadTheDocs, MkDocs, Docusaurus, etc.)
_LANG_PATTERNS: list[tuple[str, str]] = [
    (r"language-(\w+)", "{0}"),  # language-python
    (r"lang-(\w+)", "{0}"),  # lang-python
    (r"hljs\s+(\w+)", "{0}"),  # hljs python
    (r"brush:\s*(\w+)", "{0}"),  # brush: python
    (r"code-(\w+)", "{0}"),  # code-python
]


def _detect_language(class_attr: str) -> str:
    """Extract language from a <code> class attribute string."""
    if not class_attr:
        return ""
    for pattern, _ in _LANG_PATTERNS:
        m = re.search(pattern, class_attr, re.IGNORECASE)
        if m:
            lang = m.group(1).lower()
            # Normalize common aliases
            aliases = {
                "js": "javascript",
                "ts": "typescript",
                "py": "python",
                "rb": "ruby",
                "sh": "bash",
                "zsh": "bash",
                "yml": "yaml",
            }
            return aliases.get(lang, lang)
    return ""


class CodeBlockExtractor(HTMLParser):
    """Extract <pre><code> blocks from HTML with language and context.

    Usage:
        extractor = CodeBlockExtractor()
        extractor.feed(html)
        snippets = extractor.snippets

    State machine tracks:
    - Current heading stack (h1-h6) for source_section context
    - Whether we're inside a <pre> element
    - Whether we're inside a <code> element
    - Accumulated text for headings and code
    """

    def __init__(self, max_snippets: int = 20) -> None:
        super().__init__()
        self.max_snippets = max_snippets
        self.snippets: list[CodeSnippet] = []
        self._headings: list[str] = []
        self._in_pre = False
        self._in_code = False
        self._current_code: list[str] = []
        self._current_lang = ""
        self._current_title = ""
        self._heading_buffer: list[str] = []
        self._in_heading = ""
        self._last_heading_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)

        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._in_heading = tag
            self._heading_buffer = []

        elif tag == "pre":
            self._in_pre = True
            self._current_code = []
            self._current_lang = ""
            # Get title from the last heading we saw
            self._current_title = self._last_heading_text

        elif tag == "code" and self._in_pre:
            self._in_code = True
            cls = attrs_dict.get("class", "")
            self._current_lang = _detect_language(cls)

    def handle_endtag(self, tag: str) -> None:
        if tag == self._in_heading:
            text = " ".join(self._heading_buffer).strip()
            if text:
                self._last_heading_text = text
                # Maintain heading stack (pop headings at or below this level)
                level = int(tag[1])
                self._headings = [
                    h for h in self._headings if self._heading_level(h) < level
                ]
                self._headings.append(f"{tag}>{text}")
            self._in_heading = ""

        elif tag == "code" and self._in_code:
            self._in_code = False

        elif tag == "pre":
            self._in_pre = False
            self._in_code = False
            code_text = "".join(self._current_code)
            if code_text.strip():
                section = " > ".join(h.split(">", 1)[1] for h in self._headings)
                if not self._current_title:
                    self._current_title = self._last_heading_text
                self.snippets.append(
                    CodeSnippet(
                        code_title=self._current_title or "",
                        language=self._current_lang,
                        code=code_text,
                        source_section=section,
                    )
                )
            self._current_code = []
            self._current_lang = ""
            self._current_title = ""

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading_buffer.append(data)
        elif self._in_code:
            self._current_code.append(data)
        # Outside code/headings: data is ignored (saves memory)

    @staticmethod
    def _heading_level(h: str) -> int:
        try:
            return int(h[1])
        except (IndexError, ValueError):
            return 0


def extract_code_blocks(html: str, max_snippets: int = 20) -> list[CodeSnippet]:
    """Extract code blocks from HTML. Convenience wrapper.

    Args:
        html: Raw HTML from a documentation page.
        max_snippets: Maximum number of code blocks to extract.

    Returns:
        List of CodeSnippet objects (may be empty).
    """
    extractor = CodeBlockExtractor(max_snippets=max_snippets)
    try:
        extractor.feed(html)
    except Exception:
        # HTMLParser is strict; malformed HTML can raise.
        # Return whatever we got so far.
        pass
    return extractor.snippets[:max_snippets]
