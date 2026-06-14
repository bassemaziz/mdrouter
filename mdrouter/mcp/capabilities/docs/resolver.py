"""Library doc URL resolver — resolves library names to documentation URLs.

Uses heuristics only — no external API dependencies:
1. Exact URL provided → use directly
2. Known mappings dict (~50 popular libraries)
3. Pattern matching: readthedocs, docs.rs, pkg.go.dev
4. Optional Context7 API fallback (requires CONTEXT7_API_KEY env var)
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger("mdrouter.mcp.resolver")

# Known documentation URLs for popular libraries.
# Format: canonical name → (base_url, version_url_template)
# version_url_template may use {version} placeholder.
_KNOWN_LIBS: dict[str, str] = {
    # Python
    "fastapi": "https://fastapi.tiangolo.com/",
    "pydantic": "https://docs.pydantic.dev/latest/",
    "sqlalchemy": "https://docs.sqlalchemy.org/en/latest/",
    "django": "https://docs.djangoproject.com/en/stable/",
    "flask": "https://flask.palletsprojects.com/en/stable/",
    "httpx": "https://www.python-httpx.org/",
    "aiohttp": "https://docs.aiohttp.org/en/stable/",
    "celery": "https://docs.celeryq.dev/en/stable/",
    "pytest": "https://docs.pytest.org/en/stable/",
    "sphinx": "https://www.sphinx-doc.org/en/master/",
    "numpy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/docs/",
    "scipy": "https://docs.scipy.org/doc/scipy/",
    "matplotlib": "https://matplotlib.org/stable/",
    "plotly": "https://plotly.com/python/",
    "boto3": "https://boto3.amazonaws.com/v1/documentation/api/latest/",
    "redis-py": "https://redis-py.readthedocs.io/en/stable/",
    "redis": "https://redis.io/docs/latest/",
    "uvicorn": "https://www.uvicorn.org/",
    "websockets": "https://websockets.readthedocs.io/en/stable/",
    "rich": "https://rich.readthedocs.io/en/stable/",
    "click": "https://click.palletsprojects.com/en/stable/",
    "typer": "https://typer.tiangolo.com/",
    # JavaScript / TypeScript
    "react": "https://react.dev/reference/react",
    "next.js": "https://nextjs.org/docs",
    "vue": "https://vuejs.org/guide/introduction.html",
    "express": "https://expressjs.com/",
    "prisma": "https://www.prisma.io/docs",
    "tailwindcss": "https://tailwindcss.com/docs",
    "typescript": "https://www.typescriptlang.org/docs/",
    "node.js": "https://nodejs.org/docs/latest/api/",
    "svelte": "https://svelte.dev/docs",
    "astro": "https://docs.astro.build/",
    "nuxt": "https://nuxt.com/docs",
    "vite": "https://vitejs.dev/guide/",
    # Rust
    "serde": "https://docs.rs/serde/latest/serde/",
    "tokio": "https://docs.rs/tokio/latest/tokio/",
    "actix-web": "https://docs.rs/actix-web/latest/actix_web/",
    "clap": "https://docs.rs/clap/latest/clap/",
    # Go
    "gin": "https://pkg.go.dev/github.com/gin-gonic/gin",
    "echo": "https://pkg.go.dev/github.com/labstack/echo/v4",
    "fiber": "https://docs.gofiber.io/",
    # Infrastructure
    "docker": "https://docs.docker.com/",
    "kubernetes": "https://kubernetes.io/docs/",
    "nginx": "https://nginx.org/en/docs/",
    "postgresql": "https://www.postgresql.org/docs/current/",
    "mongodb": "https://www.mongodb.com/docs/manual/",
    # Tools
    "git": "https://git-scm.com/docs",
    "eslint": "https://eslint.org/docs/latest/",
    "prettier": "https://prettier.io/docs/en/",
    "webpack": "https://webpack.js.org/concepts/",
    "rollup": "https://rollupjs.org/",
    "esbuild": "https://esbuild.github.io/",
}


def resolve_library_url(
    name: str,
    version: str | None = None,
) -> dict | None:
    """Resolve a library name to a documentation URL.

    Args:
        name: Library name or a URL. E.g. "fastapi", "https://docs.python.org/3/"
        version: Optional version to pin. E.g. "0.115.0"

    Returns:
        {"library": str, "doc_url": str, "suggested_source_name": str,
         "version": str | None, "method": str} or None if not found.
    """
    # Check if it's already a URL
    parsed = urlparse(name)
    if parsed.scheme in ("http", "https") and parsed.netloc:
        return {
            "library": parsed.netloc.replace("www.", ""),
            "doc_url": name.rstrip("/"),
            "suggested_source_name": parsed.netloc.replace("www.", "").split(".")[0],
            "version": version,
            "method": "url_provided",
        }

    # Normalize: lowercase, strip extras
    clean = name.lower().strip().rstrip("/")

    # Check known mappings
    if clean in _KNOWN_LIBS:
        url = _KNOWN_LIBS[clean]
        if version:
            # Try to append version for common patterns
            url = _append_version(url, clean, version)
        return {
            "library": name,
            "doc_url": url,
            "suggested_source_name": clean.replace(".", "-").replace(" ", "-"),
            "version": version,
            "method": "known_mapping",
        }

    # Pattern: try readthedocs
    rtfd = f"https://{clean}.readthedocs.io/en/latest/"
    if _check_headable(rtfd):
        return {
            "library": name,
            "doc_url": rtfd,
            "suggested_source_name": clean,
            "version": version,
            "method": "readthedocs_pattern",
        }

    # Pattern: try docs.rs (Rust crates)
    if _check_headable(f"https://docs.rs/{clean}/"):
        return {
            "library": name,
            "doc_url": f"https://docs.rs/{clean}/latest/",
            "suggested_source_name": clean,
            "version": version,
            "method": "docsrs_pattern",
        }

    # Pattern: try pkg.go.dev (Go modules)
    godev = f"https://pkg.go.dev/{clean}"
    if _check_headable(godev):
        return {
            "library": name,
            "doc_url": godev,
            "suggested_source_name": clean.replace("/", "-"),
            "version": version,
            "method": "pkg_go_dev_pattern",
        }

    return None


async def resolve_library_async(
    name: str,
    version: str | None = None,
) -> dict | None:
    """Async wrapper that tries local resolution first, then Context7 API.

    Requires CONTEXT7_API_KEY env var for the API fallback.
    """
    # Try local resolution first
    result = resolve_library_url(name, version)
    if result:
        return result

    # Context7 API fallback
    api_key = os.getenv("CONTEXT7_API_KEY", "")
    if not api_key:
        return None

    try:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://context7.com/api/v2/libs/search",
                params={"libraryName": name},
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    best = results[0]
                    return {
                        "library": best.get("title", name),
                        "doc_url": f"https://context7.com{best.get('id', '')}",
                        "suggested_source_name": name.lower().replace("/", "-"),
                        "version": version,
                        "method": "context7_api",
                    }
    except Exception:
        logger.debug("Context7 API fallback failed for '%s'", name, exc_info=True)

    return None


def _append_version(url: str, name: str, version: str) -> str:
    """Append version to a URL, using common patterns."""
    url = url.rstrip("/")
    # readthedocs uses /en/{version}/
    if "readthedocs.io" in url:
        parts = url.split("/")
        # Replace 'latest' or 'stable' with version
        for i, p in enumerate(parts):
            if p in ("latest", "stable", "master", "main"):
                parts[i] = version
                return "/".join(parts)
        return f"{url}/en/{version}"
    # Generic: just append
    return f"{url}/{version}"


# Simple connectivity check cache (in-memory, short-lived)
_head_cache: dict[str, bool] = {}


def _check_headable(url: str) -> bool:
    """Check if a URL is reachable (synchronous, for quick discovery)."""
    import urllib.request

    if url in _head_cache:
        return _head_cache[url]

    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "mdrouter-docbot/1.0")
        with urllib.request.urlopen(req, timeout=3) as resp:
            ok = resp.status < 400
            _head_cache[url] = ok
            return ok
    except Exception:
        _head_cache[url] = False
        return False
