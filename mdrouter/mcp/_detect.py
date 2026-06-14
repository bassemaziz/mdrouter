"""Project root auto-detection for mdrouter-mcp.

Walks up from this module's location to find config/ or pyproject.toml.
Used when the binary is run from an arbitrary CWD (e.g., by Copilot).
"""

from __future__ import annotations

from pathlib import Path


def detect_project_root() -> Path | None:
    """Find the project root by walking up from this module's location.

    When installed via `pip install -e .`, the binary lives in
    .venv/bin/mdrouter-mcp and this module is in the project tree.
    We walk up looking for config/ or pyproject.toml.
    """
    try:
        here = Path(__file__).resolve().parent  # mdrouter/mcp/
        for _ in range(5):
            if (here / "config" / "mcp.json").exists():
                return here
            if (here / "pyproject.toml").exists():
                return here
            here = here.parent
    except Exception:
        pass
    return None
