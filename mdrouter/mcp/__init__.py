"""mdrouter MCP server — pluggable capabilities for AI coding agents.

Entry points:
- mdrouter-mcp CLI command (stdio transport)
- create_server() for programmatic use
"""

from __future__ import annotations

from mdrouter.mcp._detect import detect_project_root
from mdrouter.mcp._server import create_server
from mdrouter.mcp._cli import main

__all__ = ["create_server", "main", "detect_project_root"]
