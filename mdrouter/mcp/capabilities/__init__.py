"""Capability registry — maps config names to capability classes.

To add a new capability:
1. Create a module under capabilities/<name>/
2. Implement the Capability ABC
3. Add an entry here: CAPABILITY_MAP["name"] = "module.path:ClassName"
4. Add "name" to enabled_capabilities in config/mcp.json

That's it — no server code changes needed.
"""

CAPABILITY_MAP: dict[str, str] = {
    "docs": "mdrouter.mcp.capabilities.docs:DocsCapability",
}
