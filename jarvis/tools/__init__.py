"""
Jarvis' tools. Importing this package registers every tool in `registry`.

To add a tool module: write it with the @tool decorator, then import it below.
"""

from jarvis.tools.registry import ToolResult, registry, tool  # noqa: F401

# Importing a module is what registers its tools (the @tool decorators run on import)
from jarvis.tools import basic  # noqa: F401
