"""Server-side tool declarations.

Web search and web fetch run on Anthropic's infrastructure - they are declared
here and the results come back in the same response. There is nothing to
execute locally, which is why they have no handler function.
"""

from __future__ import annotations

from typing import Any

# The _20260209 variants add dynamic filtering: results are filtered before
# they reach the context window. Do not also declare code_execution alongside
# them - it creates a second execution environment and confuses the model.
WEB_TOOLS: list[dict[str, Any]] = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 8},
]

SERVER_TOOL_NAMES = {"web_search", "web_fetch"}
