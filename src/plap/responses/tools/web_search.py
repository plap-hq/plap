from __future__ import annotations

from plap.responses.contracts import FunctionTool, WebSearchTool
from plap.responses.tools.policy import ToolPolicy

WEB_SEARCH_TOOL_NAME = "web_search"


def web_search_tool(_: WebSearchTool) -> FunctionTool:
    return FunctionTool(
        description="Search the web for current or external information.",
        name=WEB_SEARCH_TOOL_NAME,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        strict=True,
        type="function",
    )


def web_search_policy() -> ToolPolicy:
    return ToolPolicy(
        name=WEB_SEARCH_TOOL_NAME,
        source="server",
        effect_class="safe",
    )
