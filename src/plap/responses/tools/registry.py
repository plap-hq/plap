from __future__ import annotations

from plap.responses.tools.types import ToolPolicy

SERVER_TOOL_POLICIES: dict[str, ToolPolicy] = {
    "web_search": ToolPolicy(
        name="web_search",
        source="server",
        effect_class="safe",
    )
}


def get_server_tool_policy(name: str) -> ToolPolicy | None:
    return SERVER_TOOL_POLICIES.get(name)
