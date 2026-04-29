from __future__ import annotations

from plap.responses.tools.policy import ToolPolicy


def web_search_policy(name: str) -> ToolPolicy:
    return ToolPolicy(
        name=name,
        source="server",
        effect_class="safe",
    )
