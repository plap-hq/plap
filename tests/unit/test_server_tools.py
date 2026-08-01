from __future__ import annotations

from dataclasses import FrozenInstanceError, field, is_dataclass, replace

import pytest

from plap.llms.completions.chat import ChatMessage, ChatToolCall
from plap.plugins.easy.server_tools import ServerTool
from plap.responses.state import State


class ExampleTool(ServerTool):
    name: str = "example"
    parameters: dict[str, object] = field(default_factory=lambda: {"type": "object"})
    strict: bool = True
    description: str = "Example server tool."

    async def __call__(
        self,
        state: State,
        call: ChatToolCall,
    ) -> ChatMessage:
        _ = state
        return ChatMessage(role="tool", tool_call_id=call.id)


class SpecializedTool(ExampleTool):
    name: str = "specialized"


def test_server_tool_inheritance_builds_frozen_dataclasses() -> None:
    tool = ExampleTool()
    other = ExampleTool()

    assert is_dataclass(ExampleTool)
    assert tool.parameters == {"type": "object"}
    assert tool.parameters is not other.parameters
    with pytest.raises(FrozenInstanceError):
        tool.name = "changed"  # type: ignore[misc]


def test_server_tool_inheritance_applies_to_deeper_subclasses_and_replace() -> None:
    tool = SpecializedTool()
    rebound = replace(tool, name="specialized_2")

    assert is_dataclass(SpecializedTool)
    assert isinstance(rebound, SpecializedTool)
    assert rebound.name == "specialized_2"
