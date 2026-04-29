from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from fastmcp import Client

from plap.responses.contracts import FunctionTool


@runtime_checkable
class IMCPToolProvider(Protocol):
    async def tools(self) -> tuple[FunctionTool, ...]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class MCPToolProvider(IMCPToolProvider):
    def __init__(
        self,
        transport: object,
        *,
        tool_names: Sequence[str] | None = None,
    ) -> None:
        self._transport = transport
        self._tool_names = frozenset(tool_names or ())

    async def tools(self) -> tuple[FunctionTool, ...]:
        async with Client(self._transport) as client:
            tools = await client.list_tools()
        return tuple(
            _mcp_tool_to_function_tool(tool)
            for tool in tools
            if not self._tool_names or tool.name in self._tool_names
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if self._tool_names and name not in self._tool_names:
            raise ValueError(f"MCP tool is not enabled: {name}")
        async with Client(self._transport) as client:
            result = await client.call_tool(name, arguments)
        if result.is_error:
            raise RuntimeError(_mcp_call_result_text(result) or "MCP tool call failed")
        return _mcp_call_result_text(result)


def _mcp_tool_to_function_tool(tool: Any) -> FunctionTool:
    return FunctionTool(
        description=getattr(tool, "description", None) or "",
        name=tool.name,
        parameters=_json_mapping(getattr(tool, "inputSchema", None)),
        strict=True,
        type="function",
    )


def _json_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "object"}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    return dict(value)


def _mcp_call_result_text(result: Any) -> str:
    if result.data is not None:
        return _json_text(result.data)
    if result.structured_content is not None:
        return _json_text(result.structured_content)
    blocks = [_content_block_text(block) for block in result.content]
    return "\n".join(block for block in blocks if block)


def _content_block_text(block: Any) -> str:
    text = getattr(block, "text", None)
    if isinstance(text, str):
        return text
    if hasattr(block, "model_dump"):
        return _json_text(block.model_dump(mode="json", exclude_none=True))
    return str(block)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
