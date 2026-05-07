from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import msgspec
from fastmcp import Client

from plap.responses.contracts import FunctionTool
from plap.settings import MCPToolConfig


@runtime_checkable
class IMCPToolProvider(Protocol):
    name: str
    tool_configs: Mapping[str, MCPToolConfig]

    async def tools(self) -> tuple[FunctionTool, ...]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any], *, request_tool: object | None = None) -> str: ...


@runtime_checkable
class IServerToolExecutor(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


@dataclass(frozen=True, slots=True)
class MCPToolExecutor(IServerToolExecutor):
    provider: IMCPToolProvider
    request_tool: object | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return await self.provider.call_tool(name, arguments, request_tool=self.request_tool)


class MCPToolProvider(IMCPToolProvider):
    def __init__(
        self,
        name: str,
        transport: object,
        *,
        tools: Mapping[str, MCPToolConfig] | None = None,
    ) -> None:
        self.name = name
        self._transport = transport
        self.tool_configs = dict(tools or {})

    async def tools(self) -> tuple[FunctionTool, ...]:
        if not self.tool_configs:
            return ()
        async with Client(self._transport) as client:
            tools = await client.list_tools()
        return tuple(_mcp_tool_to_function_tool(tool) for tool in tools if tool.name in self.tool_configs)

    async def call_tool(self, name: str, arguments: dict[str, Any], *, request_tool: object | None = None) -> str:
        _ = request_tool
        if name not in self.tool_configs:
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
    return msgspec.json.encode(value).decode()
