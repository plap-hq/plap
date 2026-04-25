from __future__ import annotations

import blake3
import msgspec

from plap.responses.contracts import FunctionTool
from plap.responses.tools.types import ToolSignature


def normalize_function_tool(tool: FunctionTool) -> dict[str, object]:
    return {
        "description": tool.description,
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
        "type": "function",
    }


def function_tool_signature(tool: FunctionTool) -> ToolSignature:
    signature = normalize_function_tool(tool)
    return ToolSignature(
        signature_hash=blake3.blake3(
            msgspec.json.encode(signature, order="deterministic")
        ).digest(),
        signature=signature,
    )


def signature_hash_hex(signature_hash: bytes) -> str:
    return signature_hash.hex()
