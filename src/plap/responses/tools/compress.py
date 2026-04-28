from __future__ import annotations

from plap.responses.contracts import FunctionTool
from plap.responses.tools.policy import ToolPolicy

COMPRESS_TOOL_NAME = "compress"


def compress_tool() -> FunctionTool:
    return FunctionTool(
        description=(
            "Compact the current conversation context. Call this alone when context "
            "should be summarized before continuing. The call is internal and does "
            "not produce user-visible function call output."
        ),
        name=COMPRESS_TOOL_NAME,
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why compaction is needed now.",
                }
            },
            "additionalProperties": False,
        },
        strict=True,
        type="function",
    )


def compress_policy() -> ToolPolicy:
    return ToolPolicy(
        name=COMPRESS_TOOL_NAME,
        source="server",
        effect_class="safe",
    )
