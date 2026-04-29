from __future__ import annotations

from plap.responses.contracts import FunctionTool
from plap.responses.tools.policy import ToolPolicy

COMPRESS_TOOL_NAME = "compress"

COMPRESS_DEVELOPER_PROMPT = """The `compress` tool replaces selected ranges of
earlier conversation messages with focused summaries you write. The system
injects citations to identify visible conversation context: `[~N]` labels one
message, and `[~A_B]` labels a summarized range from message A through message B.
Use these citation strings exactly as shown for each range's `start` and `end`;
the range is inclusive. Do not include citations in summaries. Write the
replacement summaries yourself. Use compression when prior context is long,
repetitive, or no longer needs to remain in full detail. Keep important facts,
decisions, constraints, tool results, and open threads easy to use. Do not
mention citations, the `compress` tool, compression, or compaction to the user."""


def compress_tool() -> FunctionTool:
    return FunctionTool(
        description=(
            "Replace earlier cited conversation ranges with focused summaries so "
            "you can continue with less context. Use this when prior context is "
            "long, repetitive, or no longer needs to remain in full detail. Never "
            "call this tool in parallel with any other tool. Use {\"ranges\": []} "
            "if no useful safe compression is possible right now."
        ),
        name=COMPRESS_TOOL_NAME,
        parameters={
            "type": "object",
            "properties": {
                "ranges": {
                    "type": "array",
                    "description": (
                        "Inclusive visible citation ranges to replace. The start "
                        "and end values must be citations exactly as shown in the "
                        "conversation context, such as [~0] or [~0_7]. Use an "
                        "empty array when no useful safe compression is possible."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "string",
                                "description": "Citation of the first visible span.",
                            },
                            "end": {
                                "type": "string",
                                "description": "Citation of the last visible span.",
                            },
                            "summary": {
                                "type": "string",
                                "description": (
                                    "Replacement summary for the selected range."
                                ),
                            },
                        },
                        "required": ["start", "end", "summary"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ranges"],
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
