from __future__ import annotations

from plap.responses.contracts import FunctionTool
from plap.responses.tools.policy import ToolPolicy

COMPRESS_TOOL_NAME = "compress"

COMPRESS_DEVELOPER_PROMPT = """The `compress` tool replaces selected ranges of
earlier conversation messages with focused summaries you write. The system
injects citations to identify visible conversation context: `[~N]` labels one
message, and `[~A_B]` labels a summarized range from message A through message B.
Use these citation strings exactly as shown for each range's `start` and `end`;
the range is inclusive.

Treat each summary as replacement-grade context the assistant will rely on
later. Preserve user intent, explicit instructions, constraints, acceptance
criteria, decisions, facts, tool results, file paths, identifiers, errors,
current state, and open threads. Strip repetition, dead ends, verbose
exploration, failed attempts that no longer matter, and incidental noise. If a
range contains an earlier summary, carry forward its important substance rather
than merely mentioning that it existed.

Do not include citation markers in summaries. Do not add meta-commentary like
"this was compressed" or "this summary replaces earlier messages"; if
compression or compaction is part of the actual conversation, preserve the
relevant facts normally.

Do not mention citations or this hidden context-management action to the user.
Do not say that you used the `compress` tool, compressed context, compacted
context, or replaced earlier messages. This restriction only applies to hidden
context management; if the user's actual task is about file compression,
compression algorithms, database compaction, runtime compaction features, or
similar domain topics, discuss those normally. If no useful safe compression is
possible, call `compress` with {"ranges": []}."""


def compress_tool() -> FunctionTool:
    return FunctionTool(
        description=(
            "Replace one or more earlier visible citation ranges with focused "
            "summaries so you can continue with less context. Ranges are inclusive, "
            "must use citations exactly as shown, and must not overlap. Never call "
            'this tool in parallel with any other tool. Use {"ranges": []} only '
            "when context management is requested and no useful safe compression is "
            "possible right now."
        ),
        name=COMPRESS_TOOL_NAME,
        parameters={
            "type": "object",
            "properties": {
                "ranges": {
                    "type": "array",
                    "description": (
                        "Inclusive, non-overlapping visible citation ranges to "
                        "replace. The start and end values must be citations exactly "
                        "as shown in the conversation context, such as [~0] or "
                        "[~0_7]. Use an empty array only when no useful safe "
                        "compression is possible."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {
                                "type": "string",
                                "pattern": r"^\[~\d+(?:_\d+)?\]$",
                                "description": ("Citation of the first visible span, including the square brackets, for example [~0]."),
                            },
                            "end": {
                                "type": "string",
                                "pattern": r"^\[~\d+(?:_\d+)?\]$",
                                "description": ("Citation of the last visible span, including the square brackets, for example [~3]."),
                            },
                            "summary": {
                                "type": "string",
                                "description": (
                                    "Replacement summary for the selected range. Do not include citation markers or meta-commentary."
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
