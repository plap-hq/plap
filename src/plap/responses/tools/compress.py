from __future__ import annotations

from plap.responses.contracts import FunctionTool
from plap.responses.tools.policy import ToolPolicy

COMPRESS_TOOL_NAME = "compress"
DUPLICATE_TOOL_OUTPUT_TOMBSTONE = "This tool output was omitted; a later identical call retains the full result."

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

For each summary, set `summary_fidelity` using this anchored rubric: 5 =
reliable working replacement where expansion is unlikely to change future work
except for exact wording or minor detail; 4 = solid summary that preserves the
main reusable information but may need expansion if exactness matters; 3 =
usable gist with useful details missing; 2 = lossy orientation; 1 = minimal
breadcrumb that should be expanded before relying on it substantively.

If `prune_duplicate_tool_calls` is true (the default), identical tool calls across
the full available main context are deduplicated by tool name and arguments.
Older identical tool outputs may be replaced with a tombstone while the latest
identical call retains the full result. Set this to false only when exact repeated
tool-call history matters, for example while tracking nondeterminism.

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
            "Replace visible cited conversation ranges with focused summaries. Ranges "
            "are inclusive, non-overlapping, and must use citations exactly as shown. "
            'Never call this tool in parallel with any other tool. Use {"ranges": []} '
            "only when no useful safe compression is possible."
        ),
        name=COMPRESS_TOOL_NAME,
        parameters={
            "type": "object",
            "properties": {
                "prune_duplicate_tool_calls": {
                    "type": "boolean",
                    "description": (
                        "Whether to deduplicate identical tool calls across the full available main context. "
                        "When true, older duplicate tool outputs may be omitted while the latest identical call "
                        "retains the full result. Defaults to true."
                    ),
                },
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
                            "summary_fidelity": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5,
                                "description": "Anchored 1-5 fidelity score for how well the summary can stand in for the selected range.",
                            },
                        },
                        "required": ["start", "end", "summary", "summary_fidelity"],
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
