from __future__ import annotations

from collections.abc import Callable

from plap.responses.models import ChatMessageSpan, TranscriptMessage, TranscriptToolCall


def compact_transcript(
    spans: tuple[ChatMessageSpan, ...],
    *,
    untrusted: bool = False,
) -> tuple[TranscriptMessage, ...]:
    compact: list[TranscriptMessage] = []
    pending_tool_calls: dict[str, TranscriptToolCall] = {}
    for span in spans:
        message = span.message
        if message.role == "tool":
            if message.tool_call_id is not None and message.tool_call_id in pending_tool_calls:
                current = pending_tool_calls[message.tool_call_id]
                pending_tool_calls[message.tool_call_id] = TranscriptToolCall(
                    _id=current._id,
                    name=current.name,
                    arguments=current.arguments,
                    output=message.content_text() or "",
                )
                if compact:
                    compact[-1] = TranscriptMessage(
                        role=compact[-1].role,
                        content=compact[-1].content,
                        tool_calls=tuple(pending_tool_calls.get(call._id or "", call) for call in compact[-1].tool_calls),
                    )
            continue

        item = message.to_transcript_message(untrusted=untrusted)
        compact.append(item)
        pending_tool_calls = {call._id or "": call for call in item.tool_calls if call._id is not None}
    return tuple(item.without_ids() for item in compact)


def truncate_transcript(
    transcript: tuple[TranscriptMessage, ...],
    *,
    measure: Callable[[tuple[TranscriptMessage, ...]], int],
    max_tokens: int,
) -> tuple[TranscriptMessage, ...]:
    if max_tokens < 0:
        raise ValueError("max transcript tokens must be non-negative")
    if measure(transcript) <= max_tokens:
        return transcript

    variable_indexes = [
        index for index, message in enumerate(transcript) if message.role not in {"developer", "system"}
    ]
    if not variable_indexes:
        return transcript

    best: tuple[TranscriptMessage, ...] | None = None
    low = 0
    high = len(variable_indexes)
    while low <= high:
        dropped = (low + high) // 2
        dropped_indexes = set(variable_indexes[:dropped])
        candidate = tuple(message for index, message in enumerate(transcript) if index not in dropped_indexes)
        if measure(candidate) <= max_tokens:
            best = candidate
            high = dropped - 1
        else:
            low = dropped + 1
    if best is not None:
        return best
    return tuple(message for message in transcript if message.role in {"developer", "system"})


__all__ = [
    "ChatMessageSpan",
    "TranscriptMessage",
    "compact_transcript",
    "truncate_transcript",
]
