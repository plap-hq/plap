from __future__ import annotations

from dataclasses import dataclass

from plap.responses.models import (
    ChatMessageSpan,
    CompactionPayload,
    TranscriptMessage,
    TranscriptToolCall,
)


@dataclass(frozen=True, slots=True)
class _ExpansionCandidate:
    index: int
    summary_fidelity: int
    token_delta: int
    start: int
    end: int


def render_main_transcript(
    compaction: CompactionPayload | None,
    stable_rows: tuple[ChatMessageSpan, ...],
    *,
    transcript_token_budget: int,
) -> tuple[ChatMessageSpan, ...]:
    root_rows = (
        ()
        if compaction is None
        else render_budgeted_spans(
            compaction.active,
            token_budget=transcript_token_budget,
        )
    )
    return (*root_rows, *stable_rows)


def render_budgeted_spans(
    spans: tuple[ChatMessageSpan, ...],
    *,
    token_budget: int,
) -> tuple[ChatMessageSpan, ...]:
    if token_budget < 0:
        raise ValueError("token budget must be non-negative")
    current_tokens = sum(span.token_count for span in spans)
    rendered = list(spans)
    while candidate := _best_expansion_candidate(rendered, current_tokens=current_tokens, token_budget=token_budget):
        span = rendered[candidate.index]
        rendered[candidate.index : candidate.index + 1] = span.children
        current_tokens += candidate.token_delta
    return tuple(rendered)


def _best_expansion_candidate(
    spans: list[ChatMessageSpan],
    *,
    current_tokens: int,
    token_budget: int,
) -> _ExpansionCandidate | None:
    candidates: list[_ExpansionCandidate] = []
    for index, span in enumerate(spans):
        if not span.children:
            continue
        token_delta = span.children_token_count - span.token_count
        if current_tokens + token_delta > token_budget:
            continue
        candidates.append(
            _ExpansionCandidate(
                index=index,
                summary_fidelity=span.summary_fidelity if span.summary_fidelity is not None else 3,
                token_delta=token_delta,
                start=span.start,
                end=span.end,
            )
        )
    return min(
        candidates,
        key=lambda candidate: (
            candidate.summary_fidelity,
            candidate.token_delta,
            -candidate.end,
            -candidate.start,
            candidate.index,
        ),
        default=None,
    )


def compact_transcript(spans: tuple[ChatMessageSpan, ...]) -> tuple[TranscriptMessage, ...]:
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
                        tool_calls=tuple(
                            pending_tool_calls.get(call._id or "", call)
                            for call in compact[-1].tool_calls
                        ),
                    )
            continue

        item = message.to_transcript_message()
        compact.append(item)
        pending_tool_calls = {call._id or "": call for call in item.tool_calls if call._id is not None}
    return tuple(item.without_ids() for item in compact)
