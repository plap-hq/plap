from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from plap.responses.models import ChatMessageSpan, TranscriptMessage, TranscriptToolCall


@dataclass(frozen=True, slots=True)
class _ExpansionCandidate:
    index: int
    summary_fidelity: int
    token_delta: int
    start: int
    end: int


def _candidate_sort_key(candidate: _ExpansionCandidate) -> tuple[int, int, int, int, int]:
    return (
        candidate.summary_fidelity,
        candidate.token_delta,
        -candidate.end,
        -candidate.start,
        candidate.index,
    )


def _expansion_candidates(
    spans: list[ChatMessageSpan],
    *,
    current_tokens: int,
    token_budget: int,
) -> list[_ExpansionCandidate]:
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
    return candidates


def _visible_tool_call_ids(span: ChatMessageSpan) -> set[str]:
    if not span.message.is_assistant():
        return set()
    return {tool_call.id for tool_call in span.message.tool_calls}


def _visible_tool_output_ids(span: ChatMessageSpan) -> set[str]:
    if not span.message.is_tool() or span.message.tool_call_id is None:
        return set()
    return {span.message.tool_call_id}


def _descendant_tool_call_ids(span: ChatMessageSpan) -> set[str]:
    tool_call_ids = _visible_tool_call_ids(span)
    for child in span.children:
        tool_call_ids.update(_descendant_tool_call_ids(child))
    return tool_call_ids


def _descendant_tool_output_ids(span: ChatMessageSpan) -> set[str]:
    tool_output_ids = _visible_tool_output_ids(span)
    for child in span.children:
        tool_output_ids.update(_descendant_tool_output_ids(child))
    return tool_output_ids


def _expand_dependency_closure(spans: list[ChatMessageSpan]) -> list[ChatMessageSpan]:
    rendered = [*spans]
    while True:
        visible_tool_call_ids: set[str] = set()
        visible_tool_output_ids: set[str] = set()
        for span in rendered:
            visible_tool_call_ids.update(_visible_tool_call_ids(span))
            visible_tool_output_ids.update(_visible_tool_output_ids(span))
        missing_tool_output_ids = visible_tool_call_ids - visible_tool_output_ids
        missing_tool_call_ids = visible_tool_output_ids - visible_tool_call_ids
        if not missing_tool_output_ids and not missing_tool_call_ids:
            return rendered
        indexes_to_expand: list[int] = []
        for index, span in enumerate(rendered):
            if not span.children:
                continue
            if missing_tool_output_ids & _descendant_tool_output_ids(span):
                indexes_to_expand.append(index)
                continue
            if missing_tool_call_ids & _descendant_tool_call_ids(span):
                indexes_to_expand.append(index)
        if not indexes_to_expand:
            return rendered
        for index in reversed(indexes_to_expand):
            rendered[index : index + 1] = rendered[index].children


def _expanded_candidate_render(
    spans: list[ChatMessageSpan],
    *,
    candidate: _ExpansionCandidate,
) -> list[ChatMessageSpan]:
    rendered = [*spans]
    rendered[candidate.index : candidate.index + 1] = rendered[candidate.index].children
    return _expand_dependency_closure(rendered)


def render_main_transcript(
    spans: tuple[ChatMessageSpan, ...],
    *,
    token_budget: int,
) -> tuple[ChatMessageSpan, ...]:
    return render_budgeted_spans(spans, token_budget=token_budget)


def render_budgeted_spans(
    spans: tuple[ChatMessageSpan, ...],
    *,
    measure: Callable[[tuple[ChatMessageSpan, ...]], int] | None = None,
    recount_margin: int = 0,
    token_budget: int,
) -> tuple[ChatMessageSpan, ...]:
    if token_budget < 0:
        raise ValueError("token budget must be non-negative")
    if recount_margin < 0:
        raise ValueError("recount margin must be non-negative")
    current_tokens = sum(span.token_count for span in spans)
    rendered = list(spans)
    while candidates := _expansion_candidates(rendered, current_tokens=current_tokens, token_budget=token_budget):
        applied = False
        for candidate in sorted(candidates, key=_candidate_sort_key):
            next_rendered = _expanded_candidate_render(rendered, candidate=candidate)
            next_tokens = sum(span.token_count for span in next_rendered)
            if next_tokens > token_budget:
                continue
            if measure is None or token_budget - next_tokens > recount_margin:
                rendered = next_rendered
                current_tokens = next_tokens
                applied = True
                break
            if measure(tuple(next_rendered)) <= token_budget:
                rendered = next_rendered
                current_tokens = next_tokens
                applied = True
                break
        if not applied:
            break
    return tuple(rendered)


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
                        tool_calls=tuple(pending_tool_calls.get(call._id or "", call) for call in compact[-1].tool_calls),
                    )
            continue

        item = message.to_transcript_message()
        compact.append(item)
        pending_tool_calls = {call._id or "": call for call in item.tool_calls if call._id is not None}
    return tuple(item.without_ids() for item in compact)


__all__ = [
    "ChatMessageSpan",
    "TranscriptMessage",
    "compact_transcript",
    "render_budgeted_spans",
    "render_main_transcript",
]
