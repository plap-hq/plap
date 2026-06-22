from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from plap.llms.accumulator import Snapshot
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatToolCallDelta,
)
from plap.plugins.summary.summarizer import (
    IReasoningSummarizer,
    _SummaryChunker,
)


def _delta_snapshot(
    *,
    reasoning_delta: str | None = None,
    tool_boundary: bool = False,
    finish_reason: str | None = None,
) -> Snapshot:
    return Snapshot(
        messages=(),
        results=(),
        delta=ChatCompletionDelta(
            id="chatcmpl_test",
            model="model-a",
            created_at=None,
            choice_index=0,
            reasoning_delta=reasoning_delta,
            tool_call_delta=ChatToolCallDelta(index=0) if tool_boundary else None,
            finish_reason=finish_reason,
        ),
    )


def _retry_boundary_snapshot() -> Snapshot:
    return Snapshot(messages=(), results=(), delta=None)


async def _source(*snapshots: Snapshot) -> AsyncIterator[Snapshot]:
    for snapshot in snapshots:
        yield snapshot


class _SequenceSummarizer(IReasoningSummarizer):
    def __init__(self, responses: Sequence[Sequence[object]]) -> None:
        self._responses = [tuple(response) for response in responses]
        self.calls: list[tuple[str, str | None, str]] = []

    async def stream(
        self,
        *,
        mode: str,
        prior_summary: str | None,
        fragment: str,
    ) -> AsyncIterator[str]:
        self.calls.append((mode, prior_summary, fragment))
        response = self._responses[len(self.calls) - 1]
        for item in response:
            if isinstance(item, Exception):
                raise item
            yield item


def test_summary_chunker_waits_on_rule_lists_without_paragraph_breaks() -> None:
    buffer = "\n".join(
        (
            "Key points from the rules:",
            "- Output ONLY a thread title. Nothing else.",
            "- Single line, 50 characters or fewer.",
            "- No explanations.",
            "- Use the same language as the user message.",
            "- Never include tool names in the title.",
            "- Focus on the main topic or question the user needs to retrieve.",
            "- Keep exact technical terms, numbers, filenames, and HTTP codes.",
            "- Never use tools.",
            "- Always output something meaningful.",
        )
    )

    chunker = _SummaryChunker()

    assert chunker.push(buffer) == ()
    assert chunker.buffer == buffer


def test_summary_chunker_prefers_paragraph_boundary_near_end() -> None:
    paragraph_1 = "I am checking the request constraints and comparing them with the current plan. " * 4
    paragraph_2 = "I am reviewing the main failure modes and narrowing the likely cause before changing code. " * 4
    paragraph_3 = "z" * 320
    buffer = f"{paragraph_1}\n\n{paragraph_2}\n\n{paragraph_3}"

    chunker = _SummaryChunker()

    assert chunker.push(buffer) == (f"{paragraph_1}\n\n{paragraph_2}".strip(),)
    assert chunker.buffer == paragraph_3


def test_summary_chunker_hard_flushes_large_boundary_free_text() -> None:
    buffer = "x" * 900

    chunker = _SummaryChunker()

    assert chunker.push(buffer) == (buffer,)
    assert chunker.buffer == ""
