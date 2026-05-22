from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import TypeAlias

from plap.llms.accumulator import Accumulator, Snapshot
from plap.llms.completions.chat import ChatCompletionRequest, ChatCompletionResult, ChatMessage, IChatCompletionClient

RETRY_TOOL_PLACEHOLDER = (
    "This tool call was not executed because the assistant attempt was rejected "
    "and retried. If you still need this tool, call it again."
)

Validate: TypeAlias = Callable[[ChatCompletionResult], str | None]
NextRequest: TypeAlias = Callable[[Snapshot], ChatCompletionRequest | None]


def _tool_stubs(message: ChatMessage) -> tuple[ChatMessage, ...]:
    return tuple(
        ChatMessage(
            role="tool",
            tool_call_id=call.id,
            content=RETRY_TOOL_PLACEHOLDER,
        )
        for call in (message.tool_calls or ())
    )


async def stream(
    client: IChatCompletionClient,
    *,
    next_request: NextRequest,
    validate: Validate,
    max_attempts: int = 3,
) -> AsyncIterator[Snapshot]:
    history = Snapshot(messages=(), results=(), delta=None)

    for _ in range(max_attempts):
        request = next_request(history)
        if request is None:
            return

        accumulator = Accumulator(tools=tuple(request.tools))
        last: Snapshot | None = None

        async for delta in client.stream(request):
            current = accumulator.apply(delta)
            last = Snapshot(
                messages=(*history.messages, *current.messages),
                results=(*history.results, *current.results),
                delta=current.delta,
            )
            yield last

        if last is None or not last.results:
            raise RuntimeError("stream ended without final result")

        result = last.results[-1]
        fix = validate(result)
        if fix is None:
            return

        history = Snapshot(
            messages=(
                *history.messages,
                *last.messages[len(history.messages):],
                *_tool_stubs(result.message),
                ChatMessage(role="user", content=fix),
            ),
            results=last.results,
            delta=None,
        )
        yield history

    raise RuntimeError("retry limit reached")


async def complete(
    client: IChatCompletionClient,
    *,
    next_request: NextRequest,
    validate: Validate,
    max_attempts: int = 3,
) -> Snapshot:
    final = Snapshot(messages=(), results=(), delta=None)
    async for final in stream(
        client,
        next_request=next_request,
        validate=validate,
        max_attempts=max_attempts,
    ):
        pass
    return final


__all__ = [
    "NextRequest",
    "RETRY_TOOL_PLACEHOLDER",
    "Validate",
    "complete",
    "stream",
]
