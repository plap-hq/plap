from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

import blake3
import fastjsonschema
import msgspec

from plap.llms.accumulator import Accumulator, Snapshot
from plap.llms.completions.chat import ChatCompletionRequest, ChatCompletionResult, ChatMessage, ChatTool, IChatCompletionClient
from plap.llms.completions.errors import ChatCompletionProviderError

RETRY_TOOL_PLACEHOLDER = (
    "This tool call was not executed because the assistant attempt was rejected. If you still need this tool, call it again."
)


class RetryError(Exception):
    pass


class RetryLimitExceededError(RetryError):
    pass


class RetryToolSchemaError(RetryError):
    pass


type RetryValidator = Callable[[ChatCompletionResult, ChatCompletionRequest], str | None]
type NextRequest = Callable[[Snapshot], ChatCompletionRequest | None]

type _SchemaValidator = Callable[[Any], Any]
_SCHEMA_VALIDATORS: dict[bytes, _SchemaValidator] = {}


def _tool_stubs(message: ChatMessage) -> tuple[ChatMessage, ...]:
    return tuple(
        ChatMessage(
            role="tool",
            tool_call_id=call.id,
            content=RETRY_TOOL_PLACEHOLDER,
        )
        for call in (message.tool_calls or ())
    )


def _retry_message(*, problems: Sequence[str], rules: Sequence[str]) -> str:
    problem_block = "\n".join(f"- {problem}" for problem in problems)
    rule_block = "\n".join(f"- {rule}" for rule in rules)
    return (
        "Your previous answer could not be used as written.\n\n"
        "Problem:\n"
        f"{problem_block}\n\n"
        "Reply again for the same task. Keep the substance if it was correct, and fix only the unusable part.\n"
        f"{rule_block}"
    )


def _single_declared_tool_name(tools: Sequence[ChatTool]) -> str | None:
    if len(tools) != 1:
        return None
    return f"`{tools[0].function.name}`"


def _unknown_tool_retry_message(call_name: str, *, tools: Sequence[ChatTool]) -> str:
    declared_tool_name = _single_declared_tool_name(tools)
    if declared_tool_name is None:
        rule = "Use only tool names declared in the request." if tools else "Do not call tools in your next reply."
    else:
        rule = f"Use only the declared tool name: {declared_tool_name}."
    return _retry_message(
        problems=(f"You called an undeclared tool: `{call_name}`.",),
        rules=(rule,),
    )


def _invalid_tool_arguments_retry_message(tool_name: str) -> str:
    return _retry_message(
        problems=(f"The arguments for tool `{tool_name}` were not a valid JSON object.",),
        rules=(f"If you call `{tool_name}`, its `arguments` must be a JSON object.",),
    )


def _strict_schema_mismatch_retry_message(tool_name: str, *, error_message: str) -> str:
    return _retry_message(
        problems=(
            f"The arguments for strict tool `{tool_name}` did not match its declared schema.",
            f"Validation error: `{error_message}`.",
        ),
        rules=(f"If you call `{tool_name}`, its `arguments` must match the declared schema exactly.",),
    )


def _schema_cache_key(schema: dict[str, Any] | None, *, tool_name: str) -> bytes:
    if schema is None:
        raise RetryToolSchemaError(f"strict tool schema for {tool_name!r} is missing")
    try:
        encoded = msgspec.json.encode(schema, order="deterministic")
    except Exception as exc:  # pragma: no cover - defensive encode failure
        raise RetryToolSchemaError(f"strict tool schema for {tool_name!r} could not be encoded") from exc
    return blake3.blake3(encoded).digest()


def _compiled_schema_validator(tool: ChatTool) -> _SchemaValidator:
    cache_key = _schema_cache_key(tool.function.parameters, tool_name=tool.function.name)
    validator = _SCHEMA_VALIDATORS.get(cache_key)
    if validator is not None:
        return validator
    try:
        validator = fastjsonschema.compile(tool.function.parameters, use_default=False)
    except Exception as exc:
        raise RetryToolSchemaError(f"strict tool schema for {tool.function.name!r} could not be compiled") from exc
    _SCHEMA_VALIDATORS[cache_key] = validator
    return validator


def _decoded_json_object(arguments: str) -> dict[str, Any] | None:
    try:
        value = msgspec.json.decode(arguments.encode())
    except msgspec.DecodeError:
        return None
    if not isinstance(value, dict):
        return None
    return value


def retry_on_unusable_tool_calls(
    result: ChatCompletionResult,
    request: ChatCompletionRequest,
) -> str | None:
    tool_calls = result.message.tool_calls or ()
    if not tool_calls:
        return None
    tools_by_name = {tool.function.name: tool for tool in request.tools}
    for call in tool_calls:
        tool = tools_by_name.get(call.name)
        if tool is None:
            return _unknown_tool_retry_message(call.name, tools=request.tools)
        arguments = _decoded_json_object(call.arguments)
        if arguments is None:
            return _invalid_tool_arguments_retry_message(call.name)
        if tool.function.strict is not True:
            continue
        validator = _compiled_schema_validator(tool)
        try:
            validator(arguments)
        except fastjsonschema.JsonSchemaValueException as exc:
            error_message = getattr(exc, "message", str(exc)).strip()
            return _strict_schema_mismatch_retry_message(tool.function.name, error_message=error_message)
    return None


def _first_retry_message(
    result: ChatCompletionResult,
    request: ChatCompletionRequest,
    validators: Sequence[RetryValidator],
) -> str | None:
    for validator in validators:
        retry_message = validator(result, request)
        if retry_message is not None:
            return retry_message
    return None


async def stream(
    client: IChatCompletionClient,
    *,
    next_request: NextRequest,
    validators: Sequence[RetryValidator] = (),
    max_attempts: int = 3,
) -> AsyncIterator[Snapshot]:
    history = Snapshot(messages=(), results=(), delta=None)

    for _ in range(max_attempts):
        request = next_request(history)
        if request is None:
            return

        accumulator = Accumulator(tools=tuple(request.tools))
        last: Snapshot | None = None
        stream_error: ChatCompletionProviderError | None = None

        try:
            async for delta in client.stream(request):
                current = accumulator.apply(delta)
                last = Snapshot(
                    messages=(*history.messages, *current.messages),
                    results=(*history.results, *current.results),
                    delta=current.delta,
                )
                yield last
        except ChatCompletionProviderError as exc:
            stream_error = exc

        if stream_error is not None:
            if last is None:
                raise stream_error
            if not last.results:
                yield history
                continue

        if last is None or not last.results:
            raise RuntimeError("stream ended without final result")

        result = last.results[-1]
        fix = _first_retry_message(result, request, validators)
        if fix is None:
            return

        history = Snapshot(
            messages=(
                *history.messages,
                *last.messages[len(history.messages) :],
                *_tool_stubs(result.message),
                ChatMessage(role="user", content=fix),
            ),
            results=last.results,
            delta=None,
        )
        yield history

    raise RetryLimitExceededError("retry limit reached")


async def complete(
    client: IChatCompletionClient,
    *,
    next_request: NextRequest,
    validators: Sequence[RetryValidator] = (),
    max_attempts: int = 3,
) -> Snapshot:
    final = Snapshot(messages=(), results=(), delta=None)
    async for snapshot in stream(
        client,
        next_request=next_request,
        validators=validators,
        max_attempts=max_attempts,
    ):
        final = snapshot
    return final


__all__ = [
    "RETRY_TOOL_PLACEHOLDER",
    "NextRequest",
    "RetryError",
    "RetryLimitExceededError",
    "RetryToolSchemaError",
    "RetryValidator",
    "complete",
    "retry_on_unusable_tool_calls",
    "stream",
]
