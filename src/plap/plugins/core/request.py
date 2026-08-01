from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from plap.config import CueBox
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolChoiceFunction,
    OutputEquivalence,
)
from plap.responses.contracts import FunctionTool, ResponseCreateRequest, ToolChoiceFunction
from plap.responses.state import State

DEVELOPER_PROMPT_TEMPLATE = "You are {model_name}, an AI assistant."


def apply_float_transform(
    value: float | None,
    config: object | None,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if config is None:
        return value
    if config.disabled:
        return None
    fixed = config.fixed
    if fixed is not None:
        resolved = float(fixed)
    elif value is None:
        resolved = config.default
    else:
        resolved = value * config.scale + config.offset
    if resolved is None:
        return None
    min_value = config.min_value
    if min_value is not None:
        resolved = max(resolved, float(min_value))
    max_value = config.max_value
    if max_value is not None:
        resolved = min(resolved, float(max_value))
    if minimum is not None:
        resolved = max(resolved, minimum)
    if maximum is not None:
        resolved = min(resolved, maximum)
    return resolved


def apply_int_transform(
    value: int | None,
    config: object | None,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if config is None:
        return value
    if config.disabled:
        return None
    fixed = config.fixed
    if fixed is not None:
        resolved = int(fixed)
    elif value is None:
        resolved = config.default
    else:
        resolved = value
    if resolved is None:
        return None
    min_value = config.min_value
    if min_value is not None:
        resolved = max(int(resolved), int(min_value))
    max_value = config.max_value
    if max_value is not None:
        resolved = min(int(resolved), int(max_value))
    if minimum is not None:
        resolved = max(resolved, minimum)
    if maximum is not None:
        resolved = min(resolved, maximum)
    return int(resolved)


def _response_format(state: State) -> ChatResponseFormat | None:
    text = state.request.text
    if text is None or text.format is None:
        return None
    format_ = text.format
    if format_.type == "json_schema":
        return ChatResponseFormat(
            type=format_.type,
            name=format_.name,
            schema=format_.schema_,
            strict=format_.strict,
            description=format_.description,
        )
    return ChatResponseFormat(type=format_.type)


def _tool_choice(state: State) -> str | ChatToolChoiceFunction | None:
    tool_choice = state.request.tool_choice
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, ToolChoiceFunction):  # pragma: no cover
        raise TypeError(f"unsupported tool choice: {type(tool_choice).__name__}")
    return ChatToolChoiceFunction(name=tool_choice.name)


def build_output_equivalence(config: object) -> OutputEquivalence:
    equivalence = config.output_equivalence
    return OutputEquivalence(
        uncached_input_to_output=float(equivalence.uncached_input_to_output),
        cached_input_to_output=float(equivalence.cached_input_to_output),
        output_to_output=float(equivalence.output_to_output),
    )


def build_chat_request(
    config: CueBox,
    request: ResponseCreateRequest,
    messages: Sequence[ChatMessage],
) -> ChatCompletionRequest:
    sampling = config.sampling
    return ChatCompletionRequest(
        model=config.model,
        messages=list(messages),
        max_completion_tokens=config.max_completion_tokens,
        temperature=apply_float_transform(request.temperature, sampling.temperature, minimum=0, maximum=2),
        top_p=apply_float_transform(request.top_p, sampling.top_p, minimum=0, maximum=1),
        min_p=apply_float_transform(None, sampling.min_p, minimum=0, maximum=1),
        top_k=apply_int_transform(None, sampling.top_k, minimum=0),
        frequency_penalty=apply_float_transform(None, sampling.frequency_penalty, minimum=-2, maximum=2),
        presence_penalty=apply_float_transform(None, sampling.presence_penalty, minimum=-2, maximum=2),
        repetition_penalty=apply_float_transform(None, sampling.repetition_penalty, minimum=0, maximum=2),
        seed=apply_int_transform(None, sampling.seed),
        reasoning_effort=config.reasoning_effort,
        service_tier=config.service_tier,
        output_equivalence=build_output_equivalence(config),
    )


def build_response_request(state: State) -> ChatCompletionRequest:
    request = state.request
    main = state.config.main
    sampling = main.sampling
    top_logprobs = apply_int_transform(request.top_logprobs, sampling.top_logprobs, minimum=0, maximum=20)
    instructions = [
        ChatMessage(
            role="developer",
            content=DEVELOPER_PROMPT_TEMPLATE.format(model_name=state.config.display_name),
        )
    ]
    if request.instructions is not None:
        instructions.append(ChatMessage(role="developer", content=request.instructions))
    tools = [
        ChatTool(
            function=ChatFunctionTool(
                name=tool.name,
                parameters=tool.parameters,
                strict=tool.strict,
                description=tool.description,
            )
        )
        for tool in request.tools or []
        if isinstance(tool, FunctionTool)
    ]
    return replace(
        build_chat_request(main, request, [*instructions, *state.threads["main"]]),
        tools=tools,
        tool_choice=_tool_choice(state),
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=_response_format(state),
        logprobs=True if top_logprobs is not None else None,
        top_logprobs=top_logprobs,
        stream_options=ChatStreamOptions(include_usage=True),
        user=request.user,
        prompt_cache_key=request.prompt_cache_key,
        metadata=request.metadata,
    )


__all__ = [
    "apply_float_transform",
    "apply_int_transform",
    "build_chat_request",
    "build_output_equivalence",
    "build_response_request",
]
