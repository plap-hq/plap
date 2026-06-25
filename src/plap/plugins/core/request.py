from __future__ import annotations

from plap.config import CueBox
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolChoiceFunction,
)
from plap.responses.contracts import FunctionTool, ToolChoiceFunction
from plap.responses.ingest.models import MAIN_SIDE
from plap.responses.state import State

DEVELOPER_PROMPT_TEMPLATE = "You are {model_name}, an AI assistant."


def _apply_float_transform(
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


def _apply_int_transform(
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
    text = state.prepared.execution_request.text
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
    tool_choice = state.prepared.execution_request.tool_choice
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, ToolChoiceFunction):  # pragma: no cover
        raise TypeError(f"unsupported tool choice: {type(tool_choice).__name__}")
    return ChatToolChoiceFunction(name=tool_choice.name)


def build_config_request(state: State) -> dict[str, object]:
    request = {"model": state.prepared.response_request.model}
    reasoning = state.prepared.response_request.reasoning
    if reasoning is not None and reasoning.effort is not None:
        request["reasoning_effort"] = reasoning.effort
    if state.prepared.response_request.service_tier is not None:
        request["service_tier"] = state.prepared.response_request.service_tier
    return request


def build_response_request(state: State, config: CueBox) -> ChatCompletionRequest:
    request = state.prepared.execution_request
    main = config.main
    sampling = main.sampling
    top_logprobs = _apply_int_transform(request.top_logprobs, sampling.top_logprobs, minimum=0, maximum=20)
    instructions = [
        ChatMessage(
            role="developer",
            content=DEVELOPER_PROMPT_TEMPLATE.format(model_name=config.display_name),
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
    return ChatCompletionRequest(
        model=main.model,
        messages=[*instructions, *state.history(MAIN_SIDE)],
        tools=tools,
        tool_choice=_tool_choice(state),
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=_response_format(state),
        temperature=_apply_float_transform(request.temperature, sampling.temperature, minimum=0, maximum=2),
        top_p=_apply_float_transform(request.top_p, sampling.top_p, minimum=0, maximum=1),
        top_k=_apply_int_transform(None, sampling.top_k, minimum=0),
        frequency_penalty=_apply_float_transform(None, sampling.frequency_penalty, minimum=-2, maximum=2),
        presence_penalty=_apply_float_transform(None, sampling.presence_penalty, minimum=-2, maximum=2),
        logprobs=True if top_logprobs is not None else None,
        top_logprobs=top_logprobs,
        reasoning_effort=main.reasoning_effort,
        stream_options=ChatStreamOptions(include_usage=True),
        user=request.user,
        prompt_cache_key=request.prompt_cache_key,
        metadata=request.metadata,
        service_tier=main.service_tier,
    )
