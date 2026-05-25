from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Protocol

import msgspec
import tiktoken

from plap.llms.completions.chat import ChatCompletionRequest, ChatResponseFormat, ChatRole, ChatTool, ReasoningEffort
from plap.llms.completions.chat import ChatMessage as LLMChatMessage
from plap.llms.completions.chat import ChatToolCall as LLMChatToolCall
from plap.llms.completions.encoding_dsv4 import encode_messages as encode_dsv4_messages


class ITokenizerConfig(Protocol):
    tokenizer_hf_repo: str | None
    tokenizer_revision: str | None
    tokenizer_trust_remote_code: bool


_DEFAULT_ENCODING = "o200k_base"
_DSV4_TOKENIZER_REPOS = frozenset({"deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro"})
_DIRECT_FAST_TOKENIZER_REPOS = frozenset({"stepfun-ai/Step-3.5-Flash"})


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_DEFAULT_ENCODING)


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 1
    return max(1, len(_encoding().encode(text)))


def _json_text(value: object) -> str:
    return msgspec.json.encode(value, order="deterministic").decode()


def _decoded_json_or_none(value: str) -> object | None:
    try:
        return msgspec.json.decode(value.encode())
    except msgspec.DecodeError:
        return None


def _canonical_json_text(value: str) -> str:
    decoded = _decoded_json_or_none(value)
    if decoded is None:
        return value
    return _json_text(decoded)


def _tool_definition_value(tool: ChatTool) -> dict[str, object]:
    return {
        "type": tool.type,
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters or {},
            "strict": tool.function.strict,
        },
    }


def _tool_definitions_value(tools: Sequence[ChatTool]) -> list[dict[str, object]]:
    return [_tool_definition_value(tool) for tool in tools]


def _tool_call_value(tool_call: LLMChatToolCall, *, arguments: object) -> dict[str, object]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": arguments,
        },
    }


def _template_tool_call(tool_call: LLMChatToolCall) -> dict[str, object]:
    arguments = _decoded_json_or_none(tool_call.arguments)
    if arguments is None:
        arguments = tool_call.arguments
    return _tool_call_value(tool_call, arguments=arguments)


def _fallback_tool_call(tool_call: LLMChatToolCall) -> dict[str, object]:
    return _tool_call_value(tool_call, arguments=_canonical_json_text(tool_call.arguments))


def _dsv4_tool_call(tool_call: LLMChatToolCall) -> dict[str, object]:
    return _tool_call_value(tool_call, arguments=tool_call.arguments)


def _response_format_value(response_format: ChatResponseFormat | None) -> dict[str, object] | None:
    if response_format is None:
        return None
    if response_format.type != "json_schema":
        return {"type": response_format.type}
    json_schema: dict[str, object] = {
        "name": response_format.name,
        "schema": response_format.schema or {},
    }
    if response_format.strict is not None:
        json_schema["strict"] = response_format.strict
    if response_format.description is not None:
        json_schema["description"] = response_format.description
    return {"type": "json_schema", "json_schema": json_schema}


def _fallback_message_value(message: LLMChatMessage) -> dict[str, object]:
    value: dict[str, object] = {"role": message.role}
    if message.content is not None:
        value["content"] = message.content
    if message.name is not None:
        value["name"] = message.name
    if message.tool_call_id is not None:
        value["tool_call_id"] = message.tool_call_id
    if message.refusal is not None:
        value["refusal"] = message.refusal
    if message.reasoning_content is not None:
        value["reasoning_content"] = message.reasoning_content
    if message.reasoning_details:
        value["reasoning_details"] = message.reasoning_details
    if message.tool_calls:
        value["tool_calls"] = [_fallback_tool_call(tool_call) for tool_call in message.tool_calls]
    return value


def _fallback_request_value(
    messages: Sequence[LLMChatMessage],
    *,
    tools: Sequence[ChatTool],
    response_format: ChatResponseFormat | None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "messages": [_fallback_message_value(message) for message in messages],
    }
    if tools:
        value["tools"] = _tool_definitions_value(tools)
    rendered_response_format = _response_format_value(response_format)
    if rendered_response_format is not None:
        value["response_format"] = rendered_response_format
    return value


def _template_message(message: LLMChatMessage) -> dict[str, object]:
    value: dict[str, object] = {
        "role": "system" if message.role == ChatRole.DEVELOPER else message.role,
    }
    if message.content is not None:
        value["content"] = message.content
    if message.name is not None:
        value["name"] = message.name
    if message.tool_call_id is not None:
        value["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        value["tool_calls"] = [_template_tool_call(tool_call) for tool_call in message.tool_calls]
    if message.refusal is not None:
        value["refusal"] = message.refusal
    return value


def _reasoning_value(message: LLMChatMessage) -> str | None:
    value: dict[str, object] = {}
    if message.reasoning_content is not None:
        value["reasoning_content"] = message.reasoning_content
    if message.reasoning_details:
        value["reasoning_details"] = message.reasoning_details
    if not value:
        return None
    return _json_text(value)


def _uses_dsv4_encoding(tokenizer_config: ITokenizerConfig) -> bool:
    return tokenizer_config.tokenizer_hf_repo in _DSV4_TOKENIZER_REPOS


def _dsv4_message(message: LLMChatMessage) -> dict[str, object] | None:
    if message.reasoning_details and message.reasoning_content is None:
        return None
    value: dict[str, object] = {"role": message.role}
    if message.content is not None:
        value["content"] = message.content
    elif message.refusal is not None:
        value["content"] = message.refusal
    if message.name is not None:
        value["name"] = message.name
    if message.tool_call_id is not None:
        value["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        value["tool_calls"] = [_dsv4_tool_call(tool_call) for tool_call in message.tool_calls]
    if message.reasoning_content is not None:
        value["reasoning_content"] = message.reasoning_content
    return value


def _dsv4_message_target_index(messages: list[dict[str, object]]) -> int:
    for index, message in enumerate(messages):
        if message["role"] in {ChatRole.DEVELOPER, ChatRole.SYSTEM}:
            return index
    messages.insert(0, {"role": ChatRole.SYSTEM, "content": ""})
    return 0


def _dsv4_messages(
    messages: Sequence[LLMChatMessage],
    *,
    tools: Sequence[ChatTool],
    response_format: ChatResponseFormat | None,
) -> list[dict[str, object]] | None:
    rendered_messages: list[dict[str, object]] = []
    for message in messages:
        rendered = _dsv4_message(message)
        if rendered is None:
            return None
        rendered_messages.append(rendered)
    rendered_tools = _tool_definitions_value(tools)
    rendered_response_format = _response_format_value(response_format)
    if rendered_tools or rendered_response_format is not None:
        target = rendered_messages[_dsv4_message_target_index(rendered_messages)]
        if rendered_tools:
            target["tools"] = rendered_tools
        if rendered_response_format is not None:
            target["response_format"] = rendered_response_format
    return rendered_messages


def _dsv4_thinking_mode(
    messages: Sequence[LLMChatMessage],
    *,
    reasoning_effort: ReasoningEffort | None,
) -> str:
    if reasoning_effort not in {None, "none"}:
        return "thinking"
    if any(message.reasoning_content is not None for message in messages):
        return "thinking"
    return "chat"


def _dsv4_reasoning_effort(reasoning_effort: ReasoningEffort | None) -> str | None:
    if reasoning_effort == "xhigh":
        return "max"
    if reasoning_effort == "high":
        return "high"
    return None


def _dsv4_has_tool_history(messages: Sequence[LLMChatMessage], *, tools: Sequence[ChatTool]) -> bool:
    if tools:
        return True
    return any(message.tool_calls or message.role == ChatRole.TOOL for message in messages)


def _tokenizer_chat_template_supported(tokenizer: Any) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    return callable(getattr(tokenizer, "apply_chat_template", None)) and bool(chat_template)


@lru_cache(maxsize=32)
def _hf_tokenizer(
    repo: str,
    revision: str | None,
    trust_remote_code: bool,
):
    if repo in _DIRECT_FAST_TOKENIZER_REPOS:
        from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

        # Step-3.5 publishes a working fast tokenizer and chat template, but the
        # pinned model config fails HF config validation before AutoTokenizer can
        # reach it. Loading the tokenizer directly avoids that broken config path.
        return PreTrainedTokenizerFast.from_pretrained(repo, revision=revision)
    from transformers import AutoTokenizer  # noqa: PLC0415

    return AutoTokenizer.from_pretrained(
        repo,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )


def _tokenize_text_with_tokenizer_config(text: str, *, tokenizer_config: ITokenizerConfig) -> int:
    if tokenizer_config.tokenizer_hf_repo is None:
        return estimate_text_tokens(text)
    tokenizer = _hf_tokenizer(
        tokenizer_config.tokenizer_hf_repo,
        tokenizer_config.tokenizer_revision,
        tokenizer_config.tokenizer_trust_remote_code,
    )
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def _dsv4_prompt_token_count(
    messages: Sequence[LLMChatMessage],
    *,
    tokenizer_config: ITokenizerConfig,
    tools: Sequence[ChatTool],
    response_format: ChatResponseFormat | None,
    reasoning_effort: ReasoningEffort | None,
) -> int | None:
    if not _uses_dsv4_encoding(tokenizer_config):
        return None
    rendered_messages = _dsv4_messages(messages, tools=tools, response_format=response_format)
    if rendered_messages is None:
        return None
    prompt = encode_dsv4_messages(
        rendered_messages,
        thinking_mode=_dsv4_thinking_mode(messages, reasoning_effort=reasoning_effort),
        drop_thinking=not _dsv4_has_tool_history(messages, tools=tools),
        reasoning_effort=_dsv4_reasoning_effort(reasoning_effort),
    )
    return _tokenize_text_with_tokenizer_config(prompt, tokenizer_config=tokenizer_config)


def _template_prompt_token_count(
    messages: Sequence[LLMChatMessage],
    *,
    tokenizer_config: ITokenizerConfig,
    tools: Sequence[ChatTool],
    response_format: ChatResponseFormat | None,
) -> int | None:
    if tokenizer_config.tokenizer_hf_repo is None:
        return None
    tokenizer = _hf_tokenizer(
        tokenizer_config.tokenizer_hf_repo,
        tokenizer_config.tokenizer_revision,
        tokenizer_config.tokenizer_trust_remote_code,
    )
    if not _tokenizer_chat_template_supported(tokenizer):
        return None
    token_ids = tokenizer.apply_chat_template(
        [_template_message(message) for message in messages],
        tools=_tool_definitions_value(tools) or None,
        response_format=_response_format_value(response_format),
        add_generation_prompt=False,
        tokenize=True,
    )
    count = max(1, len(token_ids))
    for message in messages:
        reasoning_text = _reasoning_value(message)
        if reasoning_text is None:
            continue
        count += _tokenize_text_with_tokenizer_config(reasoning_text, tokenizer_config=tokenizer_config)
    return count


def _measure_prompt_surface_tokens(
    messages: Sequence[LLMChatMessage],
    *,
    tokenizer_config: ITokenizerConfig,
    tools: Sequence[ChatTool] = (),
    response_format: ChatResponseFormat | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> int:
    dsv4_count = _dsv4_prompt_token_count(
        messages,
        tokenizer_config=tokenizer_config,
        tools=tools,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )
    if dsv4_count is not None:
        return dsv4_count
    template_count = _template_prompt_token_count(
        messages,
        tokenizer_config=tokenizer_config,
        tools=tools,
        response_format=response_format,
    )
    if template_count is not None:
        return template_count
    return _tokenize_text_with_tokenizer_config(
        _json_text(_fallback_request_value(messages, tools=tools, response_format=response_format)),
        tokenizer_config=tokenizer_config,
    )


def measure_prompt_tokens(
    messages: Sequence[LLMChatMessage],
    *,
    tokenizer_config: ITokenizerConfig,
    tools: Sequence[ChatTool] = (),
    response_format: ChatResponseFormat | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> int:
    return _measure_prompt_surface_tokens(
        messages,
        tokenizer_config=tokenizer_config,
        tools=tools,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )


def measure_request_tokens(
    request: ChatCompletionRequest,
    *,
    tokenizer_config: ITokenizerConfig,
) -> int:
    return _measure_prompt_surface_tokens(
        request.messages,
        tokenizer_config=tokenizer_config,
        tools=request.tools,
        response_format=request.response_format,
        reasoning_effort=request.reasoning_effort,
    )
