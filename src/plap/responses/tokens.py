from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import msgspec
import tiktoken

from plap.llms.chat import ChatCompletionRequest, ChatResponseFormat, ChatRole, ChatTool, ReasoningEffort
from plap.llms.chat import ChatMessage as LLMChatMessage
from plap.llms.chat import ChatToolCall as LLMChatToolCall
from plap.responses.encoding_dsv4 import encode_messages as encode_dsv4_messages
from plap.settings import RuntimeActorConfig

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


def _canonical_json_text(value: str) -> str:
    try:
        decoded = msgspec.json.decode(value.encode())
    except msgspec.DecodeError:
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
    try:
        arguments: object = msgspec.json.decode(tool_call.arguments.encode())
    except msgspec.DecodeError:
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


def _fallback_request_value(request: ChatCompletionRequest) -> dict[str, object]:
    value: dict[str, object] = {
        "messages": [_fallback_message_value(message) for message in request.messages],
    }
    if request.tools:
        value["tools"] = _tool_definitions_value(request.tools)
    response_format = _response_format_value(request.response_format)
    if response_format is not None:
        value["response_format"] = response_format
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


def _uses_dsv4_encoding(actor_config: RuntimeActorConfig) -> bool:
    return actor_config.tokenizer_hf_repo in _DSV4_TOKENIZER_REPOS


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


def _dsv4_messages(request: ChatCompletionRequest) -> list[dict[str, object]] | None:
    messages: list[dict[str, object]] = []
    for message in request.messages:
        rendered = _dsv4_message(message)
        if rendered is None:
            return None
        messages.append(rendered)
    tools = _tool_definitions_value(request.tools)
    response_format = _response_format_value(request.response_format)
    if tools or response_format is not None:
        target = messages[_dsv4_message_target_index(messages)]
        if tools:
            target["tools"] = tools
        if response_format is not None:
            target["response_format"] = response_format
    return messages


def _dsv4_thinking_mode(request: ChatCompletionRequest) -> str:
    if request.reasoning_effort not in {None, "none"}:
        return "thinking"
    if any(message.reasoning_content is not None for message in request.messages):
        return "thinking"
    return "chat"


def _dsv4_reasoning_effort(request: ChatCompletionRequest) -> str | None:
    if request.reasoning_effort == "xhigh":
        return "max"
    if request.reasoning_effort == "high":
        return "high"
    return None


def _dsv4_has_tool_history(request: ChatCompletionRequest) -> bool:
    if request.tools:
        return True
    return any(message.tool_calls or message.role == ChatRole.TOOL for message in request.messages)


def _tokenize_text_with_actor(text: str, *, actor_config: RuntimeActorConfig) -> int:
    if actor_config.tokenizer_hf_repo is None:
        return estimate_text_tokens(text)
    tokenizer = _hf_tokenizer(
        actor_config.tokenizer_hf_repo,
        actor_config.tokenizer_revision,
        actor_config.tokenizer_trust_remote_code,
    )
    return max(1, len(tokenizer.encode(text, add_special_tokens=False)))


def _dsv4_request_token_count(
    request: ChatCompletionRequest,
    *,
    actor_config: RuntimeActorConfig,
) -> int | None:
    if not _uses_dsv4_encoding(actor_config):
        return None
    messages = _dsv4_messages(request)
    if messages is None:
        return None
    prompt = encode_dsv4_messages(
        messages,
        thinking_mode=_dsv4_thinking_mode(request),
        drop_thinking=not _dsv4_has_tool_history(request),
        reasoning_effort=_dsv4_reasoning_effort(request),
    )
    return _tokenize_text_with_actor(prompt, actor_config=actor_config)


def _tokenizer_chat_template_supported(tokenizer: Any) -> bool:
    chat_template = getattr(tokenizer, "chat_template", None)
    return callable(getattr(tokenizer, "apply_chat_template", None)) and bool(chat_template)


def _template_request_token_count(
    request: ChatCompletionRequest,
    *,
    actor_config: RuntimeActorConfig,
) -> int | None:
    if actor_config.tokenizer_hf_repo is None:
        return None
    tokenizer = _hf_tokenizer(
        actor_config.tokenizer_hf_repo,
        actor_config.tokenizer_revision,
        actor_config.tokenizer_trust_remote_code,
    )
    if not _tokenizer_chat_template_supported(tokenizer):
        return None
    token_ids = tokenizer.apply_chat_template(
        [_template_message(message) for message in request.messages],
        tools=_tool_definitions_value(request.tools) or None,
        response_format=_response_format_value(request.response_format),
        add_generation_prompt=False,
        tokenize=True,
    )
    count = max(1, len(token_ids))
    for message in request.messages:
        reasoning_text = _reasoning_value(message)
        if reasoning_text is None:
            continue
        count += _tokenize_text_with_actor(reasoning_text, actor_config=actor_config)
    return count


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


def measure_prompt_tokens(
    messages: Sequence[LLMChatMessage],
    *,
    actor_config: RuntimeActorConfig,
    tools: Sequence[ChatTool] = (),
    response_format: ChatResponseFormat | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> int:
    return measure_request_tokens(
        ChatCompletionRequest(
            model=actor_config.model,
            messages=list(messages),
            tools=list(tools),
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        ),
        actor_config=actor_config,
    )


def measure_request_tokens(
    request: ChatCompletionRequest,
    *,
    actor_config: RuntimeActorConfig,
) -> int:
    dsv4_count = _dsv4_request_token_count(request, actor_config=actor_config)
    if dsv4_count is not None:
        return dsv4_count
    template_count = _template_request_token_count(request, actor_config=actor_config)
    if template_count is not None:
        return template_count
    return _tokenize_text_with_actor(_json_text(_fallback_request_value(request)), actor_config=actor_config)
