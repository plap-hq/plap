from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import msgspec

from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatMessage,
    ChatResponseFormat,
    ChatResponseFormatType,
    ChatToolCall,
    ChatToolChoiceFunction,
    ChatToolChoiceMode,
    content_to_primitive,
)


@dataclass(frozen=True, slots=True)
class ToolOutputTurn:
    assistant: ChatMessage
    outputs: list[ChatMessage]


def canonical_json(value: object) -> str:
    return msgspec.json.encode(value, order="deterministic").decode()


def _decode_json_or_none(value: str) -> object | None:
    try:
        return msgspec.json.decode(value)
    except msgspec.DecodeError, TypeError, ValueError:
        return None


def _fence(text: str, language: str) -> list[str]:
    fence = "```"
    while fence in text:
        fence += "`"
    return [f"{fence}{language}", text, fence]


def _json_fence(value: object) -> list[str]:
    return _fence(canonical_json(value), "json")


def _text_fence(value: str) -> list[str]:
    return _fence(value, "text")


def _content_lines(content: object | None) -> tuple[str, list[str]]:
    if content is None:
        return "text", [""]
    if isinstance(content, str):
        return "text", [content]
    return "json", [canonical_json(content_to_primitive(content))]


def _append_text_section(lines: list[str], heading: str, text: str | None) -> None:
    if text is None:
        return
    lines.append(f"### {heading}")
    lines.extend(_text_fence(text))


def _append_content_section(lines: list[str], heading: str, content: object | None) -> None:
    if content is None:
        return
    language, body_lines = _content_lines(content)
    lines.append(f"### {heading}")
    lines.extend(_fence("\n".join(body_lines), language))


def _append_tool_call(lines: list[str], call: ChatToolCall) -> None:
    decoded = _decode_json_or_none(call.arguments)
    lines.append(f"### tool_call {call.name}")
    if decoded is None:
        lines.extend(_text_fence(call.arguments))
        return
    lines.extend(_json_fence(decoded))


def _append_tool_output(lines: list[str], call: ChatToolCall, output: ChatMessage) -> None:
    language, body_lines = _content_lines(output.content)
    lines.append(f"### tool_output {call.name}")
    lines.extend(_fence("\n".join(body_lines), language))


def assistant_markdown(message: ChatMessage, *, tool_calls: list[ChatToolCall] | None = None) -> str:
    lines = ["## assistant"]
    _append_text_section(lines, "reasoning_content", message.reasoning_content)
    _append_content_section(lines, "content", message.content)
    _append_text_section(lines, "refusal", message.refusal)
    for call in message.tool_calls if tool_calls is None else tool_calls:
        _append_tool_call(lines, call)
    return "\n".join(lines)


def tool_outputs_markdown(turn: ToolOutputTurn) -> str:
    lines = ["## tool"]
    for call, output in zip(turn.assistant.tool_calls, turn.outputs, strict=True):
        _append_tool_output(lines, call, output)
    return "\n".join(lines)


def _tool_choice_value(tool_choice: object | None) -> object | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, ChatToolChoiceFunction):
        return {"type": tool_choice.type, "name": tool_choice.name}
    if isinstance(tool_choice, ChatToolChoiceMode):
        return tool_choice.value
    return str(tool_choice)


def _response_format_value(response_format: ChatResponseFormat | None) -> object | None:
    if response_format is None:
        return None
    value: dict[str, object] = {"type": response_format.type.value}
    if response_format.type == ChatResponseFormatType.JSON_SCHEMA:
        if response_format.name is not None:
            value["name"] = response_format.name
        if response_format.description is not None:
            value["description"] = response_format.description
        if response_format.strict is not None:
            value["strict"] = response_format.strict
        value["schema"] = response_format.schema or {}
    return value


def requirements_value(request: ChatCompletionRequest) -> dict[str, object | None]:
    return {
        "tool_choice": _tool_choice_value(request.tool_choice) or "auto",
        "parallel_tool_calls": True if request.parallel_tool_calls is None else request.parallel_tool_calls,
        "response_format": _response_format_value(request.response_format),
    }


def requirements_markdown(value: Mapping[str, object | None]) -> str:
    return "\n".join(["# requirements", *_json_fence(value)])


def requirements_instruction(request: ChatCompletionRequest) -> str:
    value = requirements_value(request)
    if not any(item is not None for item in value.values()):
        return ""
    encoded = canonical_json(value)
    fence = "```"
    while fence in encoded:
        fence += "`"
    lines = [f"{fence}json", encoded, fence]
    return "\n".join(["# requirements", *lines])


def note_instruction(note: str) -> str:
    return "\n".join(["# note from previous phase (may be stale)", *_text_fence(note)])


def latest_closed_tool_output_turn(history: list[ChatMessage]) -> ToolOutputTurn | None:
    if not history or not history[-1].is_tool():
        return None
    suffix_start = len(history)
    while suffix_start > 0 and history[suffix_start - 1].is_tool():
        suffix_start -= 1
    anchor_index = suffix_start - 1
    if anchor_index < 0:
        return None
    assistant = history[anchor_index]
    if not assistant.is_assistant() or not assistant.tool_calls:
        return None
    outputs_by_id: dict[str, ChatMessage] = {}
    for output in history[suffix_start:]:
        if output.tool_call_id is None:
            return None
        outputs_by_id[output.tool_call_id] = output
    outputs: list[ChatMessage] = []
    for call in assistant.tool_calls:
        output = outputs_by_id.get(call.id)
        if output is None:
            return None
        outputs.append(output)
    return ToolOutputTurn(assistant=assistant, outputs=outputs)


def _tool_output_name(msg: ChatMessage) -> str | None:
    if msg.name is not None:
        return msg.name
    advisor = msg.durable.get("advisor")
    if not isinstance(advisor, Mapping):
        return None
    tool_name = advisor.get("tool_name")
    return tool_name if isinstance(tool_name, str) else None


def render_main_message_line(msg: ChatMessage) -> list[str]:
    lines: list[str] = []
    role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
    lines.append(f"## {role}")
    if msg.role == "assistant":
        _append_text_section(lines, "reasoning_content", msg.reasoning_content)
    if msg.refusal is not None:
        _append_text_section(lines, "refusal", msg.refusal)
    tool_name = _tool_output_name(msg)
    if msg.role == "tool" and tool_name is not None:
        language, body_lines = _content_lines(msg.content)
        lines.append(f"### tool_output {tool_name}")
        lines.extend(_fence("\n".join(body_lines), language))
    else:
        _append_content_section(lines, "content", msg.content)
    for call in msg.tool_calls:
        _append_tool_call(lines, call)
    return lines


def render_main_messages(messages: list[ChatMessage]) -> list[str]:
    lines: list[str] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.is_assistant() and msg.tool_calls:
            lines.extend(render_main_message_line(msg))
            i += 1
            outputs: list[ChatMessage] = []
            while i < len(messages) and messages[i].is_tool():
                outputs.append(messages[i])
                i += 1
            if outputs:
                outputs_by_id: dict[str, ChatMessage] = {}
                for out in outputs:
                    if out.tool_call_id is not None:
                        outputs_by_id[out.tool_call_id] = out
                tool_lines = ["## tool"]
                for call in msg.tool_calls:
                    out = outputs_by_id.get(call.id)
                    if out is not None:
                        _append_tool_output(tool_lines, call, out)
                if len(tool_lines) > 1:
                    lines.extend(tool_lines)
            continue
        lines.extend(render_main_message_line(msg))
        i += 1
    return lines


__all__ = [
    "ToolOutputTurn",
    "assistant_markdown",
    "canonical_json",
    "latest_closed_tool_output_turn",
    "note_instruction",
    "render_main_messages",
    "requirements_markdown",
    "requirements_value",
    "tool_outputs_markdown",
]
