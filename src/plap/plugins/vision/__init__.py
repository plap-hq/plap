from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Mapping
from dataclasses import field, replace
from typing import Any

import blake3
import msgspec

from plap.bus import bus
from plap.config import CueBox
from plap.llms.completions.budget import BudgetedChatCompletionClient
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatContentImage,
    ChatContentPart,
    ChatContentText,
    ChatMessage,
    ChatToolCall,
)
from plap.plugins.core.request import build_chat_request
from plap.plugins.easy import ServerTool, bootstrap, server_tools
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.ingest.models import Message
from plap.responses.state import State

bootstrap.config(__file__)

VISION_TOOL_NAME = "vision"
VISION_PROMPT = (
    "Answer image questions. "
    "A standalone text token matching image-XXXX-XXXX-XXXX-XXXX labels the image immediately after it. "
    'A message starting "Selected image ids:" lists the image(s) to inspect; otherwise use images referenced by the question. '
    "Rely on visible evidence. "
    "For OCR, tables, and diagrams, read all relevant labels, numbers, units, symbols, arrows, and spatial relations exactly. "
    "Mark uncertain or unreadable details as uncertain/unreadable rather than guessing or saying absent. "
    "For charts and overlays, describe each series/object separately. "
    "Do not treat a reference curve as part of the target or infer a subgroup from mere overlap. "
    "If identifying or choosing, compare evidence with the stated criteria and qualify uncertainty. "
    "Plain text only; answer just the request, concise but complete."
)
_IMAGE_ID_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:image-)?([A-Za-z0-9]{4})-([A-Za-z0-9]{4})-([A-Za-z0-9]{4})-([A-Za-z0-9]{4})(?![A-Za-z0-9])"
)


def _data_url_bytes(url: str) -> tuple[str, bytes] | None:
    if not url.startswith("data:"):
        return None
    header, separator, payload = url[5:].partition(",")
    if not separator:
        return None
    parts = header.split(";")
    if "base64" not in {part.lower() for part in parts[1:]}:
        return None
    media_type = parts[0].strip().lower()
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.b64decode(payload + padding, validate=True)
    except binascii.Error, ValueError:
        return None
    return media_type, decoded


def _image_source(part: ChatContentImage) -> bytes:
    image_url = part.image_url
    if image_url.file_id is not None:
        return b"file_id\0" + image_url.file_id.encode("utf-8")
    url = image_url.url or ""
    data_url = _data_url_bytes(url)
    if data_url is not None:
        media_type, decoded = data_url
        return b"data\0" + media_type.encode("utf-8") + b"\0" + decoded
    return b"url\0" + url.encode("utf-8")


def _image_id(part: ChatContentImage) -> str:
    digest = blake3.blake3(_image_source(part)).digest(length=10)
    encoded = base64.b32encode(digest).decode("ascii").rstrip("=")
    return f"image-{encoded[0:4]}-{encoded[4:8]}-{encoded[8:12]}-{encoded[12:16]}"


def _replace_images(content: str | list[ChatContentPart] | None) -> tuple[str | list[ChatContentPart] | None, bool]:
    if not isinstance(content, list):
        return content, False
    rewritten: list[ChatContentPart] = []
    changed = False
    for part in content:
        if isinstance(part, ChatContentImage):
            rewritten.append(ChatContentText(text=_image_id(part)))
            changed = True
            continue
        rewritten.append(part)
    return rewritten, changed


def _rewrite_request(request: ChatCompletionRequest) -> ChatCompletionRequest:
    messages: list[ChatMessage] = []
    for message in request.messages:
        content, changed = _replace_images(message.content)
        messages.append(message if not changed else replace(message, content=content))
    return replace(request, messages=messages)


def _canonical_image_id(match: re.Match[str]) -> str:
    return "image-" + "-".join(part.upper() for part in match.groups())


def _image_ids_from_text(text: str) -> list[str]:
    return [_canonical_image_id(match) for match in _IMAGE_ID_RE.finditer(text)]


def _dedupe_image_ids(ids: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for image_id in ids:
        if image_id in seen:
            continue
        seen.add(image_id)
        deduped.append(image_id)
    return deduped


def _normalize_image_ids(value: object) -> list[str] | None:
    if isinstance(value, str) and value:
        normalized = _image_ids_from_text(value)
        if normalized:
            return _dedupe_image_ids(normalized)
        return [value]
    if not isinstance(value, list) or not value or not all(isinstance(image_id, str) and image_id for image_id in value):
        return None

    raw = list(value)
    normalized: list[str] = []
    for image_id in raw:
        normalized.extend(_image_ids_from_text(image_id))
    normalized.extend(_image_ids_from_text("".join(raw)))
    if normalized:
        return _dedupe_image_ids(normalized)
    return raw


def _tool_args(encoded: str) -> tuple[list[str], str]:
    arguments = msgspec.json.decode(encoded)
    if not isinstance(arguments, dict):
        raise TypeError("vision tool arguments must decode to an object")
    ids = _normalize_image_ids(arguments.get("ids"))
    prompt = arguments.get("prompt")
    if ids is None:
        raise RuntimeError("vision tool arguments must include image ids")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("vision tool arguments must include a non-empty prompt")
    return ids, prompt


def _tool_output_text(result: ChatCompletionResult) -> str:
    message = result.message
    if isinstance(message.content, str):
        text = message.content.strip()
        if text:
            return text
    elif isinstance(message.content, list):
        text = "\n\n".join(part.text for part in message.content if isinstance(part, ChatContentText) and part.text)
        if text:
            return text
    if message.refusal is not None and message.refusal.strip():
        return message.refusal.strip()
    raise RuntimeError("vision model returned no text output")


def _vision_reasoning_content(message: ChatMessage) -> str | None:
    raw = message.memory.get(VISION_TOOL_NAME)
    if not isinstance(raw, Mapping):
        return None
    reasoning_content = raw.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else None


def _tool_output_message(result: ChatCompletionResult, *, tool_call_id: str) -> ChatMessage:
    reasoning_content = result.message.reasoning_content
    annotation: dict[str, object] = {"status": "completed"}
    if reasoning_content is not None:
        annotation["reasoning_content"] = reasoning_content
    return ChatMessage(
        role="tool",
        tool_call_id=tool_call_id,
        content=_tool_output_text(result),
        memory={VISION_TOOL_NAME: annotation},
    )


def _vision_turn_prompt(ids: list[str], prompt: str) -> str:
    return f"Selected image ids: {', '.join(ids)}\nQuestion: {prompt}"


def _vision_content(ids: list[str], images: list[ChatContentImage]) -> list[ChatContentPart]:
    content: list[ChatContentPart] = []
    for image_id, image in zip(ids, images, strict=True):
        content.append(ChatContentText(text=image_id))
        content.append(image)
    return content


def _vision_context(messages: list[Message]) -> tuple[list[ChatMessage], set[str]]:
    transcript: list[ChatMessage] = []
    known_image_ids: set[str] = set()
    for message in messages:
        content = message.content
        if isinstance(content, list):
            images = [part for part in content if isinstance(part, ChatContentImage)]
            if images:
                ids = [_image_id(image) for image in images]
                known_image_ids.update(ids)
                transcript.append(ChatMessage(role="user", content=_vision_content(ids, images)))
        if not message.is_tool():
            continue
        record = message.memory.get("server_tools")
        if not isinstance(record, Mapping) or record.get("tool") != VISION_TOOL_NAME:
            continue
        encoded = record.get("arguments")
        if not isinstance(encoded, str):
            continue
        ids, prompt = _tool_args(encoded)
        transcript.append(ChatMessage(role="user", content=_vision_turn_prompt(ids, prompt)))
        transcript.append(ChatMessage(role="assistant", content=message.content, reasoning_content=_vision_reasoning_content(message)))
    return transcript, known_image_ids


def _vision_history_messages(messages: list[Message]) -> list[ChatMessage]:
    transcript, _known_image_ids = _vision_context(messages)
    return transcript


def _vision_request(
    config: CueBox,
    request: ResponseCreateRequest,
    transcript: list[ChatMessage],
    ids: list[str],
    prompt: str,
) -> ChatCompletionRequest:
    return build_chat_request(
        config.vision,
        request,
        messages=[
            ChatMessage(role="developer", content=VISION_PROMPT),
            *transcript,
            ChatMessage(role="user", content=_vision_turn_prompt(ids, prompt)),
        ],
    )


@server_tools.register
class VisionTool(ServerTool):
    name: str = VISION_TOOL_NAME
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Image ids to inspect.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Question for the vision model about the selected images.",
                },
            },
            "required": ["ids", "prompt"],
            "additionalProperties": False,
        }
    )
    strict: bool = True
    description: str = (
        "Ask a question about images present in the conversation. "
        "Use ids like image-XXXX-XXXX-XXXX-XXXX in the ids field to select the relevant images, and use the prompt field "
        "to describe what to inspect."
    )

    async def __call__(
        self,
        state: State,
        call: ChatToolCall,
    ) -> ChatMessage:
        ids, prompt = _tool_args(call.arguments)
        transcript, known_image_ids = _vision_context(state.threads["main"].messages)
        missing = [image_id for image_id in ids if image_id not in known_image_ids]
        if missing:
            return ChatMessage(
                role="tool",
                tool_call_id=call.id,
                content=f"Unknown image IDs: {', '.join(missing)}.",
                memory={
                    VISION_TOOL_NAME: {
                        "status": "failed",
                        "reason": "unknown_image_ids",
                    }
                },
            )

        client = await state.svcs.aget(BudgetedChatCompletionClient)
        result = await client.complete(_vision_request(state.config, state.request, transcript, ids, prompt))
        return _tool_output_message(result, tool_call_id=call.id)


@bus.listen("response.request")
async def inject_images(state: State, *, next) -> ChatCompletionRequest:
    return _rewrite_request(await next(state=state))
