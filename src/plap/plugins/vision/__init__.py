from __future__ import annotations

import base64
import binascii
from dataclasses import replace
from pathlib import Path

import blake3
import msgspec

from plap.bus import bus
from plap.config import CueBox
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatContentImage,
    ChatContentPart,
    ChatContentText,
    ChatFunctionTool,
    ChatMessage,
    ChatTool,
    IChatCompletionClient,
)
from plap.llms.json import decode_json_object_or_none
from plap.llms.retry import RetryValidator, retry_message
from plap.plugins.core.ledger import UsageLedger
from plap.plugins.core.loop import StreamResult
from plap.responses.ingest.models import MAIN_SIDE, Message
from plap.responses.state import State

VISION_TOOL_NAME = "images"
VISION_TOOL = ChatTool(
    function=ChatFunctionTool(
        name=VISION_TOOL_NAME,
        description=(
            "Inspect one or more image ids from the conversation. "
            "Use ids like image-XXXX-XXXX-XXXX-XXXX and provide a prompt describing what to inspect."
        ),
        parameters={
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Image ids to inspect.",
                },
                "prompt": {
                    "type": "string",
                    "description": "Question for the selected images.",
                },
            },
            "required": ["ids", "prompt"],
            "additionalProperties": False,
        },
        strict=True,
    )
)
VISION_PROMPT = (
    "You answer questions about referenced images. Any standalone text part matching image-XXXX-XXXX-XXXX-XXXX "
    "identifies the image part immediately following it. Use those ids to keep image references precise. "
    "Answer only the user's request in plain text."
)


def _detail_rank(detail: str | None) -> int:
    return {
        None: 0,
        "low": 1,
        "auto": 2,
        "high": 3,
        "original": 4,
    }.get(detail, 0)


def _best_image(left: ChatContentImage, right: ChatContentImage) -> ChatContentImage:
    if _detail_rank(right.image_url.detail) <= _detail_rank(left.image_url.detail):
        return left
    return replace(left, image_url=replace(left.image_url, detail=right.image_url.detail))


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
    return replace(request, messages=messages, tools=[*request.tools, VISION_TOOL])


def _images_by_id(messages: list[Message]) -> dict[str, ChatContentImage]:
    images: dict[str, ChatContentImage] = {}
    for message in messages:
        content = message.content
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, ChatContentImage):
                continue
            image_id = _image_id(part)
            existing = images.get(image_id)
            images[image_id] = part if existing is None else _best_image(existing, part)
    return images


def _tool_args_or_none(call) -> tuple[list[str], str] | None:
    arguments = decode_json_object_or_none(call.arguments)
    if arguments is None:
        return None
    ids = arguments.get("ids")
    prompt = arguments.get("prompt")
    if not isinstance(ids, list) or not ids or not all(isinstance(image_id, str) and image_id for image_id in ids):
        return None
    if not isinstance(prompt, str) or not prompt:
        return None
    return list(ids), prompt


def _tool_args(call) -> tuple[list[str], str]:
    arguments = msgspec.json.decode(call.arguments)
    if not isinstance(arguments, dict):
        raise TypeError("images tool arguments must decode to an object")
    ids = arguments.get("ids")
    prompt = arguments.get("prompt")
    if not isinstance(ids, list) or not ids or not all(isinstance(image_id, str) and image_id for image_id in ids):
        raise RuntimeError("images tool arguments must include a non-empty ids array")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("images tool arguments must include a non-empty prompt")
    return list(ids), prompt


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


def _unknown_images(ids: list[str], images: dict[str, ChatContentImage]) -> list[str]:
    return [image_id for image_id in ids if image_id not in images]


def _selected_images(ids: list[str], images: dict[str, ChatContentImage]) -> list[ChatContentImage]:
    return [images[image_id] for image_id in ids if image_id in images]


def _missing_images_retry_message(ids: list[str]) -> str:
    joined = ", ".join(sorted(ids))
    return retry_message(
        problems=(f"You requested unknown image ids: {joined}.",),
        rules=("Use only image ids that already appear in the conversation.",),
    )


def _vision_content(ids: list[str], images: list[ChatContentImage]) -> list[ChatContentPart]:
    content: list[ChatContentPart] = []
    for image_id, image in zip(ids, images, strict=True):
        content.append(ChatContentText(text=image_id))
        content.append(image)
    return content


def _vision_request(config: CueBox, ids: list[str], prompt: str, images: list[ChatContentImage]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=config.vision.model,
        messages=[
            ChatMessage(role="developer", content=VISION_PROMPT),
            ChatMessage(role="user", content=_vision_content(ids, images)),
            ChatMessage(role="user", content=prompt),
        ],
        max_completion_tokens=config.vision.max_completion_tokens,
        reasoning_effort=config.vision.reasoning_effort,
        service_tier=config.vision.service_tier,
    )


async def _vision_tool_output(
    state: State,
    config: CueBox,
    ledger: UsageLedger,
    *,
    ids: list[str],
    prompt: str,
) -> str:
    images = _images_by_id(state.history(MAIN_SIDE))
    missing = _unknown_images(ids, images)
    if missing:
        raise RuntimeError(f"unknown image ids reached execution: {', '.join(sorted(missing))}")
    selected = _selected_images(ids, images)
    if not selected:
        raise RuntimeError("images tool execution reached an empty image selection")
    client = await state.svcs.aget(IChatCompletionClient)
    result = await client.complete(_vision_request(config, ids, prompt, selected))
    ledger.hide(config.vision.public_usage, result.usage)
    return _tool_output_text(result)


def _images_validator(state: State) -> RetryValidator:
    async def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
        images = _images_by_id(state.history(MAIN_SIDE))
        for call in result.message.tool_calls:
            if call.name != VISION_TOOL_NAME:
                continue
            arguments = _tool_args_or_none(call)
            if arguments is None:
                continue
            ids, _prompt = arguments
            missing = _unknown_images(ids, images)
            if missing:
                return _missing_images_retry_message(missing)
        return None

    return validate


@bus.listen("config.collect")
async def collect(paths: tuple[str, ...], *, next):
    here = Path(__file__).resolve()
    return await next(paths=(*paths, str(here.parent / "schema.cue")))


@bus.listen("response.request")
async def inject_images(state: State, config: CueBox, *, next) -> ChatCompletionRequest:
    _ = state, config
    return _rewrite_request(await next(state=state, config=config))


@bus.listen("response.validate")
async def validate_images(
    state: State,
    config: CueBox,
    validators: tuple[RetryValidator, ...],
    *,
    next,
) -> tuple[RetryValidator, ...]:
    _ = config
    return await next(state=state, config=config, validators=(*validators, _images_validator(state)))


@bus.listen("response.loop")
async def run_images(state: State, config: CueBox, ledger: UsageLedger, *, next) -> StreamResult:
    result = await next(state=state, config=config, ledger=ledger)
    accepted = result.accepted
    if accepted is None or accepted.finish_reason != "tool_calls":
        return result
    image_calls = [call for call in accepted.message.tool_calls if call.name == VISION_TOOL_NAME]
    for call in image_calls:
        ids, prompt = _tool_args(call)
        output = await _vision_tool_output(state, config, ledger, ids=ids, prompt=prompt)
        state.main.append(ChatMessage(role="tool", tool_call_id=call.id, content=output))
    return result
