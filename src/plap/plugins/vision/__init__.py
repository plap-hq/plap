from __future__ import annotations

import base64
import binascii
import re
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
from plap.plugins.core.request import apply_float_transform, apply_int_transform
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.ingest.models import MAIN_SIDE, Message
from plap.responses.state import State

VISION_TOOL_NAME = "vision"
VISION_TOOL = ChatTool(
    function=ChatFunctionTool(
        name=VISION_TOOL_NAME,
        description=(
            "Ask a question about images present in the conversation. "
            "Use ids like image-XXXX-XXXX-XXXX-XXXX in the ids field to select the relevant images, and use the prompt field "
            "to describe what to inspect."
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
                    "description": "Question for the vision model about the selected images.",
                },
            },
            "required": ["ids", "prompt"],
            "additionalProperties": False,
        },
        strict=True,
    )
)
VISION_PROMPT = (
    "Answer questions about images from the conversation. "
    "Any standalone text part matching image-XXXX-XXXX-XXXX-XXXX identifies the image part immediately following it. "
    "When a text message begins with 'Selected image ids:', use those ids to determine which images are relevant to the question. "
    "Answer only the user's request in plain text. Keep the answer concrete, visually grounded, and concise."
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
    return replace(request, messages=messages, tools=[*request.tools, VISION_TOOL])


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


def _tool_args_or_none(call) -> tuple[list[str], str] | None:
    arguments = decode_json_object_or_none(call.arguments)
    if arguments is None:
        return None
    ids = _normalize_image_ids(arguments.get("ids"))
    prompt = arguments.get("prompt")
    if ids is None:
        return None
    if not isinstance(prompt, str) or not prompt:
        return None
    return ids, prompt


def _tool_args(call) -> tuple[list[str], str]:
    arguments = msgspec.json.decode(call.arguments)
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


def _vision_turn_prompt(ids: list[str], prompt: str) -> str:
    return f"Selected image ids: {', '.join(ids)}\nQuestion: {prompt}"


def _vision_context(messages: list[Message]) -> tuple[list[ChatMessage], set[str]]:
    transcript: list[ChatMessage] = []
    known_image_ids: set[str] = set()
    pending_calls: dict[str, tuple[list[str], str]] = {}
    for message in messages:
        content = message.content
        if isinstance(content, list):
            images = [part for part in content if isinstance(part, ChatContentImage)]
            if images:
                ids = [_image_id(image) for image in images]
                known_image_ids.update(ids)
                transcript.append(ChatMessage(role="user", content=_vision_content(ids, images)))
        if message.is_assistant():
            for call in message.tool_calls:
                if call.name != VISION_TOOL_NAME:
                    continue
                arguments = _tool_args_or_none(call)
                if arguments is None:
                    continue
                pending_calls[call.id] = arguments
            continue
        if not message.is_tool() or message.tool_call_id is None:
            continue
        arguments = pending_calls.pop(message.tool_call_id, None)
        if arguments is None:
            continue
        ids, prompt = arguments
        transcript.append(ChatMessage(role="user", content=_vision_turn_prompt(ids, prompt)))
        transcript.append(ChatMessage(role="assistant", content=message.content))
    return transcript, known_image_ids


def _vision_history_messages(messages: list[Message]) -> list[ChatMessage]:
    transcript, _known_image_ids = _vision_context(messages)
    return transcript


def _vision_content(ids: list[str], images: list[ChatContentImage]) -> list[ChatContentPart]:
    content: list[ChatContentPart] = []
    for image_id, image in zip(ids, images, strict=True):
        content.append(ChatContentText(text=image_id))
        content.append(image)
    return content


def _vision_request(
    config: CueBox,
    request: ResponseCreateRequest,
    transcript: list[ChatMessage],
    ids: list[str],
    prompt: str,
) -> ChatCompletionRequest:
    sampling = config.vision.sampling
    return ChatCompletionRequest(
        model=config.vision.model,
        messages=[
            ChatMessage(role="developer", content=VISION_PROMPT),
            *transcript,
            ChatMessage(role="user", content=_vision_turn_prompt(ids, prompt)),
        ],
        max_completion_tokens=config.vision.max_completion_tokens,
        temperature=apply_float_transform(request.temperature, sampling.temperature, minimum=0, maximum=2),
        top_p=apply_float_transform(request.top_p, sampling.top_p, minimum=0, maximum=1),
        min_p=apply_float_transform(None, sampling.min_p, minimum=0, maximum=1),
        top_k=apply_int_transform(None, sampling.top_k, minimum=0),
        frequency_penalty=apply_float_transform(None, sampling.frequency_penalty, minimum=-2, maximum=2),
        presence_penalty=apply_float_transform(None, sampling.presence_penalty, minimum=-2, maximum=2),
        repetition_penalty=apply_float_transform(None, sampling.repetition_penalty, minimum=0, maximum=2),
        seed=apply_int_transform(None, sampling.seed),
        reasoning_effort=config.vision.reasoning_effort,
        service_tier=config.vision.service_tier,
    )


async def _vision_tool_output(
    state: State,
    config: CueBox,
    ledger: UsageLedger,
    *,
    history: list[Message],
    ids: list[str],
    prompt: str,
) -> str:
    transcript, known_image_ids = _vision_context(history)
    missing = [image_id for image_id in ids if image_id not in known_image_ids]
    if missing:
        raise RuntimeError(f"unknown image ids reached execution: {', '.join(sorted(missing))}")
    client = await state.svcs.aget(IChatCompletionClient)
    result = await client.complete(_vision_request(config, state.prepared.execution_request, transcript, ids, prompt))
    ledger.hide(config.vision.public_usage, result.usage)
    return _tool_output_text(result)


def _vision_validator(state: State) -> RetryValidator:
    async def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
        _transcript, known_image_ids = _vision_context(state.history(MAIN_SIDE))
        for call in result.message.tool_calls:
            if call.name != VISION_TOOL_NAME:
                continue
            arguments = _tool_args_or_none(call)
            if arguments is None:
                return retry_message(
                    problems=("The vision tool call must include image ids and a non-empty prompt.",),
                    rules=("Use the ids field to select one or more image ids and the prompt field to describe what to inspect.",),
                )
            ids, _prompt = arguments
            missing = [image_id for image_id in ids if image_id not in known_image_ids]
            if missing:
                joined = ", ".join(sorted(missing))
                return retry_message(
                    problems=(f"You requested unknown image ids: {joined}.",),
                    rules=("Use only image ids that already appear in the conversation.",),
                )
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
    return await next(state=state, config=config, validators=(*validators, _vision_validator(state)))


@bus.listen("response.loop")
async def run_images(state: State, config: CueBox, ledger: UsageLedger, *, next) -> StreamResult:
    result = await next(state=state, config=config, ledger=ledger)
    accepted = result.accepted
    if accepted is None or accepted.finish_reason != "tool_calls":
        return result
    history = state.history(MAIN_SIDE)
    image_calls = [call for call in accepted.message.tool_calls if call.name == VISION_TOOL_NAME]
    for call in image_calls:
        ids, prompt = _tool_args(call)
        output = await _vision_tool_output(state, config, ledger, history=history, ids=ids, prompt=prompt)
        tool_message = ChatMessage(role="tool", tool_call_id=call.id, content=output)
        state.main.append(tool_message)
        history = [*history, tool_message]
    return result
