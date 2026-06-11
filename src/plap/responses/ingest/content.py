from __future__ import annotations

from plap.llms.completions.chat import (
    ChatContentFile,
    ChatContentImage,
    ChatContentPart,
    ChatContentText,
    ChatFile,
    ChatImageURL,
    ChatMessage,
)
from plap.responses.contracts import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
    OutputRefusalContent,
    OutputTextContent,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
)


def _part(value: object) -> ChatContentPart:
    if isinstance(value, InputTextContent):
        return ChatContentText(text=value.text)
    if isinstance(value, InputImageContent):
        return ChatContentImage(
            image_url=ChatImageURL(
                url=value.image_url,
                detail=value.detail,
                file_id=value.file_id,
            )
        )
    if isinstance(value, InputFileContent):
        return ChatContentFile(
            file=ChatFile(
                file_data=value.file_data,
                file_id=value.file_id,
                filename=value.filename,
                file_url=value.file_url,
                detail=value.detail,
            )
        )
    if isinstance(value, OutputTextContent):
        return ChatContentText(text=value.text)
    raise TypeError(f"unsupported content part: {type(value).__name__}")


def _parts(values: list[object]) -> list[ChatContentPart]:
    return [_part(value) for value in values]


def _assistant_message_content(value: str | list[object]) -> tuple[str | list[ChatContentPart], str | None]:
    if isinstance(value, str):
        return value, None
    parts: list[ChatContentPart] = []
    refusals: list[str] = []
    for part in value:
        if isinstance(part, OutputTextContent):
            parts.append(ChatContentText(text=part.text))
            continue
        if isinstance(part, OutputRefusalContent):
            refusals.append(part.refusal)
            continue
        raise TypeError(f"unsupported assistant message content part: {type(part).__name__}")
    refusal = "\n".join(refusals) if refusals else None
    return parts, refusal


def message(item: RequestMessageItem) -> ChatMessage:
    if item.role == "assistant":
        content, refusal = _assistant_message_content(item.content)
    else:
        if isinstance(item.content, str):
            content, refusal = item.content, None
        else:
            content, refusal = _parts(item.content), None
    return ChatMessage(role=item.role, content=content, refusal=refusal)


def tool_output(item: RequestFunctionCallOutputItem) -> str | list[ChatContentPart]:
    if isinstance(item.output, str):
        return item.output
    return _parts(item.output)


__all__ = ["message", "tool_output"]
