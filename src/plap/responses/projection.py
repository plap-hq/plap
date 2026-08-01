from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import msgspec

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.responses.contracts import (
    InputItemsPage,
    InputItemsPageItem,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseCreateRequest,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseInProgressEvent,
    ResponseObject,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseStreamEvent,
)

REASONING_ENCRYPTED_CONTENT_INCLUDE = "reasoning.encrypted_content"
STREAM_OBFUSCATION_BUCKET_SIZE = 512
_OBFUSCATION_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


type ResponseTransport = Literal["snapshot", "stream"]


def _stream_obfuscation_enabled(request: ResponseCreateRequest, *, transport: ResponseTransport) -> bool:
    if transport != "stream":
        return False
    return request.stream_options is None or request.stream_options.include_obfuscation is not False


def _project_reasoning_item(item: ResponseReasoningItem, *, include_reasoning_encrypted_content: bool) -> ResponseReasoningItem:
    if include_reasoning_encrypted_content or item.encrypted_content is None:
        return item
    return item.model_copy(update={"encrypted_content": None})


def _is_delta_event_type(event_type: str) -> bool:
    return event_type.endswith(".delta")


def _payload_length(payload: dict[str, object]) -> int:
    return len(msgspec.json.encode(payload))


def _obfuscation_target_length(length: int) -> int:
    return max(
        STREAM_OBFUSCATION_BUCKET_SIZE,
        ((length + STREAM_OBFUSCATION_BUCKET_SIZE - 1) // STREAM_OBFUSCATION_BUCKET_SIZE) * STREAM_OBFUSCATION_BUCKET_SIZE,
    )


def _random_obfuscation(length: int) -> str:
    return "".join(secrets.choice(_OBFUSCATION_ALPHABET) for _ in range(length))


def _payload_with_obfuscation(payload: dict[str, object]) -> dict[str, object]:
    obfuscated = dict(payload)
    obfuscated["obfuscation"] = ""
    current_length = _payload_length(obfuscated)
    target_length = _obfuscation_target_length(current_length)
    padding_length = target_length - current_length
    if padding_length > 0:
        obfuscated["obfuscation"] = _random_obfuscation(padding_length)
    return obfuscated


@dataclass(frozen=True, slots=True)
class ResponseProjection:
    include: frozenset[str] = frozenset()
    include_obfuscation: bool = False

    @classmethod
    def from_create_request(
        cls,
        request: ResponseCreateRequest,
        *,
        transport: ResponseTransport = "snapshot",
    ) -> ResponseProjection:
        return cls(
            include=frozenset(request.include or ()),
            include_obfuscation=_stream_obfuscation_enabled(request, transport=transport),
        )

    @classmethod
    def from_query(
        cls,
        include: Sequence[str] | None,
        *,
        include_obfuscation: bool | None = None,
    ) -> ResponseProjection:
        return cls(
            include=frozenset(include or ()),
            include_obfuscation=bool(include_obfuscation),
        )

    @property
    def include_reasoning_encrypted_content(self) -> bool:
        return REASONING_ENCRYPTED_CONTENT_INCLUDE in self.include

    def validate_create_request(self, request: ResponseCreateRequest) -> None:
        if request.store is False and not self.include_reasoning_encrypted_content:
            raise PlapError(
                public=PublicError(
                    status_code=400,
                    type="invalid_request_error",
                    code="missing_reasoning_encrypted_content_include",
                    message=(f"Requests with 'store' disabled must include '{REASONING_ENCRYPTED_CONTENT_INCLUDE}' in 'include'."),
                    param="include",
                ),
                private=PrivateError(
                    event="response.invalid_request",
                    reason="store_false_requires_reasoning_encrypted_content_include",
                    message="store=false requires include reasoning.encrypted_content",
                    level=ErrorLevel.WARNING,
                    context={"param": "include"},
                ),
            )

    def output_item(self, item: ResponseOutputItem) -> ResponseOutputItem:
        if not isinstance(item, ResponseReasoningItem):
            return item
        return _project_reasoning_item(
            item,
            include_reasoning_encrypted_content=self.include_reasoning_encrypted_content,
        )

    def input_item(self, item: InputItemsPageItem) -> InputItemsPageItem:
        if not isinstance(item, ResponseReasoningItem):
            return item
        return _project_reasoning_item(
            item,
            include_reasoning_encrypted_content=self.include_reasoning_encrypted_content,
        )

    def response(self, response: ResponseObject) -> ResponseObject:
        if self.include_reasoning_encrypted_content:
            return response
        return response.model_copy(update={"output": [self.output_item(item) for item in response.output]})

    def input_items_page(self, page: InputItemsPage) -> InputItemsPage:
        if self.include_reasoning_encrypted_content:
            return page
        return page.model_copy(update={"data": [self.input_item(item) for item in page.data]})

    def stream_payload(self, event: ResponseStreamEvent) -> dict[str, object]:
        payload = event.model_dump(mode="json", exclude_none=True)
        if isinstance(
            event,
            ResponseCreatedEvent | ResponseInProgressEvent | ResponseCompletedEvent | ResponseFailedEvent | ResponseIncompleteEvent,
        ):
            payload["response"] = self.response(event.response).model_dump(mode="json", exclude_none=True)
        elif isinstance(event, ResponseOutputItemAddedEvent | ResponseOutputItemDoneEvent):
            payload["item"] = self.output_item(event.item).model_dump(mode="json", exclude_none=True)
        if self.include_obfuscation and _is_delta_event_type(str(payload["type"])):
            return _payload_with_obfuscation(payload)
        return payload
