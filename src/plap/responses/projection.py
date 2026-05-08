from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.responses.contracts import (
    InputItemsPage,
    InputItemsPageItem,
    ResponseCreateRequest,
    ResponseObject,
    ResponseOutputItem,
    ResponseReasoningItem,
)

REASONING_ENCRYPTED_CONTENT_INCLUDE = "reasoning.encrypted_content"


@dataclass(frozen=True, slots=True)
class ResponseProjection:
    include: frozenset[str] = frozenset()
    include_obfuscation: bool = False

    @classmethod
    def from_create_request(cls, request: ResponseCreateRequest) -> ResponseProjection:
        return cls(
            include=frozenset(request.include or ()),
            include_obfuscation=bool(request.stream_options and request.stream_options.include_obfuscation),
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
                    message=(
                        f"Requests with 'store' disabled must include '{REASONING_ENCRYPTED_CONTENT_INCLUDE}' in 'include'."
                    ),
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
        if (
            self.include_reasoning_encrypted_content
            or not isinstance(item, ResponseReasoningItem)
            or item.encrypted_content is None
        ):
            return item
        return item.model_copy(update={"encrypted_content": None})

    def input_item(self, item: InputItemsPageItem) -> InputItemsPageItem:
        if (
            self.include_reasoning_encrypted_content
            or not isinstance(item, ResponseReasoningItem)
            or item.encrypted_content is None
        ):
            return item
        return item.model_copy(update={"encrypted_content": None})

    def response(self, response: ResponseObject) -> ResponseObject:
        if self.include_reasoning_encrypted_content:
            return response
        return response.model_copy(update={"output": [self.output_item(item) for item in response.output]})

    def input_items_page(self, page: InputItemsPage) -> InputItemsPage:
        if self.include_reasoning_encrypted_content:
            return page
        return page.model_copy(update={"data": [self.input_item(item) for item in page.data]})
