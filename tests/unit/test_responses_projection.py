import msgspec
import pytest

from plap.errors import PlapError
from plap.responses.contracts import (
    ResponseCreateRequest,
    ResponseOutputItemAddedEvent,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
)
from plap.responses.projection import STREAM_OBFUSCATION_BUCKET_SIZE, ResponseProjection


def test_response_projection_rejects_store_false_without_reasoning_encrypted_content_include() -> None:
    request = ResponseCreateRequest(model="plap/test", input="hello", store=False)
    projection = ResponseProjection.from_create_request(request)

    with pytest.raises(PlapError) as exc_info:
        projection.validate_create_request(request)

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "missing_reasoning_encrypted_content_include"
    assert exc_info.value.public.param == "include"


def test_response_projection_stream_payload_adds_obfuscation_for_delta_events() -> None:
    projection = ResponseProjection.from_create_request(
        ResponseCreateRequest(model="plap/test", input="hello", stream=True),
        transport="stream",
    )
    payload = projection.stream_payload(
        ResponseTextDeltaEvent(
            content_index=0,
            delta="hello",
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        ),
        sequence_number=7,
    )

    assert payload["sequence_number"] == 7
    assert isinstance(payload["obfuscation"], str)
    assert len(msgspec.json.encode(payload)) % STREAM_OBFUSCATION_BUCKET_SIZE == 0
    assert len(msgspec.json.encode(payload)) >= STREAM_OBFUSCATION_BUCKET_SIZE


def test_response_projection_stream_payload_omits_obfuscation_when_disabled() -> None:
    projection = ResponseProjection.from_create_request(
        ResponseCreateRequest(
            model="plap/test",
            input="hello",
            stream=True,
            stream_options={"include_obfuscation": False},
        ),
        transport="stream",
    )
    payload = projection.stream_payload(
        ResponseTextDeltaEvent(
            content_index=0,
            delta="hello",
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        ),
        sequence_number=7,
    )

    assert "obfuscation" not in payload


def test_response_projection_stream_payload_only_obfuscates_delta_events() -> None:
    projection = ResponseProjection.from_create_request(
        ResponseCreateRequest(model="plap/test", input="hello", stream=True),
        transport="stream",
    )
    payload = projection.stream_payload(
        ResponseTextDoneEvent(
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=0,
            text="hello",
            type="response.output_text.done",
        ),
        sequence_number=8,
    )

    assert "obfuscation" not in payload


def test_response_projection_stream_payload_redacts_reasoning_encrypted_content() -> None:
    projection = ResponseProjection.from_create_request(ResponseCreateRequest(model="plap/test", input="hello"))
    payload = projection.stream_payload(
        ResponseOutputItemAddedEvent(
            item=ResponseReasoningItem(
                encrypted_content="sealed",
                id="rs_1",
                status="completed",
                summary=[],
                type="reasoning",
            ),
            output_index=0,
            sequence_number=0,
            type="response.output_item.added",
        ),
        sequence_number=3,
    )

    assert payload["item"].get("encrypted_content") is None
