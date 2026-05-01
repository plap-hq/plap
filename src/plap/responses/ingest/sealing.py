from __future__ import annotations

import base64
from typing import Any

import msgspec
import zstandard as zstd
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESSIV
from nacl.exceptions import CryptoError
from nacl.secret import Aead

from plap.keyring import SealingKeyring, associated_data, purpose_label
from plap.responses.errors import ResponseError
from plap.responses.models import (
    ChatMessageSpan,
    CompactionPayload,
    ReasoningPayload,
    SealedCallID,
    Side,
    StateMessage,
)

COMPACTION_PURPOSE = "responses.ingest.compaction"
REASONING_PURPOSE = "responses.ingest.reasoning"
CALL_ID_PURPOSE = "responses.ingest.call_id"
CALL_ID_PREFIX = "call_"
PAYLOAD_FORMAT_VERSION = 1
COMPACTION_PAYLOAD_TYPE = "compaction"
REASONING_PAYLOAD_TYPE = "reasoning"
CALL_ID_FORMAT_VERSION = 1
CALL_ID_CONTENT_HASH_PREFIX_BYTES = 8
TAG_BYTES = 16
_CALL_ID_VERSION_SHIFT = 4
_CALL_ID_VERSION_MASK = 0xF0
_CALL_ID_SIDE_MASK = 0x0F
_SIDE_CODES: dict[Side, int] = {Side.MAIN: 1, Side.REVIEWER: 2, Side.ARBITRATOR: 3}
_SIDES = {code: side for side, code in _SIDE_CODES.items()}


def seal_compaction_payload(
    payload: CompactionPayload,
    *,
    keyring: SealingKeyring,
) -> str:
    return _xchacha_seal(
        _json_encode(_compaction_to_json(payload)),
        purpose=COMPACTION_PURPOSE,
        keyring=keyring,
    )


def open_compaction_payload(
    value: str,
    *,
    keyring: SealingKeyring,
) -> CompactionPayload:
    return _compaction_from_json(_json_decode(_xchacha_open(value, purpose=COMPACTION_PURPOSE, keyring=keyring)))


def seal_reasoning_payload(
    payload: ReasoningPayload,
    *,
    keyring: SealingKeyring,
) -> str:
    return _xchacha_seal(
        _json_encode(_reasoning_to_json(payload)),
        purpose=REASONING_PURPOSE,
        keyring=keyring,
    )


def open_reasoning_payload(
    value: str,
    *,
    keyring: SealingKeyring,
) -> ReasoningPayload:
    return _reasoning_from_json(_json_decode(_xchacha_open(value, purpose=REASONING_PURPOSE, keyring=keyring)))


def seal_call_id(
    value: SealedCallID,
    *,
    keyring: SealingKeyring,
) -> str:
    plaintext = _pack_call_id(value)
    ciphertext = AESSIV(keyring.active(CALL_ID_PURPOSE)).encrypt(
        plaintext,
        associated_data(CALL_ID_PURPOSE),
    )
    return CALL_ID_PREFIX + _b64url_encode(ciphertext)


def open_call_id(
    value: str,
    *,
    keyring: SealingKeyring,
) -> SealedCallID:
    if not value.startswith(CALL_ID_PREFIX):
        raise ResponseError.ingestion(private_message="invalid sealed function call id prefix")
    payload = _b64url_decode(value[len(CALL_ID_PREFIX) :])
    if len(payload) <= TAG_BYTES:
        raise ResponseError.ingestion(private_message="sealed function call id payload is too short")
    last_error: Exception | None = None
    for key in keyring.candidates(CALL_ID_PURPOSE):
        try:
            plaintext = AESSIV(key).decrypt(
                payload,
                associated_data(CALL_ID_PURPOSE),
            )
            return _unpack_call_id(plaintext)
        except (InvalidTag, ResponseError) as exc:
            last_error = exc
    raise ResponseError.ingestion(private_message="sealed function call id could not be opened", cause=last_error) from last_error


def content_hash(message: StateMessage) -> str:
    return message.content_hash()


def content_hash_prefix(value: str) -> bytes:
    try:
        payload = bytes.fromhex(value)
    except ValueError as exc:
        raise ResponseError.ingestion(private_message="content_hash must be lowercase hex", cause=exc) from exc
    if len(payload) < CALL_ID_CONTENT_HASH_PREFIX_BYTES:
        raise ResponseError.ingestion(private_message="content_hash is too short")
    return payload[:CALL_ID_CONTENT_HASH_PREFIX_BYTES]


def _xchacha_seal(value: bytes, *, purpose: str, keyring: SealingKeyring) -> str:
    compressed = zstd.ZstdCompressor().compress(value)
    encrypted = Aead(keyring.active(purpose)).encrypt(compressed, _aad(purpose))
    return _b64url_encode(bytes(encrypted))


def _xchacha_open(value: str, *, purpose: str, keyring: SealingKeyring) -> bytes:
    encrypted = _b64url_decode(value)
    last_error: Exception | None = None
    for key in keyring.candidates(purpose):
        try:
            compressed = Aead(key).decrypt(encrypted, _aad(purpose))
            return zstd.ZstdDecompressor().decompress(compressed)
        except (CryptoError, ValueError, zstd.ZstdError) as exc:
            last_error = exc
    raise ResponseError.ingestion(private_message=f"sealed {purpose} payload could not be opened", cause=last_error) from last_error


def _aad(purpose: str) -> bytes:
    return purpose_label(purpose)


def _json_encode(value: object) -> bytes:
    return msgspec.json.encode(value, order="deterministic")


def _json_decode(value: bytes) -> object:
    try:
        return msgspec.json.decode(value)
    except msgspec.DecodeError as exc:
        raise ResponseError.ingestion(private_message="sealed payload is not valid JSON", cause=exc) from exc


def _compaction_to_json(value: CompactionPayload) -> dict[str, Any]:
    return {
        "version": PAYLOAD_FORMAT_VERSION,
        "type": COMPACTION_PAYLOAD_TYPE,
        "active": [row.to_primitive() for row in value.active],
        "cursors": value.cursors,
    }


def _compaction_from_json(value: object) -> CompactionPayload:
    if not isinstance(value, dict):
        raise ResponseError.ingestion(private_message="compaction payload must be an object")
    if value.get("version") != PAYLOAD_FORMAT_VERSION or value.get("type") != COMPACTION_PAYLOAD_TYPE:
        raise ResponseError.ingestion(private_message="unsupported compaction payload")
    try:
        payload = CompactionPayload.from_primitive(
            {
                "active": value.get("active"),
                "cursors": value.get("cursors"),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ResponseError.ingestion(private_message=str(exc), cause=exc) from exc
    cursors = payload.cursors
    if not isinstance(cursors, dict):
        raise ResponseError.ingestion(private_message="compaction cursors are required")
    if "m" not in cursors:
        raise ResponseError.ingestion(private_message="compaction cursors must include m")
    parsed_cursors = {"m": int(cursors["m"])}
    if parsed_cursors["m"] < 0:
        raise ResponseError.ingestion(private_message="compaction cursors must be non-negative")
    _validate_active_rows(payload.active, parsed_cursors)
    return CompactionPayload(active=payload.active, cursors=parsed_cursors)


def _reasoning_to_json(value: ReasoningPayload) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": PAYLOAD_FORMAT_VERSION,
        "type": REASONING_PAYLOAD_TYPE,
        **value.to_primitive(),
    }
    return payload


def _reasoning_from_json(value: object) -> ReasoningPayload:
    if not isinstance(value, dict):
        raise ResponseError.ingestion(private_message="reasoning payload must be an object")
    if value.get("version") != PAYLOAD_FORMAT_VERSION or value.get("type") != REASONING_PAYLOAD_TYPE:
        raise ResponseError.ingestion(private_message="unsupported reasoning payload")
    side = value.get("side")
    if side not in {Side.MAIN, Side.REVIEWER, Side.ARBITRATOR, "main", "reviewer", "arbitrator"}:
        raise ResponseError.ingestion(private_message="invalid reasoning side")
    temp = value.get("temp")
    if not isinstance(temp, bool):
        raise ResponseError.ingestion(private_message="reasoning temp flag is required")
    try:
        return ReasoningPayload.from_primitive(
            {
                "side": side,
                "temp": temp,
                "messages": value.get("messages"),
                "continuation_side": value.get("continuation_side"),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ResponseError.ingestion(private_message=str(exc), cause=exc) from exc


def _pack_call_id(value: SealedCallID) -> bytes:
    if value.side not in {Side.MAIN, Side.REVIEWER, Side.ARBITRATOR}:
        raise ResponseError.ingestion(private_message="invalid function call side")
    if len(value.content_hash_prefix) != CALL_ID_CONTENT_HASH_PREFIX_BYTES:
        raise ResponseError.ingestion(private_message="function call content_hash prefix is invalid")
    if value.tool_call_index < 0:
        raise ResponseError.ingestion(private_message="tool_call_index must be non-negative")
    if not value.upstream_tool_call_id:
        raise ResponseError.ingestion(private_message="upstream_tool_call_id is required")
    header = (CALL_ID_FORMAT_VERSION << _CALL_ID_VERSION_SHIFT) | _SIDE_CODES[value.side]
    return b"".join(
        (
            bytes([header]),
            value.content_hash_prefix,
            _pack_uvarint(value.tool_call_index),
            value.upstream_tool_call_id.encode(),
        )
    )


def _unpack_call_id(value: bytes) -> SealedCallID:
    min_length = 1 + CALL_ID_CONTENT_HASH_PREFIX_BYTES + 1 + 1
    if len(value) < min_length:
        raise ResponseError.ingestion(private_message="function call id plaintext is too short")
    header = value[0]
    version = (header & _CALL_ID_VERSION_MASK) >> _CALL_ID_VERSION_SHIFT
    if version != CALL_ID_FORMAT_VERSION:
        raise ResponseError.ingestion(private_message="unsupported function call id version")
    try:
        side = _SIDES[header & _CALL_ID_SIDE_MASK]
    except KeyError as exc:
        raise ResponseError.ingestion(private_message="invalid function call id side", cause=exc) from exc
    content_hash_prefix_value = value[1 : 1 + CALL_ID_CONTENT_HASH_PREFIX_BYTES]
    tool_call_index, offset = _unpack_uvarint(value, 1 + CALL_ID_CONTENT_HASH_PREFIX_BYTES)
    return SealedCallID(
        side=side,
        content_hash_prefix=content_hash_prefix_value,
        tool_call_index=tool_call_index,
        upstream_tool_call_id=_decode_upstream_id(value[offset:]),
    )


def _pack_uvarint(value: int) -> bytes:
    if value < 0:
        raise ResponseError.ingestion(private_message="tool_call_index must be non-negative")
    chunks: list[int] = []
    remaining = value
    while True:
        byte = remaining & 0x7F
        remaining >>= 7
        if remaining:
            chunks.append(byte | 0x80)
        else:
            chunks.append(byte)
            return bytes(chunks)


def _unpack_uvarint(value: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    for index in range(offset, len(value)):
        byte = value[index]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, index + 1
        shift += 7
        if shift > 63:
            raise ResponseError.ingestion(private_message="function call id index is too large")
    raise ResponseError.ingestion(private_message="function call id index is truncated")


def _decode_upstream_id(value: bytes) -> str:
    if not value:
        raise ResponseError.ingestion(private_message="upstream_tool_call_id is required")
    try:
        return value.decode()
    except UnicodeDecodeError as exc:
        raise ResponseError.ingestion(private_message="upstream_tool_call_id is not UTF-8", cause=exc) from exc


def _validate_active_rows(rows: tuple[ChatMessageSpan, ...], cursors: dict[str, int]) -> None:
    _validate_span_rows(rows, cursors=cursors, parent=None)


def _validate_span_rows(
    rows: tuple[ChatMessageSpan, ...],
    *,
    cursors: dict[str, int],
    parent: ChatMessageSpan | None,
) -> None:
    previous_end = -1
    expected_start = parent.start if parent is not None else None
    for row in rows:
        if row.start <= previous_end:
            raise ResponseError.ingestion(private_message="compaction active spans overlap or are out of order")
        if expected_start is not None and row.start != expected_start:
            raise ResponseError.ingestion(private_message="compaction child spans do not cover parent")
        if row.end >= cursors["m"]:
            raise ResponseError.ingestion(private_message="compaction active span is outside cursor")
        _validate_span_node(row, cursors=cursors)
        previous_end = row.end
        if expected_start is not None:
            # Spans are inclusive, so the next contiguous child starts at end + 1.
            expected_start = row.end + 1
    if parent is not None and previous_end != parent.end:
        raise ResponseError.ingestion(private_message="compaction child spans do not cover parent")


def _validate_span_node(row: ChatMessageSpan, *, cursors: dict[str, int]) -> None:
    if row.children:
        if row.summary_fidelity is None:
            raise ResponseError.ingestion(private_message="compaction summary_fidelity is required")
        if row.children_pruned:
            raise ResponseError.ingestion(private_message="compaction span cannot have children and be pruned")
        _validate_span_rows(row.children, cursors=cursors, parent=row)
        children_token_count = sum(child.token_count for child in row.children)
        expanded_token_count = sum(child.expanded_token_count for child in row.children)
        if row.children_token_count != children_token_count:
            raise ResponseError.ingestion(private_message="compaction children_token_count is invalid")
        if row.expanded_token_count != expanded_token_count:
            raise ResponseError.ingestion(private_message="compaction expanded_token_count is invalid")
        return

    if row.is_leaf:
        if row.summary_fidelity is not None:
            raise ResponseError.ingestion(private_message="compaction leaf span cannot have summary_fidelity")
        if row.children_pruned:
            raise ResponseError.ingestion(private_message="compaction leaf span cannot be pruned")
        return

    if row.summary_fidelity is None:
        raise ResponseError.ingestion(private_message="compaction summary_fidelity is required")
    if not row.children_pruned:
        raise ResponseError.ingestion(private_message="compaction summary span has no children")


def _rows_from_json(value: object) -> tuple[ChatMessageSpan, ...]:
    if not isinstance(value, list):
        raise ResponseError.ingestion(private_message="message rows must be an array")
    rows: list[ChatMessageSpan] = []
    for row in value:
        if not isinstance(row, dict):
            raise ResponseError.ingestion(private_message="message row must be an object")
        start = row.get("start")
        end = row.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ResponseError.ingestion(private_message="message row span is invalid")
        if start < 0 or end < 0 or start > end:
            raise ResponseError.ingestion(private_message="message row span is invalid")
        try:
            rows.append(ChatMessageSpan.from_primitive(row))
        except (TypeError, ValueError) as exc:
            raise ResponseError.ingestion(private_message=str(exc), cause=exc) from exc
    return tuple(rows)


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64url_decode(value: str) -> bytes:
    if not value:
        raise ResponseError.ingestion(private_message="sealed payload is empty")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except ValueError as exc:
        raise ResponseError.ingestion(private_message="sealed payload is not base64url", cause=exc) from exc
