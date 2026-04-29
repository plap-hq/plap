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
from plap.responses.ingest.types import (
    ChatMessageSpan,
    CompactionPayload,
    IngestionError,
    ReasoningPayload,
    SealedCallID,
    Side,
    chat_message_hash,
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
_SIDE_CODES: dict[Side, int] = {"main": 1, "reviewer": 2, "arbitrator": 3}
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
    return _compaction_from_json(
        _json_decode(_xchacha_open(value, purpose=COMPACTION_PURPOSE, keyring=keyring))
    )


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
    return _reasoning_from_json(
        _json_decode(_xchacha_open(value, purpose=REASONING_PURPOSE, keyring=keyring))
    )


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
        raise IngestionError("invalid sealed function call id prefix")
    payload = _b64url_decode(value[len(CALL_ID_PREFIX) :])
    if len(payload) <= TAG_BYTES:
        raise IngestionError("sealed function call id payload is too short")
    last_error: Exception | None = None
    for key in keyring.candidates(CALL_ID_PURPOSE):
        try:
            plaintext = AESSIV(key).decrypt(
                payload,
                associated_data(CALL_ID_PURPOSE),
            )
            return _unpack_call_id(plaintext)
        except (InvalidTag, IngestionError) as exc:
            last_error = exc
    raise IngestionError("sealed function call id could not be opened") from last_error


def content_hash(message: dict[str, Any]) -> str:
    return chat_message_hash(message)


def content_hash_prefix(value: str) -> bytes:
    try:
        payload = bytes.fromhex(value)
    except ValueError as exc:
        raise IngestionError("content_hash must be lowercase hex") from exc
    if len(payload) < CALL_ID_CONTENT_HASH_PREFIX_BYTES:
        raise IngestionError("content_hash is too short")
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
    raise IngestionError(
        f"sealed {purpose} payload could not be opened"
    ) from last_error


def _aad(purpose: str) -> bytes:
    return purpose_label(purpose)


def _json_encode(value: object) -> bytes:
    return msgspec.json.encode(value, order="deterministic")


def _json_decode(value: bytes) -> object:
    try:
        return msgspec.json.decode(value)
    except msgspec.DecodeError as exc:
        raise IngestionError("sealed payload is not valid JSON") from exc


def _compaction_to_json(value: CompactionPayload) -> dict[str, Any]:
    return {
        "version": PAYLOAD_FORMAT_VERSION,
        "type": COMPACTION_PAYLOAD_TYPE,
        "active": [_row_to_json(row) for row in value.active],
        "cursors": value.cursors,
    }


def _compaction_from_json(value: object) -> CompactionPayload:
    if not isinstance(value, dict):
        raise IngestionError("compaction payload must be an object")
    if (
        value.get("version") != PAYLOAD_FORMAT_VERSION
        or value.get("type") != COMPACTION_PAYLOAD_TYPE
    ):
        raise IngestionError("unsupported compaction payload")
    active = _rows_from_json(value.get("active"))
    cursors = value.get("cursors")
    if not isinstance(cursors, dict):
        raise IngestionError("compaction cursors are required")
    if "m" not in cursors:
        raise IngestionError("compaction cursors must include m")
    parsed_cursors = {"m": int(cursors["m"])}
    if parsed_cursors["m"] < 0:
        raise IngestionError("compaction cursors must be non-negative")
    _validate_active_rows(active, parsed_cursors)
    return CompactionPayload(active=active, cursors=parsed_cursors)


def _reasoning_to_json(value: ReasoningPayload) -> dict[str, Any]:
    return {
        "version": PAYLOAD_FORMAT_VERSION,
        "type": REASONING_PAYLOAD_TYPE,
        "side": value.side,
        "temp": value.temp,
        "messages": list(value.messages),
    }


def _reasoning_from_json(value: object) -> ReasoningPayload:
    if not isinstance(value, dict):
        raise IngestionError("reasoning payload must be an object")
    if (
        value.get("version") != PAYLOAD_FORMAT_VERSION
        or value.get("type") != REASONING_PAYLOAD_TYPE
    ):
        raise IngestionError("unsupported reasoning payload")
    side = value.get("side")
    if side not in {"main", "reviewer", "arbitrator"}:
        raise IngestionError("invalid reasoning side")
    temp = value.get("temp")
    if not isinstance(temp, bool):
        raise IngestionError("reasoning temp flag is required")
    messages = value.get("messages")
    if not isinstance(messages, list) or not all(
        isinstance(message, dict) for message in messages
    ):
        raise IngestionError("reasoning messages must be an array of objects")
    return ReasoningPayload(side=side, temp=temp, messages=tuple(messages))


def _pack_call_id(value: SealedCallID) -> bytes:
    if value.side not in {"main", "reviewer", "arbitrator"}:
        raise IngestionError("invalid function call side")
    if len(value.content_hash_prefix) != CALL_ID_CONTENT_HASH_PREFIX_BYTES:
        raise IngestionError("function call content_hash prefix is invalid")
    if value.tool_call_index < 0:
        raise IngestionError("tool_call_index must be non-negative")
    if not value.upstream_tool_call_id:
        raise IngestionError("upstream_tool_call_id is required")
    header = (CALL_ID_FORMAT_VERSION << _CALL_ID_VERSION_SHIFT) | _SIDE_CODES[
        value.side
    ]
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
        raise IngestionError("function call id plaintext is too short")
    header = value[0]
    version = (header & _CALL_ID_VERSION_MASK) >> _CALL_ID_VERSION_SHIFT
    if version != CALL_ID_FORMAT_VERSION:
        raise IngestionError("unsupported function call id version")
    try:
        side = _SIDES[header & _CALL_ID_SIDE_MASK]
    except KeyError as exc:
        raise IngestionError("invalid function call id side") from exc
    content_hash_prefix_value = value[1 : 1 + CALL_ID_CONTENT_HASH_PREFIX_BYTES]
    tool_call_index, offset = _unpack_uvarint(
        value, 1 + CALL_ID_CONTENT_HASH_PREFIX_BYTES
    )
    return SealedCallID(
        side=side,
        content_hash_prefix=content_hash_prefix_value,
        tool_call_index=tool_call_index,
        upstream_tool_call_id=_decode_upstream_id(value[offset:]),
    )


def _pack_uvarint(value: int) -> bytes:
    if value < 0:
        raise IngestionError("tool_call_index must be non-negative")
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
            raise IngestionError("function call id index is too large")
    raise IngestionError("function call id index is truncated")


def _decode_upstream_id(value: bytes) -> str:
    if not value:
        raise IngestionError("upstream_tool_call_id is required")
    try:
        return value.decode()
    except UnicodeDecodeError as exc:
        raise IngestionError("upstream_tool_call_id is not UTF-8") from exc


def _row_to_json(value: ChatMessageSpan) -> dict[str, Any]:
    return {
        "start": value.start,
        "end": value.end,
        "message": value.message,
        "token_count": value.token_count,
        "children_token_count": value.children_token_count,
        "expanded_token_count": value.expanded_token_count,
        "children_pruned": value.children_pruned,
        "children": [_row_to_json(child) for child in value.children],
    }


def _validate_active_rows(
    rows: tuple[ChatMessageSpan, ...], cursors: dict[str, int]
) -> None:
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
            raise IngestionError("compaction active spans overlap or are out of order")
        if expected_start is not None and row.start != expected_start:
            raise IngestionError("compaction child spans do not cover parent")
        if row.end >= cursors["m"]:
            raise IngestionError("compaction active span is outside cursor")
        _validate_span_node(row, cursors=cursors)
        previous_end = row.end
        if expected_start is not None:
            # Spans are inclusive, so the next contiguous child starts at end + 1.
            expected_start = row.end + 1
    if parent is not None and previous_end != parent.end:
        raise IngestionError("compaction child spans do not cover parent")


def _validate_span_node(
    row: ChatMessageSpan, *, cursors: dict[str, int]
) -> None:
    if row.children:
        if row.children_pruned:
            raise IngestionError("compaction span cannot have children and be pruned")
        _validate_span_rows(row.children, cursors=cursors, parent=row)
        children_token_count = sum(child.token_count for child in row.children)
        expanded_token_count = sum(
            child.expanded_token_count for child in row.children
        )
        if row.children_token_count != children_token_count:
            raise IngestionError("compaction children_token_count is invalid")
        if row.expanded_token_count != expanded_token_count:
            raise IngestionError("compaction expanded_token_count is invalid")
        return

    if row.is_leaf:
        if row.children_pruned:
            raise IngestionError("compaction leaf span cannot be pruned")
        return

    if not row.children_pruned:
        raise IngestionError("compaction summary span has no children")


def _rows_from_json(value: object) -> tuple[ChatMessageSpan, ...]:
    if not isinstance(value, list):
        raise IngestionError("message rows must be an array")
    rows: list[ChatMessageSpan] = []
    for row in value:
        if not isinstance(row, dict):
            raise IngestionError("message row must be an object")
        start = row.get("start")
        end = row.get("end")
        message = row.get("message")
        children = row.get("children", [])
        if not isinstance(start, int) or not isinstance(end, int):
            raise IngestionError("message row span is invalid")
        if start < 0 or end < 0 or start > end:
            raise IngestionError("message row span is invalid")
        if not isinstance(message, dict):
            raise IngestionError("message row message is required")
        rows.append(
            ChatMessageSpan(
                start=start,
                end=end,
                message=message,
                token_count=_positive_int(row, "token_count"),
                children_token_count=_non_negative_int(
                    row, "children_token_count"
                ),
                expanded_token_count=_non_negative_int(
                    row, "expanded_token_count"
                ),
                children=_rows_from_json(children),
                children_pruned=bool(row.get("children_pruned", False)),
            )
        )
    return tuple(rows)


def _non_negative_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if not isinstance(value, int) or value < 0:
        raise IngestionError(f"message row {key} is invalid")
    return value


def _positive_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or value <= 0:
        raise IngestionError(f"message row {key} is invalid")
    return value


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64url_decode(value: str) -> bytes:
    if not value:
        raise IngestionError("sealed payload is empty")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except ValueError as exc:
        raise IngestionError("sealed payload is not base64url") from exc
