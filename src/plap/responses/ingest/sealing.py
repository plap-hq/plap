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
    ChatMessageWithOrdinal,
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
        "source": [_row_to_json(row) for row in value.source],
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
    active = _rows_from_json(value.get("active"), allowed_namespaces={"m", "s"})
    source = _rows_from_json(value.get("source"), allowed_namespaces={"m"})
    cursors = value.get("cursors")
    if not isinstance(cursors, dict):
        raise IngestionError("compaction cursors are required")
    parsed_cursors = {str(key): int(cursor) for key, cursor in cursors.items()}
    if "m" not in parsed_cursors or "s" not in parsed_cursors:
        raise IngestionError("compaction cursors must include m and s")
    if any(cursor < 0 for cursor in parsed_cursors.values()):
        raise IngestionError("compaction cursors must be non-negative")
    _validate_active_rows(active, parsed_cursors)
    return CompactionPayload(active=active, source=source, cursors=parsed_cursors)


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


def _row_to_json(value: ChatMessageWithOrdinal) -> dict[str, Any]:
    return {
        "namespace": value.namespace,
        "ordinal": value.ordinal,
        "message": value.message,
    }


def _validate_active_rows(
    rows: tuple[ChatMessageWithOrdinal, ...], cursors: dict[str, int]
) -> None:
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row.namespace, row.ordinal)
        if key in seen:
            raise IngestionError("compaction active rows contain duplicate ordinal")
        seen.add(key)
        cursor = cursors[row.namespace]
        if row.ordinal >= cursor:
            raise IngestionError("compaction active row ordinal is outside cursor")


def _rows_from_json(
    value: object,
    *,
    allowed_namespaces: set[str],
) -> tuple[ChatMessageWithOrdinal, ...]:
    if not isinstance(value, list):
        raise IngestionError("message rows must be an array")
    rows: list[ChatMessageWithOrdinal] = []
    for row in value:
        if not isinstance(row, dict):
            raise IngestionError("message row must be an object")
        namespace = row.get("namespace")
        ordinal = row.get("ordinal")
        message = row.get("message")
        if namespace not in allowed_namespaces:
            raise IngestionError("message row namespace is invalid")
        if not isinstance(ordinal, int) or ordinal < 0:
            raise IngestionError("message row ordinal is invalid")
        if not isinstance(message, dict):
            raise IngestionError("message row message is required")
        rows.append(
            ChatMessageWithOrdinal(
                namespace=namespace, ordinal=ordinal, message=message
            )
        )
    return tuple(rows)


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
