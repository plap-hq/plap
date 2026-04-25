from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import IntEnum
from uuid import UUID

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESSIV

from plap.sealing import SealingKeyring, associated_data

TOKEN_PREFIX = "call_"
PURPOSE = "tool_call_id"
FORMAT_VERSION = 1
KEY_BYTES = 32
TAG_BYTES = 16
MAX_TOOL_CALL_INDEX = 2**32 - 1
SIGNATURE_HASH_BYTES = 32
_VERSION_MASK = 0b11110000
_VERSION_SHIFT = 4
_SIDE_MASK = 0b00001100
_SIDE_SHIFT = 2
_RESERVED_MASK = 0b00000011


class InvalidToolCallIDError(ValueError):
    pass


class ModelSide(IntEnum):
    MAIN = 0
    REVIEWER = 1
    ARBITRATOR = 2


@dataclass(frozen=True, slots=True)
class ToolCallID:
    side: ModelSide
    tool_call_index: int
    upstream_tool_call_id: str


@dataclass(frozen=True, slots=True)
class ToolCallIDContext:
    scope_id: UUID
    response_id: str
    tool_name: str
    signature_hash: bytes


def seal_tool_call_id(
    value: ToolCallID,
    *,
    context: ToolCallIDContext,
    key: bytes,
) -> str:
    _validate_key(key)
    plaintext = _pack_plaintext(value)
    ciphertext = AESSIV(key).encrypt(plaintext, _associated_data(context))
    return TOKEN_PREFIX + _b64url_encode(ciphertext)


def open_tool_call_id(
    token: str,
    *,
    context: ToolCallIDContext,
    key: bytes,
) -> ToolCallID:
    _validate_key(key)
    if not token.startswith(TOKEN_PREFIX):
        raise InvalidToolCallIDError("invalid tool call id prefix")
    payload = _b64url_decode(token[len(TOKEN_PREFIX) :])
    if len(payload) <= TAG_BYTES:
        raise InvalidToolCallIDError("tool call id payload is too short")
    try:
        plaintext = AESSIV(key).decrypt(payload, _associated_data(context))
    except InvalidTag as exc:
        raise InvalidToolCallIDError("tool call id authentication failed") from exc
    return _unpack_plaintext(plaintext)


def seal_tool_call_id_with_keyring(
    value: ToolCallID,
    *,
    context: ToolCallIDContext,
    keyring: SealingKeyring,
) -> str:
    return seal_tool_call_id(
        value,
        context=context,
        key=keyring.active(PURPOSE),
    )


def open_tool_call_id_with_keyring(
    token: str,
    *,
    context: ToolCallIDContext,
    keyring: SealingKeyring,
) -> ToolCallID:
    last_error: InvalidToolCallIDError | None = None
    for key in keyring.candidates(PURPOSE):
        try:
            return open_tool_call_id(token, context=context, key=key)
        except InvalidToolCallIDError as exc:
            last_error = exc
    raise InvalidToolCallIDError("tool call id could not be opened") from last_error


def _pack_plaintext(value: ToolCallID) -> bytes:
    if value.side not in {ModelSide.MAIN, ModelSide.REVIEWER, ModelSide.ARBITRATOR}:
        raise InvalidToolCallIDError("invalid model side")
    upstream_id = value.upstream_tool_call_id.encode()
    if not upstream_id:
        raise InvalidToolCallIDError("upstream tool call id is required")
    header = (FORMAT_VERSION << _VERSION_SHIFT) | (int(value.side) << _SIDE_SHIFT)
    return bytes([header]) + _pack_uvarint(value.tool_call_index) + upstream_id


def _unpack_plaintext(value: bytes) -> ToolCallID:
    if len(value) < 3:
        raise InvalidToolCallIDError("tool call id plaintext is too short")
    header = value[0]
    if header & _RESERVED_MASK:
        raise InvalidToolCallIDError("tool call id reserved bits are set")
    version = (header & _VERSION_MASK) >> _VERSION_SHIFT
    if version != FORMAT_VERSION:
        raise InvalidToolCallIDError("unsupported tool call id version")
    try:
        side = ModelSide((header & _SIDE_MASK) >> _SIDE_SHIFT)
    except ValueError as exc:
        raise InvalidToolCallIDError("invalid model side") from exc
    tool_call_index, offset = _unpack_uvarint(value, 1)
    upstream_id = value[offset:]
    if not upstream_id:
        raise InvalidToolCallIDError("upstream tool call id is required")
    try:
        upstream_tool_call_id = upstream_id.decode()
    except UnicodeDecodeError as exc:
        raise InvalidToolCallIDError("upstream tool call id is not UTF-8") from exc
    return ToolCallID(
        side=side,
        tool_call_index=tool_call_index,
        upstream_tool_call_id=upstream_tool_call_id,
    )


def _associated_data(context: ToolCallIDContext) -> list[bytes]:
    if len(context.signature_hash) != SIGNATURE_HASH_BYTES:
        raise InvalidToolCallIDError("signature hash must be 32 bytes")
    return associated_data(
        PURPOSE,
        context.scope_id.bytes,
        context.response_id.encode(),
        context.tool_name.encode(),
        context.signature_hash,
    )


def _validate_key(key: bytes) -> None:
    if len(key) != KEY_BYTES:
        raise InvalidToolCallIDError("tool call id key must be 32 bytes")


def _pack_uvarint(value: int) -> bytes:
    if value < 0 or value > MAX_TOOL_CALL_INDEX:
        raise InvalidToolCallIDError("tool call index is out of range")
    chunks: list[int] = []
    while value >= 0x80:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    chunks.append(value)
    return bytes(chunks)


def _unpack_uvarint(value: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while offset < len(value):
        byte = value[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            if result > MAX_TOOL_CALL_INDEX:
                raise InvalidToolCallIDError("tool call index is out of range")
            return result, offset
        shift += 7
        if shift > 35:
            break
    raise InvalidToolCallIDError("invalid tool call index encoding")


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _b64url_decode(value: str) -> bytes:
    if not value:
        raise InvalidToolCallIDError("tool call id payload is empty")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except ValueError as exc:
        raise InvalidToolCallIDError("tool call id payload is not base64url") from exc
