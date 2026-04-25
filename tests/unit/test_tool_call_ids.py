from uuid import uuid4

import pytest

from plap.responses.tools import (
    InvalidToolCallIDError,
    ModelSide,
    ToolCallID,
    ToolCallIDContext,
    open_tool_call_id,
    open_tool_call_id_with_keyring,
    seal_tool_call_id,
    seal_tool_call_id_with_keyring,
)
from plap.responses.tools.call_ids import (
    MAX_TOOL_CALL_INDEX,
    _pack_uvarint,
    _unpack_plaintext,
)
from plap.sealing import SealingKeyring, derive_key


def test_tool_call_id_roundtrip() -> None:
    key = b"k" * 32
    context = _context()
    value = ToolCallID(
        side=ModelSide.REVIEWER,
        tool_call_index=300,
        upstream_tool_call_id="call_provider_123",
    )

    token = seal_tool_call_id(value, context=context, key=key)
    opened = open_tool_call_id(token, context=context, key=key)

    assert token.startswith("call_")
    assert not token.startswith("call1_")
    assert opened == value


def test_tool_call_id_is_deterministic_with_aes_siv() -> None:
    key = b"k" * 32
    context = _context()
    value = ToolCallID(
        side=ModelSide.MAIN,
        tool_call_index=2,
        upstream_tool_call_id="call_provider_123",
    )

    first = seal_tool_call_id(value, context=context, key=key)
    second = seal_tool_call_id(value, context=context, key=key)

    assert first == second


def test_tool_call_id_supports_long_upstream_ids() -> None:
    key = b"k" * 32
    context = _context()
    value = ToolCallID(
        side=ModelSide.ARBITRATOR,
        tool_call_index=MAX_TOOL_CALL_INDEX,
        upstream_tool_call_id="x" * 512,
    )

    token = seal_tool_call_id(value, context=context, key=key)

    assert open_tool_call_id(token, context=context, key=key) == value


def test_tool_call_id_binds_context() -> None:
    key = b"k" * 32
    context = _context()
    token = seal_tool_call_id(
        ToolCallID(
            side=ModelSide.MAIN,
            tool_call_index=0,
            upstream_tool_call_id="call_provider_123",
        ),
        context=context,
        key=key,
    )
    wrong_context = ToolCallIDContext(
        scope_id=context.scope_id,
        response_id=context.response_id,
        tool_name="write_file",
        signature_hash=context.signature_hash,
    )

    with pytest.raises(InvalidToolCallIDError, match="authentication failed"):
        open_tool_call_id(token, context=wrong_context, key=key)


def test_tool_call_id_keyring_wrappers_open_previous_keys() -> None:
    old_root = b"o" * 32
    active_root = b"a" * 32
    keyring = SealingKeyring(roots=(active_root, old_root))
    old_keyring = SealingKeyring(roots=(old_root,))
    context = _context()
    value = ToolCallID(
        side=ModelSide.MAIN,
        tool_call_index=0,
        upstream_tool_call_id="call_provider_123",
    )

    previous_token = seal_tool_call_id_with_keyring(
        value,
        context=context,
        keyring=old_keyring,
    )
    active_token = seal_tool_call_id_with_keyring(
        value,
        context=context,
        keyring=keyring,
    )

    assert previous_token != active_token
    assert (
        open_tool_call_id_with_keyring(
            previous_token,
            context=context,
            keyring=keyring,
        )
        == value
    )
    assert (
        open_tool_call_id_with_keyring(
            active_token,
            context=context,
            keyring=keyring,
        )
        == value
    )


def test_tool_call_id_rejects_wrong_purpose_key() -> None:
    root = b"r" * 32
    context = _context()
    value = ToolCallID(
        side=ModelSide.MAIN,
        tool_call_index=0,
        upstream_tool_call_id="call_provider_123",
    )

    token = seal_tool_call_id(
        value, context=context, key=derive_key(root, "tool_call_id")
    )

    with pytest.raises(InvalidToolCallIDError, match="authentication failed"):
        open_tool_call_id(
            token,
            context=context,
            key=derive_key(root, "encrypted_content"),
        )


def test_tool_call_id_binds_signature_hash() -> None:
    key = b"k" * 32
    context = _context()
    token = seal_tool_call_id(
        ToolCallID(
            side=ModelSide.MAIN,
            tool_call_index=0,
            upstream_tool_call_id="call_provider_123",
        ),
        context=context,
        key=key,
    )
    wrong_context = ToolCallIDContext(
        scope_id=context.scope_id,
        response_id=context.response_id,
        tool_name=context.tool_name,
        signature_hash=b"s" * 32,
    )

    with pytest.raises(InvalidToolCallIDError, match="authentication failed"):
        open_tool_call_id(token, context=wrong_context, key=key)


def test_tool_call_id_rejects_tampering() -> None:
    key = b"k" * 32
    context = _context()
    token = seal_tool_call_id(
        ToolCallID(
            side=ModelSide.ARBITRATOR,
            tool_call_index=1,
            upstream_tool_call_id="call_provider_123",
        ),
        context=context,
        key=key,
    )
    replacement = "A" if token[-1] != "A" else "B"
    tampered = token[:-1] + replacement

    with pytest.raises(InvalidToolCallIDError):
        open_tool_call_id(tampered, context=context, key=key)


def test_tool_call_id_rejects_invalid_prefix_and_key_length() -> None:
    context = _context()

    with pytest.raises(InvalidToolCallIDError, match="prefix"):
        open_tool_call_id("call2_deadbeef", context=context, key=b"k" * 32)

    with pytest.raises(InvalidToolCallIDError, match="key"):
        seal_tool_call_id(
            ToolCallID(
                side=ModelSide.MAIN,
                tool_call_index=0,
                upstream_tool_call_id="call_provider_123",
            ),
            context=context,
            key=b"short",
        )


def test_tool_call_id_supports_large_uvarint_index() -> None:
    key = b"k" * 32
    context = _context()
    value = ToolCallID(
        side=ModelSide.MAIN,
        tool_call_index=MAX_TOOL_CALL_INDEX,
        upstream_tool_call_id="call_provider_123",
    )

    token = seal_tool_call_id(value, context=context, key=key)

    assert open_tool_call_id(token, context=context, key=key) == value


def test_tool_call_id_rejects_invalid_value_limits() -> None:
    context = _context()

    with pytest.raises(InvalidToolCallIDError, match="index"):
        seal_tool_call_id(
            ToolCallID(
                side=ModelSide.MAIN,
                tool_call_index=MAX_TOOL_CALL_INDEX + 1,
                upstream_tool_call_id="call_provider_123",
            ),
            context=context,
            key=b"k" * 32,
        )


def test_tool_call_id_rejects_invalid_signature_hash_length() -> None:
    context = ToolCallIDContext(
        scope_id=uuid4(),
        response_id="resp_1",
        tool_name="read_file",
        signature_hash=b"short",
    )

    with pytest.raises(InvalidToolCallIDError, match="signature hash"):
        seal_tool_call_id(
            ToolCallID(
                side=ModelSide.MAIN,
                tool_call_index=0,
                upstream_tool_call_id="call_provider_123",
            ),
            context=context,
            key=b"k" * 32,
        )


def test_tool_call_id_rejects_invalid_encrypted_header_bits() -> None:
    with pytest.raises(InvalidToolCallIDError, match="version"):
        _unpack_plaintext(bytes([0]) + _pack_uvarint(0) + b"call_provider_123")

    with pytest.raises(InvalidToolCallIDError, match="reserved"):
        _unpack_plaintext(bytes([0b00010001]) + _pack_uvarint(0) + b"call_provider_123")


def _context() -> ToolCallIDContext:
    return ToolCallIDContext(
        scope_id=uuid4(),
        response_id="resp_1",
        tool_name="read_file",
        signature_hash=b"h" * 32,
    )
