import base64

import pytest

from plap.sealing import SealingKeyError, SealingKeyring, associated_data, derive_key
from plap.settings import Settings


def test_sealing_keyring_decodes_base64url_roots_and_preserves_order() -> None:
    first = _encoded(b"a" * 32)
    second = _encoded(b"b" * 32)

    keyring = SealingKeyring.from_encoded([first, second])

    assert keyring.roots == (b"a" * 32, b"b" * 32)
    assert keyring.active("tool_call_id") == derive_key(b"a" * 32, "tool_call_id")
    assert keyring.candidates("tool_call_id") == (
        derive_key(b"a" * 32, "tool_call_id"),
        derive_key(b"b" * 32, "tool_call_id"),
    )


def test_sealing_keyring_rejects_invalid_roots() -> None:
    with pytest.raises(SealingKeyError, match="at least one"):
        SealingKeyring.from_encoded([])

    with pytest.raises(SealingKeyError, match="base64url"):
        SealingKeyring.from_encoded(["!!!!"])

    with pytest.raises(SealingKeyError, match="32 bytes"):
        SealingKeyring.from_encoded([_encoded(b"short")])


def test_sealing_derivation_is_stable_and_purpose_separated() -> None:
    root = b"r" * 32

    assert derive_key(root, "tool_call_id") == derive_key(root, "tool_call_id")
    assert derive_key(root, "tool_call_id") != derive_key(root, "encrypted_content")
    assert associated_data("tool_call_id", b"a", b"b") == [
        b"plap:tool_call_id",
        b"a",
        b"b",
    ]


def test_settings_splits_comma_separated_sealing_keys() -> None:
    first = _encoded(b"a" * 32)
    second = _encoded(b"b" * 32)

    settings = Settings(
        api_key_pepper="pepper",
        database_url="postgresql+asyncpg://example/test",
        sealing_keys=f"{first}, {second}",
    )

    assert settings.sealing_keys == [first, second]


def _encoded(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()
