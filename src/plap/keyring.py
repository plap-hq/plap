from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ROOT_KEY_BYTES = 32
DERIVED_KEY_BYTES = 32


class SealingKeyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SealingKeyring:
    roots: tuple[bytes, ...]

    @classmethod
    def from_encoded(cls, encoded_keys: list[str]) -> SealingKeyring:
        roots = tuple(
            _decode_root_key(value) for value in encoded_keys if value.strip()
        )
        if not roots:
            raise SealingKeyError("at least one sealing key is required")
        return cls(roots=roots)

    def active(self, purpose: str) -> bytes:
        return derive_key(self.roots[0], purpose)

    def candidates(self, purpose: str) -> tuple[bytes, ...]:
        return tuple(derive_key(root, purpose) for root in self.roots)


def derive_key(root_key: bytes, purpose: str) -> bytes:
    if len(root_key) != ROOT_KEY_BYTES:
        raise SealingKeyError("sealing root key must be 32 bytes")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=DERIVED_KEY_BYTES,
        salt=None,
        info=purpose_label(purpose),
    ).derive(root_key)


def associated_data(purpose: str, *fields: bytes) -> list[bytes]:
    return [purpose_label(purpose), *fields]


def purpose_label(purpose: str) -> bytes:
    if not purpose:
        raise SealingKeyError("sealing purpose is required")
    return f"plap:{purpose}".encode()


def _decode_root_key(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(
            value + padding,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise SealingKeyError("sealing key is not valid base64url") from exc
    if len(decoded) != ROOT_KEY_BYTES:
        raise SealingKeyError("sealing root key must be 32 bytes")
    return decoded
