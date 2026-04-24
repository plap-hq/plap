from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from hmac import compare_digest
from uuid import UUID

import blake3
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from plap.models import APIKey, utcnow

API_KEY_PREFIX = "plap"


class AuthError(Exception):
    pass


@dataclass(slots=True)
class AuthContext:
    api_key_id: UUID
    organization_id: UUID | None
    user_id: UUID


@dataclass(slots=True)
class IssuedAPIKey:
    api_key_id: UUID
    key_id: str
    plaintext_key: str


def normalize_email(email: str) -> str:
    return email.strip().lower()


class APIKeyManager:
    def __init__(self, *, pepper: str) -> None:
        self._pepper = pepper

    def build_secret_hash(self, *, key_id: str, secret: str) -> str:
        hasher = blake3.blake3()
        hasher.update(self._pepper.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(key_id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(secret.encode("utf-8"))
        return hasher.hexdigest()

    def generate_plaintext_key(self) -> tuple[str, str, str]:
        key_id = secrets.token_hex(8)
        secret = secrets.token_hex(24)
        return key_id, secret, f"{API_KEY_PREFIX}_{key_id}_{secret}"

    async def issue_key(
        self,
        session: AsyncSession,
        *,
        user_id: UUID,
        name: str,
        organization_id: UUID | None = None,
        expires_at: datetime | None = None,
    ) -> IssuedAPIKey:
        key_id, secret, plaintext_key = self.generate_plaintext_key()
        record = APIKey(
            user_id=user_id,
            organization_id=organization_id,
            name=name,
            key_id=key_id,
            key_prefix=f"{API_KEY_PREFIX}_{key_id}",
            secret_hash=self.build_secret_hash(key_id=key_id, secret=secret),
            last_four=secret[-4:],
            expires_at=expires_at,
        )
        session.add(record)
        await session.flush()
        return IssuedAPIKey(
            api_key_id=record.id,
            key_id=key_id,
            plaintext_key=plaintext_key,
        )

    async def authenticate_bearer_token(
        self,
        session: AsyncSession,
        authorization_header: str | None,
    ) -> AuthContext:
        if authorization_header is None:
            raise AuthError("Missing bearer token")

        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise AuthError("Invalid bearer token")

        prefix, separator, remainder = token.partition("_")
        if prefix != API_KEY_PREFIX or not separator:
            raise AuthError("Invalid API key")

        key_id, separator, secret = remainder.partition("_")
        if not key_id or not separator or not secret:
            raise AuthError("Invalid API key")

        result = await session.execute(select(APIKey).where(APIKey.key_id == key_id))
        api_key = result.scalar_one_or_none()
        if api_key is None:
            raise AuthError("Invalid API key")

        if api_key.revoked_at is not None:
            raise AuthError("API key has been revoked")
        if api_key.expires_at is not None and api_key.expires_at <= utcnow():
            raise AuthError("API key has expired")

        expected_hash = self.build_secret_hash(key_id=key_id, secret=secret)
        if not compare_digest(api_key.secret_hash, expected_hash):
            raise AuthError("Invalid API key")

        api_key.last_used_at = utcnow()
        await session.commit()

        return AuthContext(
            api_key_id=api_key.id,
            organization_id=api_key.organization_id,
            user_id=api_key.user_id,
        )
