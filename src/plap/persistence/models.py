from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    metadata = MetaData(
        schema="identity",
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        },
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class UserEmail(Base):
    __tablename__ = "user_emails"
    __table_args__ = (UniqueConstraint("normalized_email"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("identity.users.id", ondelete="CASCADE"))
    email: Mapped[str] = mapped_column(String(320))
    normalized_email: Mapped[str] = mapped_column(String(320))
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Organization(Base):
    __tablename__ = "organizations"
    __table_args__ = (UniqueConstraint("slug"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    slug: Mapped[str] = mapped_column(String(120))
    name: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class OrganizationMembership(Base):
    __tablename__ = "organization_memberships"
    __table_args__ = (UniqueConstraint("organization_id", "user_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("identity.organizations.id", ondelete="CASCADE"))
    user_id: Mapped[UUID] = mapped_column(ForeignKey("identity.users.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(64), default="member")
    status: Mapped[str] = mapped_column(String(64), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class SSOProvider(Base):
    __tablename__ = "sso_providers"
    __table_args__ = (UniqueConstraint("organization_id", "slug"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(ForeignKey("identity.organizations.id", ondelete="CASCADE"))
    slug: Mapped[str] = mapped_column(String(120))
    display_name: Mapped[str] = mapped_column(String(255))
    provider_type: Mapped[str] = mapped_column(String(64))
    issuer: Mapped[str | None] = mapped_column(String(500), nullable=True)
    metadata_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    sso_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    client_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    audience: Mapped[str | None] = mapped_column(String(255), nullable=True)
    domain_hint: Mapped[str | None] = mapped_column(String(255), nullable=True)
    attribute_mapping: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class UserIdentity(Base):
    __tablename__ = "user_identities"
    __table_args__ = (UniqueConstraint("provider_type", "provider_name", "provider_subject"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("identity.users.id", ondelete="CASCADE"))
    organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("identity.organizations.id", ondelete="SET NULL"), nullable=True)
    sso_provider_id: Mapped[UUID | None] = mapped_column(ForeignKey("identity.sso_providers.id", ondelete="SET NULL"), nullable=True)
    provider_type: Mapped[str] = mapped_column(String(64))
    provider_name: Mapped[str] = mapped_column(String(120))
    provider_subject: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    claims: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_authenticated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class APIKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (UniqueConstraint("key_id"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("identity.users.id", ondelete="CASCADE"))
    organization_id: Mapped[UUID | None] = mapped_column(ForeignKey("identity.organizations.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(255))
    key_id: Mapped[str] = mapped_column(String(64))
    key_prefix: Mapped[str] = mapped_column(String(128))
    secret_hash: Mapped[str] = mapped_column(Text)
    last_four: Mapped[str] = mapped_column(String(4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
