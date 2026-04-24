from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer

from plap.auth import APIKeyManager, IssuedAPIKey, normalize_email
from plap.db import create_database_engine, create_session_maker
from plap.models import (
    Base,
    Organization,
    OrganizationMembership,
    SSOProvider,
    User,
    UserEmail,
    UserIdentity,
)
from plap.settings import Settings


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


@dataclass(slots=True)
class SeededAuthData:
    api_key: str
    api_key_id: UUID
    organization_id: UUID
    sso_provider_id: UUID
    user_id: UUID


@pytest.fixture(scope="session")
def postgres_container() -> PostgresContainer:
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def test_settings(postgres_container: PostgresContainer) -> Settings:
    return Settings(
        api_key_pepper="test-pepper",
        database_url=_to_asyncpg_url(postgres_container.get_connection_url()),
    )


@pytest_asyncio.fixture(autouse=True)
async def database_schema(test_settings: Settings) -> None:
    engine = create_database_engine(test_settings.database_url)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session_maker(test_settings: Settings):
    engine = create_database_engine(test_settings.database_url)
    try:
        yield create_session_maker(engine)
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def seeded_auth_data(
    db_session_maker,
    test_settings: Settings,
) -> SeededAuthData:
    manager = APIKeyManager(pepper=test_settings.api_key_pepper)

    async with db_session_maker() as session:
        user = User(display_name="Integration Test User")
        organization = Organization(
            slug=f"acme-{uuid4().hex[:8]}",
            name="Acme Test Org",
        )
        session.add_all([user, organization])
        await session.flush()

        email = UserEmail(
            user_id=user.id,
            email="test.user@example.com",
            normalized_email=normalize_email("test.user@example.com"),
            is_primary=True,
            is_verified=True,
        )
        membership = OrganizationMembership(
            organization_id=organization.id,
            user_id=user.id,
            role="owner",
            status="active",
        )
        sso_provider = SSOProvider(
            organization_id=organization.id,
            slug="okta-main",
            display_name="Okta",
            provider_type="oidc",
            issuer="https://example.okta.test/oauth2/default",
            metadata_url="https://example.okta.test/.well-known/openid-configuration",
            client_id="client-id",
            audience="plap",
            domain_hint="example.com",
            attribute_mapping={"email": "email", "name": "name"},
        )
        session.add_all([email, membership, sso_provider])
        await session.flush()

        identity = UserIdentity(
            user_id=user.id,
            organization_id=organization.id,
            sso_provider_id=sso_provider.id,
            provider_type="oidc",
            provider_name="okta",
            provider_subject=f"sub-{uuid4().hex}",
            email=email.normalized_email,
            claims={"groups": ["engineering"]},
        )
        session.add(identity)

        issued_key: IssuedAPIKey = await manager.issue_key(
            session,
            user_id=user.id,
            organization_id=organization.id,
            name="integration key",
        )
        await session.commit()

        return SeededAuthData(
            api_key=issued_key.plaintext_key,
            api_key_id=issued_key.api_key_id,
            organization_id=organization.id,
            sso_provider_id=sso_provider.id,
            user_id=user.id,
        )
