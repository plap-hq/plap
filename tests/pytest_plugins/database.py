from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import docker
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from docker.errors import ImageNotFound
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer

from plap.auth import APIKeyManager, IssuedAPIKey, normalize_email
from plap.persistence import create_database_engine, create_session_maker
from plap.persistence.models import (
    Organization,
    OrganizationMembership,
    SSOProvider,
    User,
    UserEmail,
    UserIdentity,
)
from plap.settings import Settings

POSTGRES_IMAGE = "plap-postgres-pg-cron:16"


def _to_asyncpg_url(url: str) -> str:
    if url.startswith("postgresql+psycopg2://"):
        return url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


async def _reset_database_schema(database_url: str) -> None:
    engine = create_database_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(
                text("DROP SCHEMA IF EXISTS response_state CASCADE")
            )
            await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await connection.execute(text("CREATE SCHEMA public"))
    finally:
        await engine.dispose()


def _run_migrations(database_url: str) -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


@dataclass(slots=True)
class SeededAuthData:
    api_key: str
    api_key_id: UUID
    organization_id: UUID
    sso_provider_id: UUID
    user_id: UUID


@pytest.fixture(scope="session")
def postgres_container() -> PostgresContainer:
    _ensure_postgres_image()
    with PostgresContainer(POSTGRES_IMAGE) as container:
        yield container


def _ensure_postgres_image() -> None:
    client = docker.from_env()
    try:
        client.images.get(POSTGRES_IMAGE)
    except ImageNotFound:
        dockerfile_dir = Path(__file__).resolve().parents[1] / "postgres"
        client.images.build(path=str(dockerfile_dir), tag=POSTGRES_IMAGE, rm=True)


@pytest.fixture(scope="session")
def test_settings(postgres_container: PostgresContainer) -> Settings:
    return Settings(
        api_key_pepper="test-pepper",
        database_url=_to_asyncpg_url(postgres_container.get_connection_url()),
    )


@pytest_asyncio.fixture(autouse=True)
async def database_schema(test_settings: Settings) -> None:
    await _reset_database_schema(test_settings.database_url)
    await asyncio.to_thread(_run_migrations, test_settings.database_url)


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
