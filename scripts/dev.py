#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import secrets
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anyio
import docker
from alembic import command
from alembic.config import Config
from docker.errors import DockerException, ImageNotFound
from sqlalchemy import select
from testcontainers.core.container import DockerContainer

REPO_ROOT = Path(__file__).resolve().parents[1]

from plap.app import _import_plugin, _plugin_names, _plugins  # noqa: E402
from plap.auth import API_KEY_PREFIX, APIKeyManager, normalize_email  # noqa: E402
from plap.bus import bus  # noqa: E402
from plap.config import CueBox, load  # noqa: E402
from plap.llms.completions.providers import PROVIDER_BUILDERS  # noqa: E402
from plap.persistence.db import create_database_engine, create_session_maker  # noqa: E402
from plap.persistence.models import APIKey, Organization, OrganizationMembership, User, UserEmail  # noqa: E402

STATE_ENV_KEYS = (
    "PLAP_DATABASE_URL",
    "PLAP_API_KEY_PEPPER",
    "PLAP_DEBUG_DEBATE_SUMMARIES",
    "PLAP_DEV_LOG_FILE",
    "PLAP_FOREIGN_LOG_LEVEL",
    "PLAP_LOG_LEVEL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_LOGS_EXPORTER",
    "OTEL_SERVICE_NAME",
    "OTEL_TRACES_EXPORTER",
    "PLAP_SEALING_KEYS",
    "PLAP_DEV_API_KEY",
    "PLAP_DEV_BASE_URL",
    "PLAP_DEV_MODEL",
    "PLAP_DEV_ORG_SLUG",
    "PLAP_DEV_POSTGRES_CONTAINER",
    "PLAP_DEV_POSTGRES_PORT",
    "PLAP_DEV_USER_EMAIL",
    "PLAP_DEV_WEBSOCKET_BASE_URL",
)

DEFAULT_POSTGRES_CONTAINER = "plap-dev-postgres"
DEFAULT_POSTGRES_DB = "plap"
DEFAULT_POSTGRES_USER = "plap"
DEFAULT_POSTGRES_PASSWORD = "plap"
DEFAULT_POSTGRES_IMAGE = "plap-postgres-pg-cron:16"
DEFAULT_DEV_EMAIL = "dev@example.com"
DEFAULT_DEV_API_KEY_ID = "dev"
DEFAULT_DEV_API_KEY_SECRET = "change-this-after-demo"
DEFAULT_DEV_API_KEY = f"{API_KEY_PREFIX}_{DEFAULT_DEV_API_KEY_ID}_{DEFAULT_DEV_API_KEY_SECRET}"
DEFAULT_DEV_ORG_SLUG = "plap-dev"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_OTEL_COLLECTOR_CONTAINER = "plap-dev-otelcol"
DEFAULT_OTEL_COLLECTOR_HEALTH_PORT = 13133
DEFAULT_OTEL_COLLECTOR_IMAGE = "otel/opentelemetry-collector-contrib:0.153.0"
DEFAULT_OTEL_COLLECTOR_PORT = 4318
DEFAULT_PORT = 8000
DEFAULT_POSTGRES_PORT = 55432
DEFAULT_MODEL = "plap-ai/mote"
DEFAULT_LOG_FILE = (REPO_ROOT / ".dev" / "plap.log.jsonl").resolve()
SERVER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def _provider_env_suffix(slug: str) -> str:
    return slug.upper().replace("-", "_")


def _provider_env_var(slug: str) -> str:
    return f"PLAP_LLM_{_provider_env_suffix(slug)}_API_KEY"


def _provider_env_alias(slug: str) -> str:
    return f"{_provider_env_suffix(slug)}_API_KEY"


ROUTE_ENV_VARS = {f"{slug}/": _provider_env_var(slug) for slug in PROVIDER_BUILDERS}
PROVIDER_ENV_ALIASES = {_provider_env_var(slug): _provider_env_alias(slug) for slug in PROVIDER_BUILDERS}


@dataclass(slots=True)
class EphemeralResources:
    state_file: Path
    collector_config_file: Path | None = None
    collector_container: DockerContainer | None = None
    postgres_container: DockerContainer | None = None
    server_process: subprocess.Popen[str] | None = None


def main() -> int:
    args = _parse_args()
    state_file = args.state_file.resolve()
    resources = EphemeralResources(state_file=state_file)
    try:
        explicit_env = set(os.environ)
        env_file = args.env_file.resolve()
        _remove_state_file(state_file)

        env_values = _parse_env_file(env_file)
        for key, value in env_values.items():
            if key not in explicit_env:
                os.environ[key] = value

        _populate_provider_aliases()
        config = _load_dev_config()
        dev_model = _resolve_dev_model(config)
        _require_provider_keys(config, model=dev_model)
        managed_state = {
            "PLAP_DEV_USER_EMAIL": os.environ.get("PLAP_DEV_USER_EMAIL", DEFAULT_DEV_EMAIL),
            "PLAP_DEV_ORG_SLUG": os.environ.get("PLAP_DEV_ORG_SLUG", DEFAULT_DEV_ORG_SLUG),
        }

        _ensure_managed_postgres(args, managed_state, resources)
        os.environ.setdefault("PLAP_DEBUG_DEBATE_SUMMARIES", "true")
        os.environ.setdefault("PLAP_DEV_LOG_FILE", str(DEFAULT_LOG_FILE))
        os.environ.setdefault("PLAP_FOREIGN_LOG_LEVEL", "WARNING")
        os.environ.setdefault("PLAP_LOG_LEVEL", "DEBUG")
        os.environ.setdefault("PLAP_API_KEY_PEPPER", secrets.token_hex(24))
        os.environ.setdefault("PLAP_SEALING_KEYS", json.dumps([_generate_sealing_key()]))
        _normalize_sealing_keys_env()
        _reset_log_file(Path(os.environ["PLAP_DEV_LOG_FILE"]))
        _ensure_managed_collector(managed_state, resources)

        managed_state.update(
            {
                "PLAP_DATABASE_URL": os.environ["PLAP_DATABASE_URL"],
                "PLAP_API_KEY_PEPPER": os.environ["PLAP_API_KEY_PEPPER"],
                "PLAP_DEBUG_DEBATE_SUMMARIES": os.environ["PLAP_DEBUG_DEBATE_SUMMARIES"],
                "PLAP_DEV_LOG_FILE": os.environ["PLAP_DEV_LOG_FILE"],
                "PLAP_FOREIGN_LOG_LEVEL": os.environ["PLAP_FOREIGN_LOG_LEVEL"],
                "PLAP_LOG_LEVEL": os.environ["PLAP_LOG_LEVEL"],
                "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
                "OTEL_EXPORTER_OTLP_PROTOCOL": os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"],
                "OTEL_LOGS_EXPORTER": os.environ["OTEL_LOGS_EXPORTER"],
                "OTEL_SERVICE_NAME": os.environ["OTEL_SERVICE_NAME"],
                "OTEL_TRACES_EXPORTER": os.environ["OTEL_TRACES_EXPORTER"],
                "PLAP_SEALING_KEYS": os.environ["PLAP_SEALING_KEYS"],
            }
        )

        if not args.skip_db_upgrade:
            _run_migrations(os.environ["PLAP_DATABASE_URL"])

        api_key = asyncio.run(
            _ensure_dev_api_key(
                database_url=os.environ["PLAP_DATABASE_URL"],
                api_key_pepper=os.environ["PLAP_API_KEY_PEPPER"],
                managed_state=managed_state,
            )
        )
        managed_state["PLAP_DEV_API_KEY"] = api_key
        client_host = _client_host(args.host)
        managed_state["PLAP_DEV_BASE_URL"] = f"http://{client_host}:{args.port}/v1"
        managed_state["PLAP_DEV_WEBSOCKET_BASE_URL"] = f"ws://{client_host}:{args.port}/v1"
        managed_state["PLAP_DEV_MODEL"] = dev_model
        _write_state_file(state_file, managed_state)
        _print_summary(
            state_file=state_file,
            log_file=Path(os.environ["PLAP_DEV_LOG_FILE"]),
            host=client_host,
            port=args.port,
            api_key=api_key,
            model=dev_model,
        )

        if args.no_server:
            return 0

        return _run_server(args, resources)
    finally:
        _cleanup(resources)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap a local plap dev server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--postgres-port", default=DEFAULT_POSTGRES_PORT, type=int)
    parser.add_argument("--postgres-container", default=DEFAULT_POSTGRES_CONTAINER)
    parser.add_argument("--postgres-image", default=DEFAULT_POSTGRES_IMAGE)
    parser.add_argument("--env-file", default=REPO_ROOT / ".env", type=Path)
    parser.add_argument("--state-file", default=REPO_ROOT / ".dev" / ".env", type=Path)
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--skip-db-upgrade", action="store_true")
    parser.add_argument("uvicorn_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        values[key.strip()] = _unquote(value.strip())
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _generate_sealing_key() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def _normalize_sealing_keys_env() -> None:
    raw = os.environ.get("PLAP_SEALING_KEYS")
    if not raw:
        return
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(part, str) and part for part in parsed):
        raise SystemExit("PLAP_SEALING_KEYS must be a JSON array of non-empty strings or a comma-separated string.")
    os.environ["PLAP_SEALING_KEYS"] = ",".join(parsed)


def _populate_provider_aliases() -> None:
    for target, source in PROVIDER_ENV_ALIASES.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]
        if not os.environ.get(source) and os.environ.get(target):
            os.environ[source] = os.environ[target]


def _route_model_attempts(model: str) -> list[str]:
    attempts = [part.strip() for part in model.split(",")]
    if not attempts or any(not attempt for attempt in attempts):
        raise SystemExit(f"Invalid routed model fallback chain in dev settings: {model!r}")
    return attempts


def _completion_fields(resolved: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    fields: dict[str, Mapping[str, object]] = {}
    for name, value in resolved.items():
        if not isinstance(value, Mapping):
            continue
        if not isinstance(value.get("model"), str):
            continue
        if not isinstance(value.get("sampling"), Mapping):
            continue
        if not isinstance(value.get("output_equivalence"), Mapping):
            continue
        fields[name] = value
    return fields


def _provider_env_vars(model: str) -> tuple[str, ...]:
    env_vars: list[str] = []
    for attempt in _route_model_attempts(model):
        provider, separator, _ = attempt.partition("/")
        env_var = ROUTE_ENV_VARS.get(f"{provider}/") if separator else None
        if env_var is None:
            continue
        alias = PROVIDER_ENV_ALIASES[env_var]
        if alias not in env_vars:
            env_vars.append(alias)
    return tuple(env_vars)


def _missing_provider_fields(fields: Mapping[str, Mapping[str, object]]) -> dict[str, tuple[str, ...]]:
    missing: dict[str, tuple[str, ...]] = {}
    for name, field in fields.items():
        env_vars = _provider_env_vars(str(field["model"]))
        if not env_vars or not any(os.environ.get(env_var) for env_var in env_vars):
            missing[name] = env_vars
    return missing


def _require_provider_keys(config: CueBox, *, model: str) -> None:
    fields = _completion_fields(config.resolve({"model": model}))
    if not fields:
        raise SystemExit(f"Development model {model!r} has no completion fields.")

    missing = _missing_provider_fields(fields)
    if not missing:
        return

    details = "\n".join(
        f"  {name}: {', '.join(env_vars) if env_vars else 'no recognized provider prefixes'}" for name, env_vars in sorted(missing.items())
    )
    raise SystemExit(
        f"Missing provider credentials for development model {model!r}:\n{details}\n"
        "Put at least one provider key for each field in your shell or repo .env."
    )


def _load_dev_config() -> CueBox:
    discovered = _plugins()
    plugin_names = _plugin_names()

    bus.reset()

    @bus.emit("bootstrap.config")
    async def collect_config(paths: tuple[str, ...]) -> tuple[str, ...]:
        return paths

    for name in plugin_names:
        if name not in discovered:
            raise SystemExit(f"plugin manifest requested unknown plugin: {name!r}")
        _import_plugin(discovered[name])

    config_paths = anyio.run(partial(collect_config, paths=()))
    loaded = load(*config_paths)
    if "plap" not in loaded:
        raise SystemExit("config load did not produce package 'plap'")
    return loaded.plap.config


def _docker_client() -> docker.DockerClient:
    try:
        client = docker.from_env()
        client.ping()
    except DockerException as exc:  # pragma: no cover - local env dependent
        raise SystemExit("docker is required for the ephemeral dev bootstrap, but it is not available.") from exc
    else:
        return client


def _ensure_managed_postgres(args: argparse.Namespace, managed_state: dict[str, str], resources: EphemeralResources) -> None:
    _ensure_postgres_image(args.postgres_image)

    container_name = f"{args.postgres_container}-{os.getpid()}-{secrets.token_hex(4)}"
    host_port = _pick_port(args.postgres_port)
    container = (
        DockerContainer(args.postgres_image)
        .with_name(container_name)
        .with_env("POSTGRES_DB", DEFAULT_POSTGRES_DB)
        .with_env("POSTGRES_USER", DEFAULT_POSTGRES_USER)
        .with_env("POSTGRES_PASSWORD", DEFAULT_POSTGRES_PASSWORD)
        .with_bind_ports("5432/tcp", host_port)
        .with_kwargs(labels={"plap.dev.ephemeral": "true"})
        .with_command(f"postgres -c shared_preload_libraries=pg_cron -c cron.database_name={DEFAULT_POSTGRES_DB}")
    )
    try:
        container.start()
    except Exception as exc:  # pragma: no cover - local env dependent
        raise SystemExit(f"failed to start managed postgres container {container_name!r}: {exc}") from exc
    resources.postgres_container = container

    database_url = f"postgresql+asyncpg://{DEFAULT_POSTGRES_USER}:{DEFAULT_POSTGRES_PASSWORD}@127.0.0.1:{host_port}/{DEFAULT_POSTGRES_DB}"
    os.environ["PLAP_DATABASE_URL"] = database_url
    managed_state["PLAP_DATABASE_URL"] = database_url
    managed_state["PLAP_DEV_POSTGRES_CONTAINER"] = container_name
    managed_state["PLAP_DEV_POSTGRES_PORT"] = str(host_port)
    asyncio.run(_wait_for_database(database_url))


def _collector_config(*, log_file: Path) -> str:
    return f"""
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  debug/traces:
    verbosity: basic
  file/logs:
    path: /data/{log_file.name}
    format: json
    flush_interval: 1s

extensions:
  health_check:
    endpoint: 0.0.0.0:13133

service:
  extensions: [health_check]
  pipelines:
    logs:
      receivers: [otlp]
      processors: [batch]
      exporters: [file/logs]
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [debug/traces]
""".strip()


def _ensure_managed_collector(managed_state: dict[str, str], resources: EphemeralResources) -> None:
    log_file = Path(os.environ["PLAP_DEV_LOG_FILE"]).resolve()
    _prepare_collector_output(log_file)
    config_file = log_file.parent / "otelcol-dev.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(_collector_config(log_file=log_file) + "\n", encoding="utf-8")
    resources.collector_config_file = config_file

    container_name = f"{DEFAULT_OTEL_COLLECTOR_CONTAINER}-{os.getpid()}-{secrets.token_hex(4)}"
    host_port = _pick_port(DEFAULT_OTEL_COLLECTOR_PORT)
    health_port = _pick_port(DEFAULT_OTEL_COLLECTOR_HEALTH_PORT)
    container = (
        DockerContainer(DEFAULT_OTEL_COLLECTOR_IMAGE)
        .with_name(container_name)
        .with_bind_ports("4318/tcp", host_port)
        .with_bind_ports("13133/tcp", health_port)
        .with_kwargs(labels={"plap.dev.ephemeral": "true"})
        .with_volume_mapping(config_file, "/etc/otelcol/config.yaml", mode="ro")
        .with_volume_mapping(log_file.parent, "/data", mode="rw")
        .with_command("--config=/etc/otelcol/config.yaml")
    )
    try:
        container.start()
    except Exception as exc:  # pragma: no cover - local env dependent
        raise SystemExit(f"failed to start collector container {container_name!r}: {exc}") from exc
    resources.collector_container = container
    _wait_for_collector(container, health_port=health_port)

    otlp_endpoint = f"http://127.0.0.1:{host_port}"
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "http/protobuf"
    os.environ["OTEL_LOGS_EXPORTER"] = "otlp"
    os.environ["OTEL_SERVICE_NAME"] = "plap"
    os.environ["OTEL_TRACES_EXPORTER"] = "otlp"
    managed_state["OTEL_EXPORTER_OTLP_ENDPOINT"] = otlp_endpoint
    managed_state["OTEL_EXPORTER_OTLP_PROTOCOL"] = os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"]
    managed_state["OTEL_LOGS_EXPORTER"] = os.environ["OTEL_LOGS_EXPORTER"]
    managed_state["OTEL_SERVICE_NAME"] = os.environ["OTEL_SERVICE_NAME"]
    managed_state["OTEL_TRACES_EXPORTER"] = os.environ["OTEL_TRACES_EXPORTER"]


def _ensure_postgres_image(image: str) -> None:
    client = _docker_client()
    try:
        client.images.get(image)
    except ImageNotFound:
        pass
    else:
        return
    dockerfile_dir = REPO_ROOT / "tests" / "postgres"
    client.images.build(path=str(dockerfile_dir), tag=image, rm=True)


def _container_logs(container: DockerContainer) -> str:
    stdout, stderr = container.get_logs()
    output = (stdout + stderr).decode("utf-8", errors="replace").strip()
    return output or "<no collector logs available>"


def _pick_port(preferred_port: int) -> int:
    if _port_is_available(preferred_port):
        return preferred_port
    return _pick_ephemeral_port()


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pick_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_tcp_listener(host: str, port: int, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.25)
    raise SystemExit(f"endpoint {host}:{port} did not become ready in {timeout_seconds:.0f}s")


def _wait_for_collector(container: DockerContainer, *, health_port: int, timeout_seconds: float = 30.0) -> None:
    _wait_for_tcp_listener("127.0.0.1", health_port, timeout_seconds=timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    health_url = f"http://127.0.0.1:{health_port}/"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as response:
                if 200 <= response.status < 300:
                    return
        except urllib.error.URLError, OSError:
            time.sleep(0.25)
            continue
    logs = _container_logs(container)
    raise SystemExit(f"collector did not become healthy in {timeout_seconds:.0f}s\nCollector logs:\n{logs}")


def _prepare_collector_output(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)
    log_file.chmod(0o666)
    log_file.parent.chmod(0o777)


async def _wait_for_database(database_url: str, *, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        engine = create_database_engine(database_url)
        try:
            async with engine.connect():
                return
        except Exception as exc:  # pragma: no cover - transient env-dependent bootstrap path
            last_error = exc
            await asyncio.sleep(1)
        finally:
            await engine.dispose()
    raise SystemExit(f"database did not become ready in {timeout_seconds:.0f}s: {last_error}")


async def _ensure_dev_api_key(
    *,
    database_url: str,
    api_key_pepper: str,
    managed_state: dict[str, str],
) -> str:
    engine = create_database_engine(database_url)
    session_maker = create_session_maker(engine)
    manager = APIKeyManager(pepper=api_key_pepper)
    email_address = managed_state.get("PLAP_DEV_USER_EMAIL", DEFAULT_DEV_EMAIL)
    organization_slug = managed_state.get("PLAP_DEV_ORG_SLUG", DEFAULT_DEV_ORG_SLUG)
    try:
        async with session_maker() as session:
            user = await _ensure_user(session, email_address)
            organization = await _ensure_organization(session, organization_slug)
            await _ensure_membership(session, user_id=user.id, organization_id=organization.id)
            record = await session.scalar(select(APIKey).where(APIKey.key_id == DEFAULT_DEV_API_KEY_ID))
            if record is None:
                record = APIKey(
                    user_id=user.id,
                    organization_id=organization.id,
                    name="local dev key",
                    key_id=DEFAULT_DEV_API_KEY_ID,
                    key_prefix=f"{API_KEY_PREFIX}_{DEFAULT_DEV_API_KEY_ID}",
                    secret_hash="",
                    last_four=DEFAULT_DEV_API_KEY_SECRET[-4:],
                )
                session.add(record)
            record.user_id = user.id
            record.organization_id = organization.id
            record.name = "local dev key"
            record.key_prefix = f"{API_KEY_PREFIX}_{DEFAULT_DEV_API_KEY_ID}"
            record.secret_hash = manager.build_secret_hash(
                key_id=DEFAULT_DEV_API_KEY_ID,
                secret=DEFAULT_DEV_API_KEY_SECRET,
            )
            record.last_four = DEFAULT_DEV_API_KEY_SECRET[-4:]
            record.expires_at = None
            record.revoked_at = None
            await session.commit()
            return DEFAULT_DEV_API_KEY
    finally:
        await engine.dispose()


async def _ensure_user(session, email_address: str) -> User:
    normalized_email = normalize_email(email_address)
    existing_email = await session.scalar(select(UserEmail).where(UserEmail.normalized_email == normalized_email))
    if existing_email is not None:
        user = await session.get(User, existing_email.user_id)
        if user is None:
            raise SystemExit(f"user email {email_address!r} exists without a backing user")
        return user

    user = User(display_name="Local Dev User")
    session.add(user)
    await session.flush()
    session.add(
        UserEmail(
            user_id=user.id,
            email=email_address,
            normalized_email=normalized_email,
            is_primary=True,
            is_verified=True,
        )
    )
    await session.flush()
    return user


async def _ensure_organization(session, organization_slug: str) -> Organization:
    organization = await session.scalar(select(Organization).where(Organization.slug == organization_slug))
    if organization is not None:
        return organization
    organization = Organization(slug=organization_slug, name="Local Dev Org")
    session.add(organization)
    await session.flush()
    return organization


async def _ensure_membership(session, *, user_id, organization_id) -> None:
    membership = await session.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.organization_id == organization_id,
        )
    )
    if membership is not None:
        return
    session.add(
        OrganizationMembership(
            user_id=user_id,
            organization_id=organization_id,
            role="owner",
            status="active",
        )
    )
    await session.flush()


def _resolve_dev_model(config: CueBox) -> str:
    model_names = sorted(str(name) for name in config.overlays.get("model", {}) if isinstance(name, str))
    if DEFAULT_MODEL in model_names:
        return DEFAULT_MODEL
    if model_names:
        return model_names[0]
    raise SystemExit("no configured response models were found in config overlays")


def _run_migrations(database_url: str) -> None:
    config = Config(toml_file="pyproject.toml")
    config.set_main_option("sqlalchemy.url", database_url)
    command.upgrade(config, "head")


def _write_state_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Generated by scripts/dev.py",
        "# Ephemeral: this file is deleted when scripts/dev.py exits.",
        "# Safe to source from another shell while the dev server is running.",
    ]
    lines.extend(f"export {key}={shlex.quote(values[key])}" for key in sorted(values))
    path.write_text("\n".join(lines) + "\n")


def _reset_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def _client_host(host: str) -> str:
    return "127.0.0.1" if host in {"0.0.0.0", "::"} else host


def _print_summary(*, state_file: Path, log_file: Path, host: str, port: int, api_key: str, model: str) -> None:
    print(f"State file (removed on exit): {state_file}")
    print(f"Log file: {log_file}")
    print(f"Base URL: http://{host}:{port}/v1")
    print(f"WebSocket URL: ws://{host}:{port}/v1")
    print(f"Model: {model}")
    print(f"API key: {api_key}")
    print("Source this from another shell while the server is running: source .dev/.env")
    print(f"Explore logs: lnav {shlex.quote(str(log_file))}")
    print(f"Smoke test: curl http://{host}:{port}/v1/models -H 'Authorization: Bearer {api_key}'")


def _run_server(args: argparse.Namespace, resources: EphemeralResources) -> int:
    uvicorn_command = [
        "uvicorn",
        "plap.app:create_app",
        "--factory",
        "--host",
        args.host,
        "--log-level",
        "warning",
        "--port",
        str(args.port),
    ]
    if not args.no_reload:
        uvicorn_command.append("--reload")
    uvicorn_command.extend(args.uvicorn_args)

    process = subprocess.Popen(uvicorn_command, cwd=REPO_ROOT, env=os.environ.copy())
    resources.server_process = process
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    received_signal_count = 0

    def _handle_signal(signum, _frame) -> None:
        nonlocal received_signal_count
        received_signal_count += 1
        if process.poll() is not None:
            return
        if received_signal_count == 1:
            process.terminate()
            return
        process.kill()

    for sig in previous_handlers:
        signal.signal(sig, _handle_signal)

    try:
        return process.wait()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def _cleanup(resources: EphemeralResources) -> None:
    _stop_server(resources.server_process)
    _remove_container(resources.collector_container)
    _remove_container(resources.postgres_container)
    _remove_state_file(resources.collector_config_file)
    _remove_state_file(resources.state_file)


def _stop_server(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _remove_container(container: DockerContainer | None) -> None:
    if container is None:
        return
    with contextlib.suppress(Exception):
        container.stop()


def _remove_state_file(path: Path | None) -> None:
    if path is None:
        return
    if path.exists():
        path.unlink()
    if path.parent.exists() and not any(path.parent.iterdir()):
        path.parent.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
