#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plap.auth import APIKeyManager, normalize_email  # noqa: E402
from plap.persistence.db import create_database_engine, create_session_maker  # noqa: E402
from plap.persistence.models import Organization, OrganizationMembership, User, UserEmail  # noqa: E402
from plap.settings import Settings, _default_runtime_model_profiles  # noqa: E402

STATE_ENV_KEYS = (
    "PLAP_DATABASE_URL",
    "PLAP_API_KEY_PEPPER",
    "PLAP_DEBUG",
    "PLAP_DEBUG_PAYLOADS",
    "PLAP_DEBUG_DEBATE_SUMMARIES",
    "PLAP_LOG_FILE",
    "PLAP_LOG_JSON",
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
DEFAULT_DEV_ORG_SLUG = "plap-dev"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_POSTGRES_PORT = 55432
DEFAULT_MODEL = "plap-ai/wisp-mini"
DEFAULT_LOG_FILE = (REPO_ROOT / ".dev" / "plap.log.jsonl").resolve()
SERVER_SHUTDOWN_TIMEOUT_SECONDS = 10.0
ROUTE_ENV_VARS = {
    "lightning/": "PLAP_LLM_LIGHTNING_API_KEY",
    "gmicloud/": "PLAP_LLM_GMICLOUD_API_KEY",
    "novita/": "PLAP_LLM_NOVITA_API_KEY",
    "fireworks/": "PLAP_LLM_FIREWORKS_API_KEY",
    "crof/": "PLAP_LLM_CROF_API_KEY",
    "openrouter/": "PLAP_LLM_OPENROUTER_API_KEY",
}
PROVIDER_ENV_ALIASES = {
    "PLAP_LLM_OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
    "PLAP_LLM_LIGHTNING_API_KEY": "LIGHTNING_API_KEY",
    "PLAP_LLM_GMICLOUD_API_KEY": "GMICLOUD_API_KEY",
    "PLAP_LLM_CROF_API_KEY": "CROF_API_KEY",
    "PLAP_LLM_NOVITA_API_KEY": "NOVITA_API_KEY",
    "PLAP_LLM_FIREWORKS_API_KEY": "FIREWORKS_API_KEY",
}


@dataclass(slots=True)
class EphemeralResources:
    state_file: Path
    postgres_container: str | None = None
    server_process: subprocess.Popen[bytes] | None = None


def main() -> int:
    _ensure_subprocess_pythonpath()
    args = _parse_args()
    state_file = args.state_file.resolve()
    resources = EphemeralResources(state_file=state_file)
    try:
        explicit_env = set(os.environ)
        env_file = args.env_file.resolve()
        fallback_env_file = REPO_ROOT / "tests" / ".env"
        _remove_state_file(state_file)

        env_values = _parse_env_file(env_file)
        for key, value in env_values.items():
            if key not in explicit_env:
                os.environ[key] = value

        if fallback_env_file != env_file:
            fallback_env_values = _parse_env_file(fallback_env_file)
            for key, value in fallback_env_values.items():
                os.environ.setdefault(key, value)

        _populate_provider_aliases()
        managed_state = {
            "PLAP_DEV_USER_EMAIL": os.environ.get("PLAP_DEV_USER_EMAIL", DEFAULT_DEV_EMAIL),
            "PLAP_DEV_ORG_SLUG": os.environ.get("PLAP_DEV_ORG_SLUG", DEFAULT_DEV_ORG_SLUG),
        }

        _ensure_managed_postgres(args, managed_state, resources)
        os.environ.setdefault("PLAP_DEBUG", "true")
        os.environ.setdefault("PLAP_DEBUG_PAYLOADS", "true")
        os.environ.setdefault("PLAP_DEBUG_DEBATE_SUMMARIES", "true")
        os.environ.setdefault("PLAP_LOG_JSON", "true")
        os.environ.setdefault("PLAP_LOG_FILE", str(DEFAULT_LOG_FILE))
        os.environ.setdefault("PLAP_API_KEY_PEPPER", secrets.token_hex(24))
        os.environ.setdefault("PLAP_SEALING_KEYS", json.dumps([_generate_sealing_key()]))
        _normalize_sealing_keys_env()
        _reset_log_file(Path(os.environ["PLAP_LOG_FILE"]))
        _require_provider_keys()

        managed_state.update(
            {
                "PLAP_DATABASE_URL": os.environ["PLAP_DATABASE_URL"],
                "PLAP_API_KEY_PEPPER": os.environ["PLAP_API_KEY_PEPPER"],
                "PLAP_DEBUG": os.environ["PLAP_DEBUG"],
                "PLAP_DEBUG_PAYLOADS": os.environ["PLAP_DEBUG_PAYLOADS"],
                "PLAP_DEBUG_DEBATE_SUMMARIES": os.environ["PLAP_DEBUG_DEBATE_SUMMARIES"],
                "PLAP_LOG_FILE": os.environ["PLAP_LOG_FILE"],
                "PLAP_LOG_JSON": os.environ["PLAP_LOG_JSON"],
                "PLAP_SEALING_KEYS": os.environ["PLAP_SEALING_KEYS"],
            }
        )

        if not args.skip_db_upgrade:
            _run_checked(["alembic", "upgrade", "head"], cwd=REPO_ROOT)

        settings = Settings()
        dev_model = _resolve_dev_model(settings)
        api_key = asyncio.run(_ensure_dev_api_key(settings, managed_state))
        managed_state["PLAP_DEV_API_KEY"] = api_key
        client_host = _client_host(args.host)
        managed_state["PLAP_DEV_BASE_URL"] = f"http://{client_host}:{args.port}/v1"
        managed_state["PLAP_DEV_WEBSOCKET_BASE_URL"] = f"ws://{client_host}:{args.port}/v1"
        managed_state["PLAP_DEV_MODEL"] = dev_model
        _write_state_file(state_file, managed_state)
        _print_summary(
            state_file=state_file,
            log_file=Path(os.environ["PLAP_LOG_FILE"]),
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
    parser.add_argument("--state-file", default=REPO_ROOT / ".dev" / "dev.env", type=Path)
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
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"\"", "'"}:
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
        raise SystemExit(
            "PLAP_SEALING_KEYS must be a JSON array of non-empty strings or a comma-separated string."
        )
    os.environ["PLAP_SEALING_KEYS"] = json.dumps(parsed)


def _populate_provider_aliases() -> None:
    for target, source in PROVIDER_ENV_ALIASES.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]


def _require_provider_keys() -> None:
    missing = [
        name
        for name in _required_provider_env_vars()
        if not os.environ.get(name)
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(
            "Missing required provider env vars for the current default runtime profiles and tool classifiers: "
            f"{missing_list}. Put them in your shell or repo .env before running this script."
        )


def _required_provider_env_vars() -> list[str]:
    models = {
        os.environ.get("PLAP_TOOL_EFFECT_CLASSIFIER_MODEL") or Settings.model_fields["tool_effect_classifier_model"].default,
        os.environ.get("PLAP_TOOL_CALL_EFFECT_CLASSIFIER_MODEL") or Settings.model_fields["tool_call_effect_classifier_model"].default,
    }
    for profile in _default_runtime_model_profiles().values():
        models.update(profile.all_models())

    required: set[str] = set()
    for model in models:
        for attempt in _route_model_attempts(model):
            for prefix, env_var in ROUTE_ENV_VARS.items():
                if attempt.startswith(prefix):
                    required.add(env_var)
                    break
    return sorted(required)


def _route_model_attempts(model: str) -> list[str]:
    attempts = [part.strip() for part in model.split(",")]
    if not attempts or any(not attempt for attempt in attempts):
        raise SystemExit(f"Invalid routed model fallback chain in dev settings: {model!r}")
    return attempts


def _ensure_managed_postgres(args: argparse.Namespace, managed_state: dict[str, str], resources: EphemeralResources) -> None:
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit("docker is required for the ephemeral dev bootstrap, but it is not installed.")

    _ensure_postgres_image(docker, args.postgres_image)

    container_name = f"{args.postgres_container}-{os.getpid()}-{secrets.token_hex(4)}"
    host_port = _pick_port(args.postgres_port)
    _run_checked(
        [
            docker,
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            "plap.dev.ephemeral=true",
            "-e",
            f"POSTGRES_DB={DEFAULT_POSTGRES_DB}",
            "-e",
            f"POSTGRES_USER={DEFAULT_POSTGRES_USER}",
            "-e",
            f"POSTGRES_PASSWORD={DEFAULT_POSTGRES_PASSWORD}",
            "-p",
            f"127.0.0.1:{host_port}:5432",
            args.postgres_image,
            "postgres",
            "-c",
            "shared_preload_libraries=pg_cron",
            "-c",
            f"cron.database_name={DEFAULT_POSTGRES_DB}",
        ],
        cwd=REPO_ROOT,
    )
    resources.postgres_container = container_name
    inspected = _docker_inspect(docker, container_name)
    if inspected is None:
        raise SystemExit(f"failed to inspect managed postgres container {container_name!r} after creation")

    host_port = _docker_host_port(inspected)
    if host_port is None:
        raise SystemExit(f"managed postgres container {container_name!r} is missing a published 5432 port")

    database_url = (
        f"postgresql+asyncpg://{DEFAULT_POSTGRES_USER}:{DEFAULT_POSTGRES_PASSWORD}@127.0.0.1:{host_port}/{DEFAULT_POSTGRES_DB}"
    )
    os.environ["PLAP_DATABASE_URL"] = database_url
    managed_state["PLAP_DATABASE_URL"] = database_url
    managed_state["PLAP_DEV_POSTGRES_CONTAINER"] = container_name
    managed_state["PLAP_DEV_POSTGRES_PORT"] = str(host_port)
    asyncio.run(_wait_for_database(database_url))


def _ensure_postgres_image(docker: str, image: str) -> None:
    result = subprocess.run(
        [docker, "image", "inspect", image],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return

    dockerfile_dir = REPO_ROOT / "tests" / "postgres"
    _run_checked([docker, "build", "-t", image, str(dockerfile_dir)], cwd=REPO_ROOT)


def _docker_inspect(docker: str, container_name: str) -> dict[str, object] | None:
    result = subprocess.run(
        [docker, "inspect", container_name],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    return payload[0] if payload else None


def _docker_host_port(inspected: dict[str, object]) -> int | None:
    network_settings = inspected.get("NetworkSettings")
    if not isinstance(network_settings, dict):
        return None
    ports = network_settings.get("Ports")
    if not isinstance(ports, dict):
        return None
    published = ports.get("5432/tcp")
    if not isinstance(published, list) or not published:
        return None
    first = published[0]
    if not isinstance(first, dict):
        return None
    host_port = first.get("HostPort")
    if not isinstance(host_port, str):
        return None
    return int(host_port)


def _container_uses_image(inspected: dict[str, object], image: str) -> bool:
    config = inspected.get("Config")
    if not isinstance(config, dict):
        return False
    configured_image = config.get("Image")
    return configured_image == image


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
    settings: Settings,
    managed_state: dict[str, str],
) -> str:
    engine = create_database_engine(settings.database_url)
    session_maker = create_session_maker(engine)
    manager = APIKeyManager(pepper=settings.api_key_pepper)
    email_address = managed_state.get("PLAP_DEV_USER_EMAIL", DEFAULT_DEV_EMAIL)
    organization_slug = managed_state.get("PLAP_DEV_ORG_SLUG", DEFAULT_DEV_ORG_SLUG)
    try:
        async with session_maker() as session:
            user = await _ensure_user(session, email_address)
            organization = await _ensure_organization(session, organization_slug)
            await _ensure_membership(session, user_id=user.id, organization_id=organization.id)
            issued_key = await manager.issue_key(
                session,
                user_id=user.id,
                organization_id=organization.id,
                name="local dev key",
            )
            await session.commit()
            return issued_key.plaintext_key
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


def _resolve_dev_model(settings: Settings) -> str:
    if DEFAULT_MODEL in settings.runtime_model_profiles:
        return DEFAULT_MODEL
    return next(iter(settings.runtime_model_profiles))


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
    print("Source this from another shell while the server is running: source .dev/dev.env")
    print(f"Explore logs: lnav {shlex.quote(str(log_file))}")
    print(
        f"Smoke test: curl http://{host}:{port}/v1/models -H 'Authorization: Bearer {api_key}'"
    )


def _run_server(args: argparse.Namespace, resources: EphemeralResources) -> int:
    uvicorn_command = [
        "uvicorn",
        "plap.app:create_app",
        "--factory",
        "--host",
        args.host,
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
    _remove_postgres_container(resources.postgres_container)
    _remove_state_file(resources.state_file)


def _stop_server(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _remove_postgres_container(container_name: str | None) -> None:
    if container_name is None:
        return
    docker = shutil.which("docker")
    if docker is None:
        return
    subprocess.run(
        [docker, "rm", "-f", container_name],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )


def _remove_state_file(path: Path) -> None:
    if path.exists():
        path.unlink()
    if path.parent.exists() and not any(path.parent.iterdir()):
        path.parent.rmdir()


def _run_checked(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


def _ensure_subprocess_pythonpath() -> None:
    current = os.environ.get("PYTHONPATH", "")
    paths = [path for path in current.split(os.pathsep) if path]
    for required in (str(SRC_ROOT), str(REPO_ROOT)):
        if required not in paths:
            paths.insert(0, required)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


if __name__ == "__main__":
    raise SystemExit(main())
