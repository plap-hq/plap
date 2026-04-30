from __future__ import annotations

import os
import socket
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import anyio
import pytest
import uvicorn
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import async_sessionmaker

from plap.app import create_app
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
from plap.settings import RuntimeModelProfileConfig, Settings
from tests.pytest_plugins.database import (
    _reset_database_schema,
    _run_migrations,
    _to_asyncpg_url,
)

pytestmark = pytest.mark.money

RUNTIME_PROFILE = "plap-ai/wisp-nano"
COMPRESSION_PROFILE = "plap-ai/wisp-nano-money-compress"
MONEY_MCP_TOOL_NAME = "money_search"
REQUIRED_ENV_KEYS = ("CROF_API_KEY", "LIGHTNING_API_KEY")


@dataclass(frozen=True, slots=True)
class _MoneyProviderKeys:
    crof_api_key: str
    lightning_api_key: str


@dataclass(frozen=True, slots=True)
class _MoneyLiveServer:
    base_url: str
    websocket_base_url: str


@dataclass(frozen=True, slots=True)
class _MoneyAuthData:
    api_key: str


@pytest.fixture(scope="session")
def money_provider_keys() -> _MoneyProviderKeys:
    _load_money_env()
    missing = [key for key in REQUIRED_ENV_KEYS if not os.getenv(key)]
    if missing:
        pytest.skip(f"missing money provider env keys: {', '.join(missing)}")
    return _MoneyProviderKeys(
        crof_api_key=os.environ["CROF_API_KEY"],
        lightning_api_key=os.environ["LIGHTNING_API_KEY"],
    )


@pytest.fixture(scope="session")
def money_settings(
    money_provider_keys: _MoneyProviderKeys,
    postgres_container,
) -> Settings:
    return Settings(
        api_key_pepper="money-test-pepper",
        database_url=_to_asyncpg_url(postgres_container.get_connection_url()),
        llm_crof_api_key=money_provider_keys.crof_api_key,
        llm_lightning_api_key=money_provider_keys.lightning_api_key,
        runtime_model_profiles={
            RUNTIME_PROFILE: _runtime_profile(),
            COMPRESSION_PROFILE: _runtime_profile(
                transcript_token_budget=2_000,
                compression_soft_token_budget=120,
                compression_hard_token_budget=180,
                compression_max_rounds=2,
            ),
        },
        sealing_keys=["a" * 43],
        web_search_mcp_config={
            "mcpServers": {
                "money": {
                    "command": sys.executable,
                    "args": [str(Path(__file__).with_name("_money_mcp_server.py"))],
                }
            }
        },
        web_search_mcp_tool_names=[MONEY_MCP_TOOL_NAME],
    )


@pytest.fixture(scope="session")
def money_database_schema(money_settings: Settings) -> None:
    async def reset_and_migrate() -> None:
        await _reset_database_schema(money_settings.database_url)
        await anyio.to_thread.run_sync(_run_migrations, money_settings.database_url)

    anyio.run(reset_and_migrate)


@pytest.fixture
async def money_session_maker(
    money_settings: Settings,
    money_database_schema: None,
) -> AsyncIterator[async_sessionmaker]:
    _ = money_database_schema
    engine = create_database_engine(money_settings.database_url)
    try:
        yield create_session_maker(engine)
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def money_live_server(
    money_settings: Settings,
    money_database_schema: None,
) -> _MoneyLiveServer:
    _ = money_database_schema
    port = _find_free_port()
    app = create_app(money_settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Uvicorn money test server did not start")

    yield _MoneyLiveServer(
        base_url=f"http://127.0.0.1:{port}",
        websocket_base_url=f"ws://127.0.0.1:{port}",
    )

    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
async def money_auth_data(
    money_session_maker: async_sessionmaker,
    money_settings: Settings,
) -> _MoneyAuthData:
    manager = APIKeyManager(pepper=money_settings.api_key_pepper)
    email_address = f"money.runtime.{uuid4().hex}@example.com"
    async with money_session_maker() as session:
        user = User(display_name="Money Runtime Test User")
        organization = Organization(
            slug=f"money-{uuid4().hex[:8]}",
            name="Money Runtime Test Org",
        )
        session.add_all([user, organization])
        await session.flush()

        email = UserEmail(
            user_id=user.id,
            email=email_address,
            normalized_email=normalize_email(email_address),
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
            slug="money-main",
            display_name="Money Test",
            provider_type="oidc",
            issuer="https://money.example.test/oauth2/default",
            metadata_url="https://money.example.test/.well-known/openid-configuration",
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
            provider_name="money",
            provider_subject=f"sub-{uuid4().hex}",
            email=email.normalized_email,
            claims={"groups": ["money-tests"]},
        )
        session.add(identity)

        issued_key: IssuedAPIKey = await manager.issue_key(
            session,
            user_id=user.id,
            organization_id=organization.id,
            name="money runtime key",
        )
        await session.commit()

    return _MoneyAuthData(api_key=issued_key.plaintext_key)


@pytest.fixture
async def money_openai_client(
    money_live_server: _MoneyLiveServer,
    money_auth_data: _MoneyAuthData,
) -> AsyncIterator[AsyncOpenAI]:
    client = AsyncOpenAI(
        api_key=money_auth_data.api_key,
        base_url=f"{money_live_server.base_url}/v1",
        websocket_base_url=f"{money_live_server.websocket_base_url}/v1",
        max_retries=0,
        timeout=120,
    )
    try:
        yield client
    finally:
        await client.close()


async def test_money_responses_wisp_nano_basic_completion(
    money_openai_client: AsyncOpenAI,
) -> None:
    response = await money_openai_client.responses.create(
        model=RUNTIME_PROFILE,
        input="Reply with one short sentence containing the word wisp.",
        max_output_tokens=256,
        temperature=0,
    )

    assert response.object == "response"
    assert response.status == "completed"
    assert _response_text(response).strip()


async def test_money_responses_wisp_nano_sse_stream(
    money_openai_client: AsyncOpenAI,
) -> None:
    stream = await money_openai_client.responses.create(
        model=RUNTIME_PROFILE,
        input="Reply with exactly: streaming wisp",
        max_output_tokens=256,
        stream=True,
        temperature=0,
    )

    event_types: list[str] = []
    completed_response = None
    async for event in stream:
        event_types.append(event.type)
        if event.type == "response.completed":
            completed_response = event.response

    assert event_types[0] == "response.created"
    assert "response.in_progress" in event_types
    assert "response.output_item.added" in event_types
    assert event_types[-1] == "response.completed"
    assert completed_response is not None
    assert _response_text(completed_response).strip()


async def test_money_responses_wisp_nano_client_tool_continuation_loop(
    money_openai_client: AsyncOpenAI,
) -> None:
    responses = await _run_client_tool_loop(
        money_openai_client,
        input=("Call get_constant_value first. After the tool result arrives, answer with the exact constant value from the tool."),
        tools=[_constant_tool_definition()],
        handlers={"get_constant_value": lambda _arguments: "constant value: 42"},
        tool_choice={"type": "function", "name": "get_constant_value"},
    )

    assert len(responses) == 2
    assert responses[-1].status == "completed"
    assert "42" in _response_text(responses[-1])


async def test_money_responses_wisp_nano_compression_replay_loop(
    money_openai_client: AsyncOpenAI,
) -> None:
    response = await money_openai_client.responses.create(
        model=COMPRESSION_PROFILE,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": (
                    "Project note alpha. Preserve marker RETAIN-MONEY-314. "
                    "This is repetitive context about a harmless runtime money "
                    "context replay test. " * 4
                ),
            },
            {
                "type": "message",
                "role": "assistant",
                "content": ("Acknowledged the marker RETAIN-MONEY-314 and the repetitive runtime replay notes. " * 4),
            },
            {
                "type": "message",
                "role": "user",
                "content": (
                    "Use these project notes to answer with the preserved marker RETAIN-MONEY-314 and keep the replay state useful. " * 4
                ),
            },
        ],
        max_output_tokens=512,
        temperature=0,
    )

    assert response.status == "completed"
    assert any(item.type == "compaction" for item in response.output)
    assert _response_text(response).strip()

    replay_input = [_item_to_input(item) for item in response.output]
    replay_input.append(
        {
            "type": "message",
            "role": "user",
            "content": "What marker did the compressed context preserve?",
        }
    )
    followup = await money_openai_client.responses.create(
        model=RUNTIME_PROFILE,
        input=replay_input,
        max_output_tokens=256,
        temperature=0,
    )

    assert followup.status == "completed"
    assert "RETAIN-MONEY-314" in _response_text(followup)


async def test_money_responses_wisp_nano_server_mcp_loopback(
    money_openai_client: AsyncOpenAI,
) -> None:
    response = await money_openai_client.responses.create(
        model=RUNTIME_PROFILE,
        input=("Use the search tool to find the runtime MCP marker, then answer with the exact marker string from the tool result."),
        max_output_tokens=512,
        temperature=0,
        tool_choice={"type": "function", "name": MONEY_MCP_TOOL_NAME},
        tools=[{"type": "web_search"}],
    )

    assert response.status == "completed"
    output_types = [item.type for item in response.output]
    assert "function_call" in output_types
    assert "function_call_output" in output_types
    server_outputs = [item for item in response.output if item.type == "function_call_output"]
    assert server_outputs[0].created_by == "server"
    assert "runtime-mcp-731" in server_outputs[0].output
    assert "runtime-mcp-731" in _response_text(response)


async def test_money_responses_wisp_nano_reasoning_summary(
    money_openai_client: AsyncOpenAI,
) -> None:
    response = await money_openai_client.responses.create(
        model=RUNTIME_PROFILE,
        input="Think through 17 + 25 and give a short final answer.",
        max_output_tokens=512,
        reasoning={"effort": "low", "summary": "concise"},
        temperature=0,
    )

    reasoning_items = [item for item in response.output if item.type == "reasoning"]
    assert reasoning_items
    assert reasoning_items[0].summary
    assert reasoning_items[0].summary[0].text.strip()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _response_text(response: object) -> str:
    return "".join(part.text for item in response.output if item.type == "message" for part in item.content if part.type == "output_text")


async def _run_client_tool_loop(
    client: AsyncOpenAI,
    *,
    input: str,
    tools: list[dict[str, object]],
    handlers: Mapping[str, Callable[[str], str]],
    tool_choice: dict[str, str] | None = None,
    max_turns: int = 4,
) -> list[object]:
    responses: list[object] = []
    next_input: str | list[dict[str, object]] = input
    next_tools: list[dict[str, object]] | None = tools
    next_tool_choice = tool_choice
    for _ in range(max_turns):
        response = await client.responses.create(
            model=RUNTIME_PROFILE,
            input=next_input,
            max_output_tokens=256,
            temperature=0,
            tool_choice=next_tool_choice,
            tools=next_tools,
        )
        responses.append(response)
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return responses
        replay_input = [_item_to_input(item) for item in response.output]
        for function_call in function_calls:
            handler = handlers[function_call.name]
            replay_input.append(
                {
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": handler(function_call.arguments),
                    "status": "completed",
                }
            )
        next_input = replay_input
        next_tools = None
        next_tool_choice = None
    raise AssertionError("client tool loop did not terminate")


def _item_to_input(item: object) -> dict[str, object]:
    value = item.model_dump(mode="json", exclude_none=True) if hasattr(item, "model_dump") else item.to_dict()
    if value.get("type") in {"compaction", "function_call_output"}:
        value.pop("created_by", None)
    return value


def _constant_tool_definition() -> dict[str, object]:
    return {
        "type": "function",
        "name": "get_constant_value",
        "description": ("Return a harmless constant string. This tool does not write data, change state, or contact external services."),
        "parameters": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "description": "Optional label for the constant.",
                }
            },
            "additionalProperties": False,
        },
        "strict": True,
    }


def _runtime_profile(
    *,
    transcript_token_budget: int = 200_000,
    compression_soft_token_budget: int | None = 100_000,
    compression_hard_token_budget: int | None = 150_000,
    compression_max_rounds: int = 3,
) -> RuntimeModelProfileConfig:
    return RuntimeModelProfileConfig(
        display_name="Wisp Nano",
        main_model="crof/qwen3.5-9b",
        main_debate_model="crof/qwen3.5-9b",
        reviewer_model="crof/qwen3.5-9b",
        arbitrator_model="crof/qwen3.5-9b",
        reasoning_summarizer_model="lightning/lightning-ai/gpt-oss-120b",
        transcript_token_budget=transcript_token_budget,
        compression_soft_token_budget=compression_soft_token_budget,
        compression_hard_token_budget=compression_hard_token_budget,
        compression_max_rounds=compression_max_rounds,
    )


def _load_money_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        os.environ.setdefault(key.strip(), _unquote(value.strip()))


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
