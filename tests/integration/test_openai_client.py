import pytest
from openai import AsyncOpenAI, AuthenticationError


@pytest.fixture
async def openai_client(live_server, seeded_auth_data):
    client = AsyncOpenAI(
        api_key=seeded_auth_data.api_key,
        base_url=f"{live_server.base_url}/v1",
        websocket_base_url=f"{live_server.websocket_base_url}/v1",
        max_retries=0,
    )
    yield client
    await client.close()


async def test_async_openai_client_http_methods(openai_client: AsyncOpenAI) -> None:
    created = await openai_client.responses.create(
        model="gpt-4.1",
        input="hello world",
        context_management=[{"type": "compaction", "compact_threshold": 128}],
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "name": "lookup_record",
                "parameters": {"type": "object"},
                "strict": True,
            },
            {"type": "web_search"},
        ],
    )
    retrieved = await openai_client.responses.retrieve(created.id)
    cancelled = await openai_client.responses.cancel(created.id)
    compacted = await openai_client.responses.compact(
        model="gpt-4.1", input="compact me"
    )
    input_items = await openai_client.responses.input_items.list(created.id)
    token_count = await openai_client.responses.input_tokens.count(
        model="gpt-4.1",
        input="count these tokens",
    )
    deleted = await openai_client.responses.delete(created.id)

    assert created.object == "response"
    assert created.id.startswith("resp_")
    assert any(item.type == "message" for item in created.output)
    assert any(item.type == "function_call" for item in created.output)
    assert not any(item.type == "web_search_call" for item in created.output)
    assert retrieved.id == created.id
    assert cancelled.status == "cancelled"
    assert compacted.object == "response.compaction"
    assert input_items.object == "list"
    assert input_items.data[0].type == "message"
    assert token_count.object == "response.input_tokens"
    assert token_count.input_tokens == 3
    assert deleted is None


async def test_async_openai_client_sse_stream(openai_client: AsyncOpenAI) -> None:
    stream = await openai_client.responses.create(
        model="gpt-4.1",
        input="hello stream",
        stream=True,
    )

    event_types = [event.type async for event in stream]

    assert event_types[0] == "response.created"
    assert "response.output_item.added" in event_types
    assert event_types[-1] == "response.completed"


async def test_async_openai_client_websocket(openai_client: AsyncOpenAI) -> None:
    manager = openai_client.responses.connect()
    connection = await manager.enter()

    try:
        await connection.send(
            {
                "type": "response.create",
                "response": {
                    "model": "gpt-4.1",
                    "input": "hello websocket",
                },
            }
        )

        event_types: list[str] = []
        while True:
            event = await connection.recv()
            event_types.append(event.type)
            if event.type == "response.completed":
                break
    finally:
        await connection.close()

    assert event_types[0] == "response.created"
    assert "response.output_item.added" in event_types
    assert event_types[-1] == "response.completed"


async def test_async_openai_client_rejects_invalid_api_key(live_server) -> None:
    client = AsyncOpenAI(
        api_key="plap_deadbeef_notavalidsecret",
        base_url=f"{live_server.base_url}/v1",
        websocket_base_url=f"{live_server.websocket_base_url}/v1",
        max_retries=0,
    )

    try:
        with pytest.raises(AuthenticationError):
            await client.responses.create(model="gpt-4.1", input="hello")
    finally:
        await client.close()
