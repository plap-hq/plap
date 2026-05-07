import pytest
from openai import APIStatusError, AsyncOpenAI, AuthenticationError


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
        model="plap/test",
        input="hello world",
        context_management=[{"type": "compaction", "soft_compact_threshold": 128}],
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "name": "lookup_record",
                "parameters": {"type": "object"},
                "strict": True,
            },
        ],
    )
    assert created.object == "response"
    assert created.id.startswith("resp_")
    assert created.output[0].type == "message"
    assert created.output[0].content[0].text == "test response"


async def test_async_openai_client_stateful_methods(openai_client: AsyncOpenAI) -> None:
    created = await openai_client.responses.create(model="plap/test", input="hello stateful")
    retrieved = await openai_client.responses.retrieve(created.id)
    input_items = await openai_client.responses.input_items.list(created.id)
    await openai_client.responses.delete(created.id)

    assert retrieved.id == created.id
    assert retrieved.output[0].content[0].text == created.output[0].content[0].text
    assert input_items.object == "list"
    assert len(input_items.data) == 1
    assert input_items.data[0].type == "message"
    assert input_items.data[0].content == "hello stateful"
    assert input_items.data[0].id.startswith(f"in_{created.id}_")

    with pytest.raises(APIStatusError) as missing:
        await openai_client.responses.retrieve(created.id)
    assert missing.value.status_code == 404


async def test_async_openai_client_models_list(openai_client: AsyncOpenAI) -> None:
    models = await openai_client.models.list()

    assert models.object == "list"
    assert len(models.data) == 1
    assert models.data[0].id == "plap/test"
    assert models.data[0].object == "model"
    assert models.data[0].owned_by == "plap"


async def test_async_openai_client_unsupported_methods(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(APIStatusError) as retrieved:
        await openai_client.responses.retrieve("resp_missing")
    with pytest.raises(APIStatusError) as input_items:
        await openai_client.responses.input_items.list("resp_missing")
    with pytest.raises(APIStatusError) as token_count:
        await openai_client.responses.input_tokens.count(
            model="plap/test",
            input="count these tokens",
        )
    with pytest.raises(APIStatusError) as deleted:
        await openai_client.responses.delete("resp_missing")

    assert retrieved.value.status_code == 404
    assert input_items.value.status_code == 404
    assert deleted.value.status_code == 404
    assert token_count.value.status_code == 400


async def test_async_openai_client_compact_method(openai_client: AsyncOpenAI) -> None:
    compacted = await openai_client.responses.compact(model="plap/test", input="compact me")

    assert compacted.object == "response.compaction"
    assert compacted.id.startswith("cmpresp_")
    assert compacted.output[0].type == "compaction"


async def test_async_openai_client_sse_stream(openai_client: AsyncOpenAI) -> None:
    stream = await openai_client.responses.create(
        model="plap/test",
        input="hello stream",
        stream=True,
    )

    event_types = [event.type async for event in stream]

    assert event_types[0] == "response.created"
    assert "response.in_progress" in event_types
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
                    "model": "plap/test",
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
    assert "response.in_progress" in event_types
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
            await client.responses.create(model="plap/test", input="hello")
    finally:
        await client.close()
