import anyio
import pytest
from openai import APIStatusError, AsyncOpenAI, AuthenticationError

from plap.llms.completions.chat import ChatCompletionRequest, ChatCompletionResult, IChatCompletionClient

MODEL = "plap-ai/wisp"


class _FailingStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise RuntimeError("private-provider-secret")


class _FailingChatCompletionClient(IChatCompletionClient):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        _ = request
        raise RuntimeError("private-provider-secret")

    def stream(self, request: ChatCompletionRequest) -> _FailingStream:
        _ = request
        return _FailingStream()

    async def aclose(self) -> None:
        return None


class _BlockingStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        await anyio.sleep_forever()


class _BlockingChatCompletionClient(IChatCompletionClient):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        _ = request
        await anyio.sleep_forever()

    def stream(self, request: ChatCompletionRequest) -> _BlockingStream:
        _ = request
        return _BlockingStream()

    async def aclose(self) -> None:
        return None


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
    with anyio.fail_after(5):
        created = await openai_client.responses.create(
            model=MODEL,
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
            ],
        )
    assert created.object == "response"
    assert created.id.startswith("resp_")
    message_item = next(item for item in created.output if item.type == "message")
    assert message_item.content[0].text == "test response"


async def test_async_openai_client_stateful_methods(openai_client: AsyncOpenAI) -> None:
    with anyio.fail_after(5):
        created = await openai_client.responses.create(model=MODEL, input="hello stateful")
    retrieved = await openai_client.responses.retrieve(created.id)
    input_items = await openai_client.responses.input_items.list(created.id)
    await openai_client.responses.delete(created.id)

    assert retrieved.id == created.id
    retrieved_message = next(item for item in retrieved.output if item.type == "message")
    created_message = next(item for item in created.output if item.type == "message")
    assert retrieved_message.content[0].text == created_message.content[0].text
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
    assert [model.id for model in models.data] == ["plap-ai/mote", "plap-ai/wisp"]
    assert all(model.object == "model" for model in models.data)
    assert all(model.owned_by == "plap" for model in models.data)


async def test_async_openai_client_unsupported_methods(
    openai_client: AsyncOpenAI,
) -> None:
    with pytest.raises(APIStatusError) as retrieved:
        await openai_client.responses.retrieve("resp_missing")
    with pytest.raises(APIStatusError) as input_items:
        await openai_client.responses.input_items.list("resp_missing")
    with pytest.raises(APIStatusError) as token_count:
        await openai_client.responses.input_tokens.count(
            model=MODEL,
            input="count these tokens",
        )
    with pytest.raises(APIStatusError) as deleted:
        await openai_client.responses.delete("resp_missing")

    assert retrieved.value.status_code == 404
    assert input_items.value.status_code == 404
    assert deleted.value.status_code == 404
    assert token_count.value.status_code == 405


async def test_async_openai_client_compact_method(openai_client: AsyncOpenAI) -> None:
    with pytest.raises(APIStatusError) as not_implemented:
        await openai_client.responses.compact(model=MODEL, input="compact me")

    assert not_implemented.value.status_code == 501


async def test_async_openai_client_sse_stream(openai_client: AsyncOpenAI) -> None:
    stream = await openai_client.responses.create(
        model=MODEL,
        input="hello stream",
        stream=True,
    )

    event_types = [event.type async for event in stream]

    assert event_types[0] == "response.created"
    assert "response.in_progress" in event_types
    assert "response.output_item.added" in event_types
    assert event_types[-1] == "response.completed"


async def test_async_openai_client_sse_disconnect_cancels_response(live_server_factory, seeded_auth_data) -> None:
    with live_server_factory(chat_completion_client=_BlockingChatCompletionClient()) as server:
        stream_client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            websocket_base_url=f"{server.websocket_base_url}/v1",
            max_retries=0,
        )
        retrieval_client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            websocket_base_url=f"{server.websocket_base_url}/v1",
            max_retries=0,
        )
        try:
            stream = await stream_client.responses.create(model=MODEL, input="disconnect stream", stream=True)
            created = await anext(stream)
            await stream_client.close()

            with anyio.fail_after(5):
                while True:
                    response = await retrieval_client.responses.retrieve(created.response.id)
                    if response.status == "cancelled":
                        break
                    await anyio.sleep(0.01)
        finally:
            await stream_client.close()
            await retrieval_client.close()

    assert created.type == "response.created"
    assert response.status == "cancelled"


async def test_async_openai_client_rejects_unknown_model_before_streaming(openai_client: AsyncOpenAI) -> None:
    with pytest.raises(APIStatusError) as unknown_model:
        await openai_client.responses.create(model="plap-ai/missing", input="hello", stream=True)

    assert unknown_model.value.status_code == 404


async def test_async_openai_client_persists_and_streams_failed_responses(live_server_factory, seeded_auth_data) -> None:
    with live_server_factory(chat_completion_client=_FailingChatCompletionClient()) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            websocket_base_url=f"{server.websocket_base_url}/v1",
            max_retries=0,
        )
        try:
            failed = await client.responses.create(model=MODEL, input="fail once")
            retrieved = await client.responses.retrieve(failed.id)
            stream = await client.responses.create(model=MODEL, input="fail stream", stream=True)
            events = [event async for event in stream]
            stateless_stream = await client.responses.create(
                model=MODEL,
                input="fail stateless stream",
                include=["reasoning.encrypted_content"],
                store=False,
                stream=True,
            )
            stateless_events = [event async for event in stateless_stream]
            with pytest.raises(APIStatusError) as not_stored:
                await client.responses.retrieve(stateless_events[-1].response.id)
            continued = await client.responses.create(
                model=MODEL,
                input="continue failed",
                previous_response_id=failed.id,
            )
        finally:
            await client.close()

    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "server_error"
    assert "private-provider-secret" not in failed.model_dump_json()
    assert retrieved.status == "failed"
    assert retrieved.error == failed.error
    assert events[-1].type == "response.failed"
    assert events[-1].response.status == "failed"
    assert stateless_events[-1].type == "response.failed"
    assert stateless_events[-1].response.status == "failed"
    assert not_stored.value.status_code == 404
    assert continued.status == "failed"
    assert continued.previous_response_id == failed.id


async def test_async_openai_client_websocket(openai_client: AsyncOpenAI) -> None:
    manager = openai_client.responses.connect()
    connection = await manager.enter()

    try:
        await connection.send(
            {
                "type": "response.create",
                "response": {
                    "model": MODEL,
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
            await client.responses.create(model=MODEL, input="hello")
    finally:
        await client.close()
