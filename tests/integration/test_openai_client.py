import anyio
import pytest
from openai import APIStatusError, AsyncOpenAI, AuthenticationError

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatToolCallDelta,
    ChatUsage,
    IChatCompletionClient,
)

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


class _ReasoningChatCompletionClient(IChatCompletionClient):
    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []

    @staticmethod
    def _has_user_text(request: ChatCompletionRequest, text: str) -> bool:
        return any(message.role == "user" and message.content == text for message in request.messages)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        raise AssertionError(f"unexpected non-streaming provider call for {request.model}")

    async def stream(self, request: ChatCompletionRequest):
        self.requests.append(request)
        usage = ChatUsage(input_tokens=10, output_tokens=4, total_tokens=14, reasoning_tokens=2)
        if self._has_user_text(request, "second"):
            yield ChatCompletionDelta(
                id="chatcmpl_second",
                model=request.model,
                created_at=2,
                choice_index=0,
                content_delta="second answer",
            )
        elif self._has_user_text(request, "first"):
            yield ChatCompletionDelta(
                id="chatcmpl_first",
                model=request.model,
                created_at=1,
                choice_index=0,
                reasoning_delta="private reasoning",
            )
            yield ChatCompletionDelta(
                id="chatcmpl_first",
                model=request.model,
                created_at=1,
                choice_index=0,
                content_delta="first answer",
            )
        else:
            yield ChatCompletionDelta(
                id="chatcmpl_summary",
                model=request.model,
                created_at=1,
                choice_index=0,
                content_delta="reasoning summary",
            )
        yield ChatCompletionDelta(
            id="chatcmpl_done",
            model=request.model,
            created_at=2,
            choice_index=0,
            finish_reason="stop",
            usage=usage,
        )

    async def aclose(self) -> None:
        return None


class _ToolContinuationChatCompletionClient(IChatCompletionClient):
    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []
        self.advisor_calls = 0

    @staticmethod
    def _is_advisor_request(request: ChatCompletionRequest) -> bool:
        return any(tool.function.name == "advise" for tool in request.tools)

    @staticmethod
    def _has_tool_output(request: ChatCompletionRequest) -> bool:
        return any(message.role == "tool" for message in request.messages)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        raise AssertionError(f"unexpected non-streaming provider call for {request.model}")

    async def stream(self, request: ChatCompletionRequest):
        self.requests.append(request)
        if self._is_advisor_request(request):
            self.advisor_calls += 1
            yield ChatCompletionDelta(
                id="chatcmpl_advisor",
                model=request.model,
                created_at=2,
                choice_index=0,
                tool_call_delta=ChatToolCallDelta(
                    arguments_delta='{"advice":""}',
                    id=f"advisor_call_{self.advisor_calls}",
                    index=0,
                    name="advise",
                ),
            )
            finish_reason = "tool_calls"
        elif self._has_tool_output(request):
            yield ChatCompletionDelta(
                id="chatcmpl_tool_result",
                model=request.model,
                created_at=2,
                choice_index=0,
                content_delta="tool result",
            )
            finish_reason = "stop"
        else:
            yield ChatCompletionDelta(
                id="chatcmpl_tool_call",
                model=request.model,
                created_at=1,
                choice_index=0,
                tool_call_delta=ChatToolCallDelta(
                    arguments_delta='{"id":"1"}',
                    id="provider_call_1",
                    index=0,
                    name="lookup",
                ),
            )
            finish_reason = "tool_calls"
        yield ChatCompletionDelta(
            id="chatcmpl_tool_done",
            model=request.model,
            created_at=2,
            choice_index=0,
            finish_reason=finish_reason,
            usage=ChatUsage(input_tokens=8, output_tokens=3, total_tokens=11),
        )

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


async def test_async_openai_chat_completion_round_trips_encrypted_reasoning(
    live_server_factory,
    seeded_auth_data,
) -> None:
    completion_client = _ReasoningChatCompletionClient()
    with live_server_factory(chat_completion_client=completion_client) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        try:
            first = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "first"}],
            )
            first_message = first.choices[0].message
            reasoning_details = first_message.reasoning_details
            second = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": first_message.content,
                        "reasoning_details": reasoning_details,
                    },
                    {"role": "user", "content": "second"},
                ],
            )
        finally:
            await client.close()

    assert first.object == "chat.completion"
    assert first_message.content == "first answer"
    assert [detail["type"] for detail in reasoning_details] == [
        "reasoning.summary",
        "reasoning.encrypted",
    ]
    assert second.choices[0].message.content == "second answer"
    second_request = next(
        request for request in completion_client.requests if _ReasoningChatCompletionClient._has_user_text(request, "second")
    )
    restored = next(message for message in second_request.messages if message.role == "assistant" and message.content == "first answer")
    assert restored.reasoning_content == "private reasoning"


async def test_async_openai_chat_completion_falls_back_when_encrypted_reasoning_is_dropped(
    live_server_factory,
    seeded_auth_data,
) -> None:
    completion_client = _ReasoningChatCompletionClient()
    with live_server_factory(chat_completion_client=completion_client) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        try:
            first = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "first"}],
            )
            first_message = first.choices[0].message
            reasoning_details = [detail for detail in first_message.reasoning_details if detail["type"] != "reasoning.encrypted"]
            second = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": first_message.content,
                        "reasoning_details": reasoning_details,
                    },
                    {"role": "user", "content": "second"},
                ],
            )
        finally:
            await client.close()

    assert [detail["type"] for detail in reasoning_details] == ["reasoning.summary"]
    assert second.choices[0].message.content == "second answer"
    second_request = next(
        request for request in completion_client.requests if _ReasoningChatCompletionClient._has_user_text(request, "second")
    )
    restored = next(message for message in second_request.messages if message.role == "assistant" and message.content == "first answer")
    assert restored.reasoning_content is None


async def test_async_openai_chat_stream_round_trips_reasoning_and_reports_usage(
    live_server_factory,
    seeded_auth_data,
) -> None:
    completion_client = _ReasoningChatCompletionClient()
    with live_server_factory(chat_completion_client=completion_client) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        try:
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "first"}],
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = [chunk async for chunk in stream]
            content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
            reasoning_details = [
                detail for chunk in chunks if chunk.choices for detail in (getattr(chunk.choices[0].delta, "reasoning_details", None) or [])
            ]
            second = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "first"},
                    {
                        "role": "assistant",
                        "content": content,
                        "reasoning_details": reasoning_details,
                    },
                    {"role": "user", "content": "second"},
                ],
            )
        finally:
            await client.close()

    assert chunks[0].choices[0].delta.role == "assistant"
    assert content == "first answer"
    assert [detail["type"] for detail in reasoning_details] == [
        "reasoning.summary",
        "reasoning.encrypted",
    ]
    assert next(choice.finish_reason for chunk in chunks for choice in chunk.choices if choice.finish_reason is not None) == "stop"
    assert chunks[-1].choices == []
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.total_tokens == (chunks[-1].usage.prompt_tokens + chunks[-1].usage.completion_tokens)
    assert chunks[-1].usage.completion_tokens_details.reasoning_tokens > 0
    assert second.choices[0].message.content == "second answer"


async def test_async_openai_chat_completion_rejects_tampered_reasoning(
    live_server_factory,
    seeded_auth_data,
) -> None:
    completion_client = _ReasoningChatCompletionClient()
    with live_server_factory(chat_completion_client=completion_client) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        try:
            first = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "first"}],
            )
            first_message = first.choices[0].message
            reasoning_details = [dict(detail) for detail in first_message.reasoning_details]
            encrypted = next(detail for detail in reasoning_details if detail["type"] == "reasoning.encrypted")
            encrypted["data"] = "tampered"

            with pytest.raises(APIStatusError) as rejected:
                await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "user", "content": "first"},
                        {
                            "role": "assistant",
                            "content": first_message.content,
                            "reasoning_details": reasoning_details,
                        },
                        {"role": "user", "content": "tampered second"},
                    ],
                )
        finally:
            await client.close()

    assert rejected.value.status_code == 400
    assert rejected.value.body["code"] == "invalid_reasoning_replay"
    assert not any(_ReasoningChatCompletionClient._has_user_text(request, "tampered second") for request in completion_client.requests)


async def test_async_openai_chat_completion_round_trips_sealed_tool_call(
    live_server_factory,
    seeded_auth_data,
) -> None:
    completion_client = _ToolContinuationChatCompletionClient()
    with live_server_factory(chat_completion_client=completion_client) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one record.",
                "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
            },
        }
        try:
            first = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "find it"}],
                tools=[tool],
            )
            first_call = first.choices[0].message.tool_calls[0]
            second = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "user", "content": "find it"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": first_call.id,
                                "type": "function",
                                "function": {
                                    "name": first_call.function.name,
                                    "arguments": first_call.function.arguments,
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": first_call.id, "content": "record"},
                ],
                tools=[tool],
            )
        finally:
            await client.close()

    assert first.choices[0].finish_reason == "tool_calls"
    assert first_call.id != "provider_call_1"
    assert second.choices[0].message.content == "tool result"
    second_request = next(
        request
        for request in completion_client.requests
        if not completion_client._is_advisor_request(request) and completion_client._has_tool_output(request)
    )
    assistant = next(message for message in second_request.messages if message.role == "assistant")
    tool_output = next(message for message in second_request.messages if message.role == "tool")
    assert assistant.content == ""
    assert assistant.tool_calls[0].id == "provider_call_1"
    assert tool_output.tool_call_id == "provider_call_1"


async def test_async_openai_chat_completion_redacts_provider_failure(
    live_server_factory,
    seeded_auth_data,
) -> None:
    with live_server_factory(chat_completion_client=_FailingChatCompletionClient()) as server:
        client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        try:
            with pytest.raises(APIStatusError) as failed:
                await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": "fail privately"}],
                )
        finally:
            await client.close()

    assert failed.value.status_code == 500
    assert "private-provider-secret" not in str(failed.value.body)


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


async def test_async_openai_chat_sse_disconnect_cancels_response(live_server_factory, seeded_auth_data) -> None:
    with live_server_factory(chat_completion_client=_BlockingChatCompletionClient()) as server:
        stream_client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        retrieval_client = AsyncOpenAI(
            api_key=seeded_auth_data.api_key,
            base_url=f"{server.base_url}/v1",
            max_retries=0,
        )
        try:
            stream = await stream_client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": "disconnect chat stream"}],
                store=True,
                stream=True,
            )
            created = await anext(stream)
            await stream_client.close()

            with anyio.fail_after(5):
                while True:
                    response = await retrieval_client.responses.retrieve(created.id)
                    if response.status == "cancelled":
                        break
                    await anyio.sleep(0.01)
        finally:
            await stream_client.close()
            await retrieval_client.close()

    assert created.choices[0].delta.role == "assistant"
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
