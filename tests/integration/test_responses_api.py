import json
from collections.abc import AsyncIterator

import anyio
from litestar.testing import AsyncTestClient
from sqlalchemy import text

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.llms.chat import ChatCompletionDelta, ChatCompletionRequest, ChatCompletionResult, ChatMessage, IChatCompletionClient
from plap.responses.contracts import (
    ConversationReference,
    OutputTextContent,
    ReasoningConfig,
    ResponseCreateRequest,
    ResponseMessageItem,
    ResponseObject,
    ResponseReasoningItem,
)
from plap.responses.contracts.events import ResponseTextDoneEvent
from plap.responses.ingest import content_hash, seal_reasoning_payload
from plap.responses.io import ResponseEventIO
from plap.responses.models import ReasoningMessagePatch, ReasoningPayload, StateMessage
from plap.responses.projection import ResponseProjection
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.routes import _sse_payload
from plap.responses.store import ResponseStore
from plap.responses.tools import (
    IToolClassifier,
    ToolClassification,
    ToolSignature,
)


def _request_payload(stream: bool = False) -> dict[str, object]:
    return {
        "context_management": [{"soft_compact_threshold": 128, "type": "compaction"}],
        "input": [
            {
                "content": "hello from the client",
                "role": "user",
                "type": "message",
            }
        ],
        "model": "plap/test",
        "stream": stream,
        "tool_choice": "auto",
        "tools": [
            {
                "description": "Lookup a record",
                "name": "lookup_record",
                "parameters": {"type": "object"},
                "strict": True,
                "type": "function",
            },
        ],
    }


async def test_http_routes_require_bearer_auth(test_app) -> None:
    async with AsyncTestClient(app=test_app) as client:
        response = await client.post("/v1/responses", json={"model": "plap/test"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["type"] == "authentication_error"


async def test_authenticated_create_routes_return_model_output(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post("/v1/responses", json=_request_payload(), headers=headers)
        streamed = await client.post(
            "/v1/responses",
            json=_request_payload(stream=True),
            headers=headers,
        )

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["object"] == "response"
    assert body["status"] == "completed"
    message_item = next(item for item in body["output"] if item["type"] == "message")
    assert message_item["content"][0]["text"] == "test response"
    assert body["usage"] is None

    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "response.created" in streamed.text
    assert "response.in_progress" in streamed.text
    assert "response.completed" in streamed.text
    assert "response.output_item.added" in streamed.text
    assert "test response" in streamed.text


async def test_sse_stream_includes_obfuscation_by_default(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        streamed = await client.post(
            "/v1/responses",
            json=_request_payload(stream=True),
            headers=headers,
        )

    assert streamed.status_code == 200
    assert '"type":"response.output_text.delta"' in streamed.text
    assert '"obfuscation":"' in streamed.text


async def test_sse_stream_can_disable_obfuscation(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}
    payload = _request_payload(stream=True)
    payload["stream_options"] = {"include_obfuscation": False}

    async with AsyncTestClient(app=test_app) as client:
        streamed = await client.post(
            "/v1/responses",
            json=payload,
            headers=headers,
        )

    assert streamed.status_code == 200
    assert '"type":"response.output_text.delta"' in streamed.text
    assert '"obfuscation":"' not in streamed.text


async def test_model_routes_return_public_synthetic_metadata(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        models = await client.get("/v1/models", headers=headers)
        model_info = await client.get(
            "/v1/model/info",
            params={"model": "plap/test"},
            headers=headers,
        )

    assert models.status_code == 200
    assert models.json() == {
        "object": "list",
        "data": [
            {
                "id": "plap/test",
                "object": "model",
                "created": 0,
                "owned_by": "plap",
            }
        ],
    }

    info = model_info.json()
    assert model_info.status_code == 200
    assert info["object"] == "list"
    assert info["data"][0]["id"] == "plap/test"
    assert info["data"][0]["object"] == "model_info"
    assert info["data"][0]["display_name"] == "Test Model"
    assert info["data"][0]["provider"] == "plap"
    assert info["data"][0]["max_input_tokens"] == 8192
    assert "main" not in info["data"][0]
    assert "reviewer" not in info["data"][0]


async def test_unimplemented_response_routes_return_honest_errors(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        retrieved = await client.get("/v1/responses/resp_test", headers=headers)
        deleted = await client.delete("/v1/responses/resp_test", headers=headers)
        input_items = await client.get("/v1/responses/resp_test/input_items", headers=headers)
        input_tokens = await client.post(
            "/v1/responses/input_tokens",
            json={"input": "count these tokens", "model": "plap/test"},
            headers=headers,
        )

    assert retrieved.status_code == 404
    assert retrieved.json()["error"]["code"] == "response_not_found"
    assert deleted.status_code == 404
    assert deleted.json()["error"]["code"] == "response_not_found"
    assert input_items.status_code == 404
    assert input_items.json()["error"]["code"] == "response_not_found"
    assert _error_code(input_tokens) == (400, "unsupported_operation", "Response input token counting is not supported.")


async def test_compact_route_returns_compaction_output(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        compacted = await client.post(
            "/v1/responses/compact",
            json={"input": "compact me", "model": "plap/test"},
            headers=headers,
        )

    body = compacted.json()
    assert compacted.status_code == 200, compacted.text
    assert body["object"] == "response.compaction"
    assert body["id"].startswith("cmpresp_")
    assert body["output"][0]["type"] == "compaction"


async def test_stateful_response_routes_persist_retrieve_input_items_and_delete(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        created = await client.post("/v1/responses", json=_request_payload(), headers=headers)
        created_body = created.json()
        response_id = created_body["id"]

        retrieved = await client.get(f"/v1/responses/{response_id}", headers=headers)
        input_items = await client.get(f"/v1/responses/{response_id}/input_items", headers=headers)
        deleted = await client.delete(f"/v1/responses/{response_id}", headers=headers)
        missing = await client.get(f"/v1/responses/{response_id}", headers=headers)

    assert created.status_code == 200
    assert retrieved.status_code == 200
    assert retrieved.json()["id"] == response_id
    retrieved_message = next(item for item in retrieved.json()["output"] if item["type"] == "message")
    assert retrieved_message["content"][0]["text"] == "test response"

    input_items_body = input_items.json()
    assert input_items.status_code == 200
    assert input_items_body["object"] == "list"
    assert len(input_items_body["data"]) == 1
    assert input_items_body["data"][0]["type"] == "message"
    assert input_items_body["data"][0]["content"] == "hello from the client"
    assert input_items_body["data"][0]["id"].startswith(f"in_{response_id}_")

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "id": response_id, "object": "response"}
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "response_not_found"


async def test_create_response_requires_reasoning_encrypted_content_include_when_store_is_false(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "plap/test", "input": "hello", "store": False},
            headers=headers,
        )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "missing_reasoning_encrypted_content_include"
    assert body["error"]["param"] == "include"


async def test_previous_response_id_replays_persisted_history(
    test_app,
    seeded_auth_data,
    db_session_maker,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        first = await client.post(
            "/v1/responses",
            json={"model": "plap/test", "input": "first turn"},
            headers=headers,
        )
        first_id = first.json()["id"]
        second = await client.post(
            "/v1/responses",
            json={"model": "plap/test", "input": "second turn", "previous_response_id": first_id},
            headers=headers,
        )
        second_body = second.json()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second_body["previous_response_id"] == first_id

    async with db_session_maker() as session:
        replay_rows = (
            await session.execute(
                text(
                    """
                    select direction, payload
                      from responses.list_response_replay(:scope_id, :response_id)
                    """
                ),
                {"scope_id": seeded_auth_data.organization_id, "response_id": second_body["id"]},
            )
        ).all()

    assert [row.direction for row in replay_rows] == ["input", "output", "input", "output"]
    assert replay_rows[0].payload["content"] == "first turn"
    assert replay_rows[1].payload["content"][0]["text"] == "test response"
    assert replay_rows[2].payload["content"] == "second turn"


async def test_item_reference_inputs_expand_for_execution_and_persist_raw_references(
    test_app,
    seeded_auth_data,
    db_session_maker,
) -> None:
    response_store = ResponseStore(test_app.state.database)
    auth_context = AuthContext(
        api_key_id=seeded_auth_data.api_key_id,
        organization_id=seeded_auth_data.organization_id,
        user_id=seeded_auth_data.user_id,
    )
    first_message = StateMessage(role="assistant", content="first reply")
    first_message_hash = content_hash(first_message)

    async with db_session_maker() as session:
        await _append_response(
            session,
            seeded_auth_data.organization_id,
            "resp_first",
            input_items=[{"type": "message", "id": "in_resp_first_0", "role": "user", "content": "hello"}],
            output_items=[
                {
                    "type": "reasoning",
                    "id": "rs_first",
                    "status": "completed",
                    "summary": [],
                    "encrypted_content": seal_reasoning_payload(
                        ReasoningPayload(
                            side="main",
                            temp=False,
                            messages=(
                                ReasoningMessagePatch(
                                    content_hash=first_message_hash,
                                    reasoning_content="first thinking",
                                ),
                            ),
                        ),
                        keyring=test_app.state.sealing_keyring,
                    ),
                },
                {
                    "type": "message",
                    "id": "msg_first",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "first reply"}],
                },
            ],
        )
        await _append_response(
            session,
            seeded_auth_data.organization_id,
            "resp_second",
            input_items=[
                {"type": "item_reference", "id": "rs_first"},
                {"type": "item_reference", "id": "msg_first"},
                {"type": "message", "id": "in_resp_second_2", "role": "user", "content": "follow up"},
            ],
            output_items=[
                {
                    "type": "message",
                    "id": "msg_second",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "second reply"}],
                }
            ],
        )
        await session.commit()

    prepared_second = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(
            model="plap/test",
            input=[
                {"type": "item_reference", "id": "rs_first"},
                {"type": "item_reference", "id": "msg_first"},
                {"type": "message", "role": "user", "content": "follow up"},
            ],
        ),
    )
    prepared_third = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(model="plap/test", input="third turn", previous_response_id="resp_second"),
    )
    second_input_items = await response_store.list_input_items(
        auth_context,
        "resp_second",
        after=None,
        limit=None,
        order=None,
    )

    second_execution_input = prepared_second.execution_request.input
    assert [item.type for item in second_execution_input] == ["reasoning", "message", "message"]
    assert second_execution_input[0].id == "rs_first"
    assert second_execution_input[1].content[0].text == "first reply"
    assert second_execution_input[2].content == "follow up"

    third_execution_input = prepared_third.execution_request.input
    assert [item.type for item in third_execution_input] == ["reasoning", "message", "message", "message", "message"]
    assert third_execution_input[0].id == "rs_first"
    assert third_execution_input[1].content[0].text == "first reply"
    assert third_execution_input[2].content == "follow up"
    assert third_execution_input[3].content[0].text == "second reply"
    assert third_execution_input[4].content == "third turn"

    assert [item.type for item in second_input_items.data] == ["item_reference", "item_reference", "message"]
    assert [item.id for item in second_input_items.data[:2]] == ["rs_first", "msg_first"]
    assert second_input_items.data[2].content == "follow up"


async def test_retrieve_response_redacts_reasoning_encrypted_content_but_item_reference_replay_still_works(
    test_app,
    seeded_auth_data,
) -> None:
    response_store = ResponseStore(test_app.state.database)
    auth_context = AuthContext(
        api_key_id=seeded_auth_data.api_key_id,
        organization_id=seeded_auth_data.organization_id,
        user_id=seeded_auth_data.user_id,
    )
    prepared = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(model="plap/test", input="hello"),
    )
    first_message = StateMessage(role="assistant", content="first reply")
    sealed_reasoning = seal_reasoning_payload(
        ReasoningPayload(
            side="main",
            temp=False,
            messages=(
                ReasoningMessagePatch(
                    content_hash=content_hash(first_message),
                    reasoning_content="first thinking",
                ),
            ),
        ),
        keyring=test_app.state.sealing_keyring,
    )
    response = ResponseObject(
        created_at=0,
        id="resp_retrieve_reasoning",
        model="plap/test",
        output=[],
        status="in_progress",
    )
    reasoning_item = ResponseReasoningItem(
        encrypted_content=sealed_reasoning,
        id="rs_retrieve_reasoning",
        status="completed",
        summary=[],
        type="reasoning",
    )
    message_item = ResponseMessageItem(
        content=[OutputTextContent(text="first reply", type="output_text")],
        id="msg_retrieve_reasoning",
        role="assistant",
        status="completed",
        type="message",
    )
    await response_store.begin_response(prepared, response)
    await response_store.append_output_item(
        prepared,
        response.id,
        0,
        reasoning_item.model_dump(mode="json", exclude_none=True),
    )
    await response_store.append_output_item(
        prepared,
        response.id,
        1,
        message_item.model_dump(mode="json", exclude_none=True),
    )
    await response_store.finish_response(
        prepared,
        response.model_copy(
            update={
                "completed_at": 1,
                "output": [reasoning_item, message_item],
                "status": "completed",
            }
        ),
    )

    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}
    async with AsyncTestClient(app=test_app) as client:
        retrieved = await client.get(f"/v1/responses/{response.id}", headers=headers)
        retrieved_with_include = await client.get(
            f"/v1/responses/{response.id}",
            params=[("include", "reasoning.encrypted_content")],
            headers=headers,
        )

    assert retrieved.status_code == 200
    assert next(item for item in retrieved.json()["output"] if item["type"] == "reasoning").get("encrypted_content") is None
    assert retrieved_with_include.status_code == 200
    sealed_reasoning = next(
        item["encrypted_content"]
        for item in retrieved_with_include.json()["output"]
        if item["type"] == "reasoning"
    )

    prepared = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(
            model="plap/test",
            input=[{"type": "item_reference", "id": reasoning_item.id}],
        ),
    )

    assert prepared.execution_request.input is not None
    assert prepared.execution_request.input[0].type == "reasoning"
    assert prepared.execution_request.input[0].encrypted_content == sealed_reasoning


async def test_reasoning_item_reference_resolves_before_summary_finishes(
    test_app,
    seeded_auth_data,
) -> None:
    response_store = ResponseStore(test_app.state.database)
    auth_context = AuthContext(
        api_key_id=seeded_auth_data.api_key_id,
        organization_id=seeded_auth_data.organization_id,
        user_id=seeded_auth_data.user_id,
    )
    request = ResponseCreateRequest(
        model="plap/test",
        input="hello",
        include=["reasoning.encrypted_content"],
        reasoning=ReasoningConfig(summary="auto"),
    )
    prepared = await response_store.prepare_request(auth_context, request)
    projection = ResponseProjection.from_create_request(request, transport="stream")
    send, receive = anyio.create_memory_object_stream(16)
    summarizer = _BlockingReasoningSummarizer()
    assistant_message = StateMessage(role="assistant", content="reply")
    reasoning_patch = ReasoningMessagePatch(
        content_hash=content_hash(assistant_message),
        reasoning_content="private thinking",
    )
    reasoning_item = ResponseReasoningItem(
        encrypted_content=seal_reasoning_payload(
            ReasoningPayload(side="main", temp=False, messages=(reasoning_patch,)),
            keyring=test_app.state.sealing_keyring,
        ),
        id="rs_blocked_summary",
        status="completed",
        summary=[],
        type="reasoning",
    )
    out = ResponseEventIO(
        request=prepared.response_request,
        projection=projection,
        prepared=prepared,
        response_store=response_store,
        send=send,
        reasoning_summarizer=summarizer,
        reasoning_summarizer_model="plap/test-summarizer",
        reasoning_summarizer_prompt_cache_key_base=None,
        reasoning_summarizer_reasoning_effort=None,
        reasoning_summarizer_service_tier=None,
        reasoning_summary_mode="auto",
    )

    async with send, receive, anyio.create_task_group() as task_group:
        out.start(task_group)
        await out.created()
        await out.in_progress()
        await out.output(
            reasoning_item,
            reasoning_side="main",
            reasoning_messages=(reasoning_patch,),
        )
        await summarizer.entered.wait()

        replay = await response_store.prepare_request(
            auth_context,
            ResponseCreateRequest(
                model="plap/test",
                input=[{"type": "item_reference", "id": reasoning_item.id}],
            ),
        )

        assert replay.execution_request.input is not None
        assert replay.execution_request.input[0].type == "reasoning"
        assert replay.execution_request.input[0].id == reasoning_item.id

        summarizer.release.set()
        await out.aclose()


async def test_fail_response_does_not_advance_conversation_head_and_remains_explicitly_replayable(
    test_app,
    seeded_auth_data,
) -> None:
    response_store = ResponseStore(test_app.state.database)
    auth_context = AuthContext(
        api_key_id=seeded_auth_data.api_key_id,
        organization_id=seeded_auth_data.organization_id,
        user_id=seeded_auth_data.user_id,
    )
    conversation_id = "conv_failed"

    first_prepared = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(model="plap/test", input="first turn", conversation=conversation_id),
    )
    first_response = ResponseObject(
        conversation=ConversationReference(id=conversation_id),
        created_at=0,
        id="resp_conv_first",
        model="plap/test",
        output=[],
        previous_response_id=first_prepared.parent_response_id,
        status="in_progress",
    )
    first_message = ResponseMessageItem(
        content=[OutputTextContent(text="first reply", type="output_text")],
        id="msg_conv_first",
        role="assistant",
        status="completed",
        type="message",
    )
    await response_store.begin_response(first_prepared, first_response)
    await response_store.append_output_item(
        first_prepared,
        first_response.id,
        0,
        first_message.model_dump(mode="json", exclude_none=True),
    )
    await response_store.finish_response(
        first_prepared,
        first_response.model_copy(
            update={
                "completed_at": 1,
                "output": [first_message],
                "status": "completed",
            }
        ),
    )

    failed_prepared = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(model="plap/test", input="second turn", conversation=conversation_id),
    )
    assert failed_prepared.parent_response_id == first_response.id

    failed_response = ResponseObject(
        conversation=ConversationReference(id=conversation_id),
        created_at=0,
        id="resp_conv_failed",
        model="plap/test",
        output=[],
        previous_response_id=failed_prepared.parent_response_id,
        status="in_progress",
    )
    failed_message = ResponseMessageItem(
        content=[OutputTextContent(text="partial reply", type="output_text")],
        id="msg_conv_failed",
        role="assistant",
        status="completed",
        type="message",
    )
    await response_store.begin_response(failed_prepared, failed_response)
    await response_store.append_output_item(
        failed_prepared,
        failed_response.id,
        0,
        failed_message.model_dump(mode="json", exclude_none=True),
    )

    assert await response_store.fail_response(failed_prepared, failed_response.id) is True

    retrieved_failed = await response_store.get_response(auth_context, failed_response.id)
    continued = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(model="plap/test", input="third turn", conversation=conversation_id),
    )
    explicit = await response_store.prepare_request(
        auth_context,
        ResponseCreateRequest(model="plap/test", input="fourth turn", previous_response_id=failed_response.id),
    )

    assert retrieved_failed is not None
    assert retrieved_failed.status == "failed"
    assert [item.id for item in retrieved_failed.output] == [failed_message.id]
    assert continued.parent_response_id == first_response.id
    assert explicit.parent_response_id == failed_response.id


async def test_input_items_route_redacts_reasoning_encrypted_content_unless_included(
    test_app,
    seeded_auth_data,
    db_session_maker,
) -> None:
    sealed_reasoning = seal_reasoning_payload(
        ReasoningPayload(
            side="main",
            temp=False,
            messages=(
                ReasoningMessagePatch(
                    content_hash=content_hash(StateMessage(role="assistant", content="reply")),
                    reasoning_content="stored thinking",
                ),
            ),
        ),
        keyring=test_app.state.sealing_keyring,
    )

    async with db_session_maker() as session:
        await _append_response(
            session,
            seeded_auth_data.organization_id,
            "resp_input_reasoning",
            input_items=[
                {
                    "type": "reasoning",
                    "id": "rs_input_reasoning",
                    "status": "completed",
                    "summary": [],
                    "encrypted_content": sealed_reasoning,
                },
                {"type": "message", "id": "in_resp_input_reasoning_1", "role": "user", "content": "hello"},
            ],
            output_items=[
                {
                    "type": "message",
                    "id": "msg_input_reasoning",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "reply"}],
                }
            ],
        )
        await session.commit()

    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}
    async with AsyncTestClient(app=test_app) as client:
        input_items = await client.get("/v1/responses/resp_input_reasoning/input_items", headers=headers)
        input_items_with_include = await client.get(
            "/v1/responses/resp_input_reasoning/input_items",
            params=[("include", "reasoning.encrypted_content")],
            headers=headers,
        )

    assert input_items.status_code == 200
    assert input_items.json()["data"][0].get("encrypted_content") is None
    assert input_items_with_include.status_code == 200
    assert input_items_with_include.json()["data"][0]["encrypted_content"] == sealed_reasoning


async def test_create_response_rejects_missing_item_reference(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "plap/test", "input": [{"type": "item_reference", "id": "msg_missing"}]},
            headers=headers,
        )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "item_not_found"
    assert body["error"]["message"] == "Item with id 'msg_missing' not found."
    assert body["error"]["param"] == "input"


async def test_sse_payload_emits_error_event_on_late_stream_failure() -> None:
    async def events() -> AsyncIterator[ResponseTextDoneEvent]:
        yield ResponseTextDoneEvent(
            content_index=0,
            item_id="msg_1",
            output_index=0,
            sequence_number=1,
            text="partial",
            type="response.output_text.done",
        )
        raise PlapError(
            public=PublicError(
                status_code=400,
                type="invalid_request_error",
                code="provider_error",
                message="Provider returned error",
            ),
            private=PrivateError(
                event="response.invalid_request",
                reason="provider_bad_request",
                message="provider rejected request",
                level=ErrorLevel.WARNING,
            ),
        )

    chunks = [chunk async for chunk in _sse_payload(events())]

    assert '"type":"response.output_text.done"' in chunks[0]
    assert json.loads(chunks[1]) == {
        "type": "error",
        "sequence_number": 2,
        "code": "provider_error",
        "message": "Provider returned error",
    }
    assert chunks[2] == "[DONE]"


async def test_create_response_prepares_runtime_tools_without_changing_behavior(
    test_app_factory,
    seeded_auth_data,
) -> None:
    classifier = _RecordingToolClassifier()
    test_app = test_app_factory(tool_classifier=classifier)
    test_app.state.tool_policy_l1_cache.clear()
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post("/v1/responses", json=_request_payload(), headers=headers)

    assert response.status_code == 200, response.text
    assert response.json()["object"] == "response"
    assert any(item["type"] == "message" for item in response.json()["output"])
    assert classifier.tool_names == [["lookup_record"]]


async def test_http_validation_rejects_unsupported_context_management(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "context_management": [{"type": "retain_all"}],
                "model": "plap/test",
            },
            headers=headers,
        )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["message"] == "Invalid request."


async def test_create_response_sanitizes_ingestion_errors(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "input": [
                    {
                        "encrypted_content": "not-valid",
                        "id": "rs_bad",
                        "summary": [{"text": "bad", "type": "summary_text"}],
                        "type": "reasoning",
                    }
                ],
                "model": "plap/test",
            },
            headers=headers,
        )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["message"] == "Input replay items are invalid."
    assert body["error"]["code"] == "invalid_input_replay"
    assert body["error"]["param"] == "input"
    assert "sealed" not in body["error"]["message"]
    assert "encrypted" not in body["error"]["message"]


async def test_create_response_sanitizes_tool_preparation_errors(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post(
            "/v1/responses",
            json={
                "input": "search please",
                "model": "plap/test",
                "tools": [{"type": "web_search"}],
            },
            headers=headers,
        )

    body = response.json()
    assert response.status_code == 400
    assert body["error"]["message"] == "Web search is not available for this model."
    assert body["error"]["code"] == "unsupported_tool"
    assert body["error"]["param"] == "tools"
    assert "web_search" not in body["error"]["message"]
    assert "MCP" not in body["error"]["message"]


async def test_websocket_streams_response_events_with_auth(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        with await client.websocket_connect("/v1/responses", headers=headers) as socket:
            socket.send_json({"type": "response.create", "response": _request_payload()})
            event_types: list[str] = []

            while True:
                event = socket.receive_json()
                event_types.append(event["type"])
                if event["type"] == "response.completed":
                    break

    assert event_types[0] == "response.created"
    assert "response.in_progress" in event_types
    assert "response.output_item.added" in event_types
    assert event_types[-1] == "response.completed"


async def test_websocket_create_uses_runtime_validation(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        with await client.websocket_connect("/v1/responses", headers=headers) as socket:
            socket.send_json(
                {
                    "type": "response.create",
                    "response": {"input": "hello", "model": "unknown/model"},
                }
            )
            event = socket.receive_json()

    assert event["type"] == "error"
    assert event["code"] == "model_not_found"
    assert event["message"] == "Model 'unknown/model' not found."
    assert event["param"] == "model"


class _BlockingReasoningSummarizer(IReasoningSummarizer):
    def __init__(self) -> None:
        self.entered = anyio.Event()
        self.release = anyio.Event()

    async def stream(
        self,
        *,
        model: str,
        prompt_cache_key: str | None,
        reasoning_effort: object,
        service_tier: object,
        mode: str,
        side: str,
        messages,
    ) -> AsyncIterator[str]:
        _ = model, prompt_cache_key, reasoning_effort, service_tier, mode, side, messages
        self.entered.set()
        await self.release.wait()
        if False:
            yield ""


class _RecordingToolClassifier(IToolClassifier):
    classifier = "fake"
    classifier_model = "fake/model"
    prompt_hash = b"p" * 32

    def __init__(self) -> None:
        self.tool_names: list[list[str]] = []

    async def classify_many(self, signatures: list[ToolSignature]) -> dict[bytes, ToolClassification]:
        self.tool_names.append([str(signature.signature["name"]) for signature in signatures])
        return {
            signature.signature_hash: ToolClassification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                effect_class="safe",
                confidence=1.0,
                rationale="test classifier",
                raw_output={"effect_class": "safe"},
            )
            for signature in signatures
        }


class _RecordingChatCompletionClient(IChatCompletionClient):
    def __init__(self) -> None:
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.requests.append(request)
        return ChatCompletionResult(
            id=f"chatcmpl_{len(self.requests)}",
            model=request.model,
            created_at=None,
            message=ChatMessage(role="assistant", content=f"reply {len(self.requests)}"),
            finish_reason="stop",
        )

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        _ = request
        if False:
            yield ChatCompletionDelta(id="chatcmpl_test", model=None, created_at=None, choice_index=0)


async def _append_response(
    session,
    scope_id,
    response_id: str,
    *,
    prev_response_id: str | None = None,
    input_items: list[dict[str, object]] | None = None,
    output_items: list[dict[str, object]] | None = None,
) -> None:
    await session.execute(
        text(
            """
            select responses.append_response(
              :scope_id,
              :response_id,
              :prev_response_id,
              cast(:input_items as jsonb),
              cast(:output_items as jsonb),
              null
            )
            """
        ),
        {
            "scope_id": scope_id,
            "response_id": response_id,
            "prev_response_id": prev_response_id,
            "input_items": json.dumps(input_items or []),
            "output_items": json.dumps(output_items or []),
        },
    )


def _error_code(response) -> tuple[int, str | None, str]:
    body = response.json()
    return response.status_code, body["error"]["code"], body["error"]["message"]
