from litestar.testing import AsyncTestClient

from plap.responses.tools import (
    IToolClassifier,
    ToolClassification,
    ToolSignature,
)


def _request_payload(stream: bool = False) -> dict[str, object]:
    return {
        "context_management": [{"compact_threshold": 128, "type": "compaction"}],
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
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["text"] == "test response"
    assert body["usage"] is None

    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert "response.created" in streamed.text
    assert "response.in_progress" in streamed.text
    assert "response.completed" in streamed.text
    assert "response.output_item.added" in streamed.text
    assert "test response" in streamed.text


async def test_unimplemented_response_routes_return_honest_errors(
    test_app,
    seeded_auth_data,
) -> None:
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        retrieved = await client.get("/v1/responses/resp_test", headers=headers)
        deleted = await client.delete("/v1/responses/resp_test", headers=headers)
        compacted = await client.post(
            "/v1/responses/compact",
            json={"input": "compact me", "model": "plap/test"},
            headers=headers,
        )
        input_items = await client.get("/v1/responses/resp_test/input_items", headers=headers)
        input_tokens = await client.post(
            "/v1/responses/input_tokens",
            json={"input": "count these tokens", "model": "plap/test"},
            headers=headers,
        )

    assert _error_code(retrieved) == (404, "unsupported_operation")
    assert _error_code(deleted) == (404, "unsupported_operation")
    assert _error_code(input_items) == (404, "unsupported_operation")
    assert _error_code(compacted) == (501, "unsupported_operation")
    assert _error_code(input_tokens) == (501, "unsupported_operation")


async def test_create_response_prepares_runtime_tools_without_changing_behavior(
    test_app,
    seeded_auth_data,
) -> None:
    classifier = _RecordingToolClassifier()
    test_app.state.tool_classifier = classifier
    test_app.state.tool_policy_l1_cache.clear()
    headers = {"Authorization": f"Bearer {seeded_auth_data.api_key}"}

    async with AsyncTestClient(app=test_app) as client:
        response = await client.post("/v1/responses", json=_request_payload(), headers=headers)

    assert response.status_code == 200, response.text
    assert response.json()["object"] == "response"
    assert response.json()["output"][0]["type"] == "message"
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
    assert body["error"]["message"] == "Invalid request."
    assert body["error"]["code"] == "invalid_request"
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
    assert body["error"]["message"] == "Invalid request."
    assert body["error"]["code"] == "invalid_tool"
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
    assert event["message"] == "Invalid request."


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


def _error_code(response) -> tuple[int, str | None]:
    body = response.json()
    assert body["error"]["message"] == "Operation is not supported."
    return response.status_code, body["error"]["code"]
