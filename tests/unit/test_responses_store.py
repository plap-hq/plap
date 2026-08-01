from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

import pytest

from plap.auth import AuthContext
from plap.errors import PlapError
from plap.responses.contracts import RequestInputItem, RequestItemReference, ResponseCreateRequest
from plap.responses.store import ResponseStore


class _Database:
    @asynccontextmanager
    async def connection(self):
        yield object()


class _ResponseStore(ResponseStore):
    async def _conversation_head(self, connection, scope_id, conversation_id: str) -> str | None:
        _ = connection, scope_id
        assert conversation_id == "conv_123"
        return "resp_parent"

    async def _replay_items(self, connection, scope_id, response_id: str) -> list[RequestInputItem]:
        _ = connection, scope_id
        assert response_id == "resp_parent"
        return [RequestItemReference(id="msg_prior", type="item_reference")]

    async def _stored_item_payloads_by_id(self, connection, scope_id, item_ids: list[str]):
        _ = connection, scope_id
        assert item_ids == ["msg_prior", "msg_current"]
        return {
            "msg_prior": [{"content": "prior", "id": "msg_prior", "role": "user", "type": "message"}],
            "msg_current": [{"content": "current", "id": "msg_current", "role": "user", "type": "message"}],
        }


class _ReplayStatusStore(ResponseStore):
    def __init__(self, status: str) -> None:
        super().__init__(_Database())
        self._status = status

    async def _response_replay_status(self, connection, scope_id, response_id: str) -> str | None:
        _ = connection, scope_id
        assert response_id == "resp_parent"
        return self._status


def _auth_context() -> AuthContext:
    return AuthContext(api_key_id=uuid4(), organization_id=None, user_id=uuid4())


async def test_prepare_request_keeps_canonical_input_and_resolves_execution_history() -> None:
    reference = RequestItemReference(id="msg_current", type="item_reference")
    request = ResponseCreateRequest(
        conversation="conv_123",
        input=[reference],
        model="plap/test",
        store=True,
        temperature=0.4,
    )
    store = _ResponseStore(_Database())

    prepared = await store.prepare_request(_auth_context(), request)

    assert prepared.response_request.previous_response_id == "resp_parent"
    assert prepared.response_request.input == [reference]
    assert prepared.stored_input_items == [reference]
    assert prepared.execution_request.previous_response_id == "resp_parent"
    assert prepared.execution_request.temperature == 0.4
    assert isinstance(prepared.execution_request.input, list)
    assert [item.id for item in prepared.execution_request.input] == ["msg_prior", "msg_current"]
    assert [item.type for item in prepared.execution_request.input] == ["message", "message"]


@pytest.mark.parametrize("status", ["completed", "incomplete", "failed", "cancelled"])
async def test_terminal_responses_are_replayable(status: str) -> None:
    store = _ReplayStatusStore(status)

    await store._require_replayable_response(object(), uuid4(), "resp_parent", param="previous_response_id")


@pytest.mark.parametrize("status", ["queued", "in_progress"])
async def test_nonterminal_responses_are_not_replayable(status: str) -> None:
    store = _ReplayStatusStore(status)

    with pytest.raises(PlapError) as exc_info:
        await store._require_replayable_response(object(), uuid4(), "resp_parent", param="previous_response_id")

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "previous_response_not_replayable"
