from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import ValidationError

from plap.keyring import SealingKeyring
from plap.plugins.chat_completions.contracts import ChatCompletionCreateRequest
from plap.plugins.chat_completions.translation import chat_completion_stream, to_chat_completion, to_response_request
from plap.responses.contracts import (
    FunctionTool,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    OutputRefusalContent,
    OutputTextContent,
    OutputTextLogprob,
    OutputTextLogprobTopLogprob,
    ReasoningTextContent,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseTextDeltaEvent,
    ResponseTextEventLogprob,
    ResponseTextEventLogprobTopLogprob,
    ResponseUsage,
    ResponseUsageInputTokensDetails,
    ResponseUsageOutputTokensDetails,
    SummaryTextContent,
    TextFormatJSONSchema,
    ToolChoiceFunction,
)
from plap.responses.ingest.ingest import ingest_response_request
from plap.responses.ingest.models import Message, MessagePatch, ReasoningCheckpoint, ReasoningPayload, ToolCall
from plap.responses.ingest.sealing import seal_reasoning_payload


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"c" * 32,))


def _chat_request(messages: list[dict[str, object]], **values: object) -> ChatCompletionCreateRequest:
    return ChatCompletionCreateRequest.model_validate(
        {
            "messages": messages,
            "model": "plap/test",
            **values,
        }
    )


def _usage() -> ResponseUsage:
    return ResponseUsage(
        input_tokens=10,
        input_tokens_details=ResponseUsageInputTokensDetails(cached_tokens=2),
        output_tokens=6,
        output_tokens_details=ResponseUsageOutputTokensDetails(reasoning_tokens=3),
        total_tokens=16,
    )


def _response(*, output, status: str = "completed") -> ResponseObject:
    return ResponseObject(
        created_at=100.9,
        id="resp_test",
        model="plap/test",
        output=output,
        status=status,
        usage=_usage(),
    )


async def _events(*events) -> AsyncIterator:
    for event in events:
        yield event


def test_chat_request_maps_plain_transcript_and_tool_pair_without_reasoning() -> None:
    request = _chat_request(
        [
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "foreign_call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id":"1"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "foreign_call_1", "content": "record"},
        ]
    )

    response_request = to_response_request(request)

    assert response_request.store is False
    assert response_request.include == ["reasoning.encrypted_content"]
    assert response_request.input is not None
    assert isinstance(response_request.input[0], RequestMessageItem)
    assert response_request.input[0].content == "find it"
    assert isinstance(response_request.input[1], RequestMessageItem)
    assert response_request.input[1].content == ""
    assert isinstance(response_request.input[2], RequestFunctionCallItem)
    assert response_request.input[2].call_id == "foreign_call_1"
    assert isinstance(response_request.input[3], RequestFunctionCallOutputItem)
    assert response_request.input[3].call_id == "foreign_call_1"


def test_chat_request_preserves_store_and_logprob_controls() -> None:
    response_request = to_response_request(
        _chat_request(
            [{"role": "user", "content": "hello"}],
            logprobs=True,
            store=True,
            top_logprobs=3,
        )
    )

    assert response_request.store is True
    assert response_request.top_logprobs == 3


def test_chat_request_merges_compatible_reasoning_controls() -> None:
    summary_request = to_response_request(
        _chat_request(
            [{"role": "user", "content": "hello"}],
            reasoning={"summary": "concise"},
            reasoning_effort="max",
        )
    )
    duplicate_request = to_response_request(
        _chat_request(
            [{"role": "user", "content": "hello"}],
            reasoning={"effort": "high"},
            reasoning_effort="high",
        )
    )

    assert summary_request.reasoning is not None
    assert summary_request.reasoning.effort == "max"
    assert summary_request.reasoning.summary == "concise"
    assert duplicate_request.reasoning is not None
    assert duplicate_request.reasoning.effort == "high"


def test_chat_request_maps_multimodal_input_tools_and_structured_output() -> None:
    response_request = to_response_request(
        _chat_request(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "system instruction"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "inspect these"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"},
                        },
                        {
                            "type": "file",
                            "file": {"file_id": "file_1", "filename": "report.pdf"},
                        },
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
            tool_choice={"type": "function", "function": {"name": "lookup"}},
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a record.",
                        "parameters": {"type": "object"},
                        "strict": True,
                    },
                }
            ],
        )
    )

    assert response_request.input is not None
    system = response_request.input[0]
    user = response_request.input[1]
    assert isinstance(system, RequestMessageItem)
    assert system.content == [InputTextContent(text="system instruction", type="input_text")]
    assert isinstance(user, RequestMessageItem)
    assert user.content == [
        InputTextContent(text="inspect these", type="input_text"),
        InputImageContent(
            detail="high",
            image_url="data:image/png;base64,AA==",
            type="input_image",
        ),
        InputFileContent(file_id="file_1", filename="report.pdf", type="input_file"),
    ]
    assert response_request.tools == [
        FunctionTool(
            description="Look up a record.",
            name="lookup",
            parameters={"type": "object"},
            strict=True,
            type="function",
        )
    ]
    assert response_request.tool_choice == ToolChoiceFunction(name="lookup", type="function")
    assert response_request.text is not None
    assert response_request.text.format == TextFormatJSONSchema(
        name="result",
        schema={"type": "object"},
        strict=True,
        type="json_schema",
    )


def test_chat_request_rejects_conflicting_reasoning_efforts_and_empty_messages() -> None:
    with pytest.raises(ValidationError, match="cannot differ"):
        _chat_request(
            [{"role": "user", "content": "hello"}],
            reasoning={"effort": "low"},
            reasoning_effort="high",
        )
    with pytest.raises(ValidationError, match="at least 1"):
        _chat_request([])


async def test_unsealed_reasoning_metadata_replays_visible_assistant_as_fabricated() -> None:
    request = _chat_request(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": [
                    {
                        "type": "reasoning.summary",
                        "summary": "brief summary",
                    },
                    {
                        "type": "reasoning.text",
                        "text": "visible reasoning",
                    },
                ],
            },
            {"role": "user", "content": "second"},
        ]
    )

    response_request = to_response_request(request)
    assert response_request.input is not None
    assert not any(isinstance(item, RequestReasoningItem) for item in response_request.input)
    ingested = await ingest_response_request(
        response_request,
        keyring=_keyring(),
        thread_codes={"main": 0},
    )

    assert ingested.memory == {}
    assert ingested.threads["main"] == [
        Message(role="user", content="first"),
        Message(role="assistant", content="answer"),
        Message(role="user", content="second"),
    ]


def test_unsealed_reasoning_metadata_keeps_tool_call_assistant_anchor() -> None:
    response_request = to_response_request(
        _chat_request(
            [
                {"role": "user", "content": "find it"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "summary": "I should call the tool.",
                        }
                    ],
                    "tool_calls": [
                        {
                            "id": "foreign_call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":"1"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "foreign_call_1", "content": "record"},
            ]
        )
    )

    assert response_request.input is not None
    assert isinstance(response_request.input[1], RequestMessageItem)
    assert response_request.input[1].content == ""
    assert isinstance(response_request.input[2], RequestFunctionCallItem)
    assert isinstance(response_request.input[3], RequestFunctionCallOutputItem)


async def test_plain_foreign_tool_transcript_ingests_as_fabricated_main_history() -> None:
    request = _chat_request(
        [
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "foreign_call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id":"1"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "foreign_call_1", "content": "record"},
            {"role": "user", "content": "summarize it"},
        ]
    )

    ingested = await ingest_response_request(
        to_response_request(request),
        keyring=_keyring(),
        thread_codes={"main": 0},
    )

    assert ingested.memory == {}
    assert ingested.threads["main"] == [
        Message(role="user", content="find it"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="foreign_call_1", name="lookup", arguments='{"id":"1"}')],
        ),
        Message(role="tool", tool_call_id="foreign_call_1", content="record"),
        Message(role="user", content="summarize it"),
    ]


async def test_encrypted_reasoning_restores_hidden_state_and_private_assistant() -> None:
    keyring = _keyring()
    private = Message(role="assistant", content="answer", reasoning_content="hidden chain of thought")
    payload = ReasoningPayload(
        id="rs_test",
        previous_reasoning_id=None,
        previous_compaction_id=None,
        state=ReasoningCheckpoint(memory={"private": True}, enabled={"main"}, threads={}),
        main=[MessagePatch(message=private)],
    )
    encrypted = seal_reasoning_payload(payload, keyring=keyring)
    request = _chat_request(
        [
            {"role": "user", "content": "first"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_details": [
                    {
                        "type": "reasoning.summary",
                        "summary": "brief summary",
                        "id": "rs_test:summary:0",
                        "index": 0,
                    },
                    {
                        "type": "reasoning.text",
                        "text": "visible reasoning",
                        "id": "rs_test:text:0",
                        "index": 1,
                    },
                    {
                        "type": "reasoning.encrypted",
                        "data": encrypted,
                        "id": "rs_test",
                        "index": 2,
                    },
                ],
            },
            {"role": "user", "content": "second"},
        ]
    )

    response_request = to_response_request(request)
    assert response_request.input is not None
    assert isinstance(response_request.input[1], RequestReasoningItem)
    assert response_request.input[1].summary == [SummaryTextContent(text="brief summary", type="summary_text")]
    assert response_request.input[1].content == [ReasoningTextContent(text="visible reasoning", type="reasoning_text")]
    ingested = await ingest_response_request(
        response_request,
        keyring=keyring,
        thread_codes={"main": 0},
    )

    assert ingested.memory == {"private": True}
    assert ingested.threads["main"] == [
        Message(role="user", content="first"),
        private,
        Message(role="user", content="second"),
    ]


def test_response_projects_encrypted_state_text_refusal_tools_and_usage() -> None:
    reasoning = ResponseReasoningItem(
        content=[ReasoningTextContent(text="visible reasoning", type="reasoning_text")],
        encrypted_content="sealed-state",
        id="rs_test",
        status="completed",
        summary=[SummaryTextContent(text="brief summary", type="summary_text")],
        type="reasoning",
    )
    message = ResponseMessageItem(
        content=[
            OutputTextContent(
                annotations=[],
                logprobs=[
                    OutputTextLogprob(
                        bytes=[112],
                        logprob=-0.1,
                        token="p",
                        top_logprobs=[
                            OutputTextLogprobTopLogprob(bytes=[98], logprob=-1.0, token="b"),
                        ],
                    )
                ],
                text="partial",
                type="output_text",
            ),
            OutputRefusalContent(refusal="refused", type="refusal"),
        ],
        id="msg_test",
        role="assistant",
        status="completed",
        type="message",
    )
    call = ResponseFunctionCallItem(
        arguments='{"id":"1"}',
        call_id="sealed_call_1",
        id="fc_test",
        name="lookup",
        status="completed",
        type="function_call",
    )

    completion = to_chat_completion(_response(output=[reasoning, message, call]))

    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content == "partial"
    assert choice.message.refusal == "refused"
    assert choice.message.reasoning_details is not None
    assert [detail.type for detail in choice.message.reasoning_details] == [
        "reasoning.summary",
        "reasoning.text",
        "reasoning.encrypted",
    ]
    assert choice.message.reasoning_details[2].model_dump(exclude_none=True) == {
        "data": "sealed-state",
        "id": "rs_test",
        "index": 2,
        "type": "reasoning.encrypted",
    }
    assert choice.message.tool_calls is not None
    assert choice.message.tool_calls[0].id == "sealed_call_1"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens_details is not None
    assert completion.usage.prompt_tokens_details.cached_tokens == 2
    assert completion.usage.completion_tokens_details is not None
    assert completion.usage.completion_tokens_details.reasoning_tokens == 3
    assert choice.logprobs is not None
    assert choice.logprobs.content is not None
    assert choice.logprobs.content[0].token == "p"


async def test_stream_projects_reasoning_content_tool_calls_terminal_and_usage() -> None:
    created = _response(output=[], status="in_progress")
    reasoning = ResponseReasoningItem(
        encrypted_content="sealed-state",
        id="rs_test",
        status="completed",
        summary=[SummaryTextContent(text="brief summary", type="summary_text")],
        type="reasoning",
    )
    call = ResponseFunctionCallItem(
        arguments='{"id":"1"}',
        call_id="sealed_call_1",
        id="fc_test",
        name="lookup",
        status="completed",
        type="function_call",
    )
    completed = _response(output=[reasoning, call])
    payloads = [
        payload
        async for payload in chat_completion_stream(
            _events(
                ResponseCreatedEvent(
                    response=created,
                    sequence_number=1,
                    type="response.created",
                ),
                ResponseReasoningSummaryPartDoneEvent(
                    item_id="rs_test",
                    output_index=0,
                    part=SummaryTextContent(text="brief summary", type="summary_text"),
                    sequence_number=2,
                    summary_index=0,
                    type="response.reasoning_summary_part.done",
                ),
                ResponseOutputItemDoneEvent(
                    item=reasoning,
                    output_index=0,
                    sequence_number=3,
                    type="response.output_item.done",
                ),
                ResponseTextDeltaEvent(
                    content_index=0,
                    delta="hello",
                    item_id="msg_test",
                    logprobs=[
                        ResponseTextEventLogprob(
                            logprob=-0.1,
                            token="hello",
                            top_logprobs=[
                                ResponseTextEventLogprobTopLogprob(logprob=-1.0, token="hi"),
                            ],
                        )
                    ],
                    output_index=1,
                    sequence_number=4,
                    type="response.output_text.delta",
                ),
                ResponseOutputItemAddedEvent(
                    item=call,
                    output_index=2,
                    sequence_number=5,
                    type="response.output_item.added",
                ),
                ResponseFunctionCallArgumentsDeltaEvent(
                    delta='{"id":"1"}',
                    item_id="fc_test",
                    output_index=2,
                    sequence_number=6,
                    type="response.function_call_arguments.delta",
                ),
                ResponseCompletedEvent(
                    response=completed,
                    sequence_number=7,
                    type="response.completed",
                ),
            ),
            include_usage=True,
        )
    ]

    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert payloads[1]["choices"][0]["delta"]["reasoning_details"][0]["summary"] == "brief summary"
    assert payloads[2]["choices"][0]["delta"]["reasoning_details"][0]["data"] == "sealed-state"
    assert payloads[3]["choices"][0]["delta"]["content"] == "hello"
    assert payloads[3]["choices"][0]["logprobs"]["content"][0]["token"] == "hello"
    assert payloads[4]["choices"][0]["delta"]["tool_calls"][0]["id"] == "sealed_call_1"
    assert payloads[5]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"id":"1"}'
    assert payloads[6]["choices"][0]["finish_reason"] == "tool_calls"
    assert payloads[7]["choices"] == []
    assert payloads[7]["usage"]["total_tokens"] == 16
