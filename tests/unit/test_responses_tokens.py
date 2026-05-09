from __future__ import annotations

import json

import transformers

from plap.llms.chat import ChatCompletionRequest, ChatFunctionTool, ChatMessage, ChatResponseFormat, ChatTool, ChatToolCall
from plap.responses import tokens as tokens_module
from plap.responses.ingest.render import compact_transcript, render_budgeted_spans
from plap.responses.models import ChatMessageSpan, StateMessage, StateToolCall
from plap.responses.tokens import measure_prompt_tokens, measure_request_tokens
from plap.settings import RuntimeActorConfig


def test_estimates_never_return_zero() -> None:
    empty_message = StateMessage(role="assistant", content="")
    cited_span = ChatMessageSpan(
        start=0,
        end=7,
        message=StateMessage(role="assistant", content="summary"),
        token_count=1,
    )

    assert empty_message.estimated_token_count() >= 1
    assert cited_span.citation_token_count() >= 1


def test_message_estimate_is_deterministic_for_key_order() -> None:
    first = StateMessage(role="user", content="hello", name="alice")
    second = StateMessage.from_primitive({"name": "alice", "content": "hello", "role": "user"})

    assert first.estimated_token_count() == second.estimated_token_count()


def test_measure_prompt_tokens_uses_model_visible_surface(monkeypatch) -> None:
    captured_payloads: list[str] = []

    def fake_estimate_text_tokens(text: str | None) -> int:
        assert text is not None
        captured_payloads.append(text)
        return len(text)

    monkeypatch.setattr("plap.responses.tokens.estimate_text_tokens", fake_estimate_text_tokens)

    measure_prompt_tokens(
        [
            ChatMessage(
                role="assistant",
                content="visible content",
                tool_call_id="tool_output_1",
                tool_calls=[ChatToolCall(id="tool_call_1", name="search", arguments='{"b":2,"a":1}')],
                reasoning_content="kept thinking",
                reasoning_details=[{"b": 2, "a": 1}],
            )
        ],
        actor_config=RuntimeActorConfig(model="crof/qwen3.5-9b"),
    )

    assert len(captured_payloads) == 1
    assert json.loads(captured_payloads[0]) == {
        "messages": [
            {
                "role": "assistant",
                "content": "visible content",
                "tool_call_id": "tool_output_1",
                "reasoning_content": "kept thinking",
                "reasoning_details": [{"a": 1, "b": 2}],
                "tool_calls": [
                    {
                        "id": "tool_call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"a":1,"b":2}',
                        },
                    }
                ],
            }
        ]
    }


def test_measure_request_tokens_uses_hf_chat_template_when_available(monkeypatch) -> None:
    captured_messages = None
    captured_tools = None
    encoded_reasoning: list[str] = []

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, response_format, add_generation_prompt, tokenize):
            nonlocal captured_messages, captured_tools
            captured_messages = messages
            captured_tools = tools
            assert response_format is None
            assert add_generation_prompt is False
            assert tokenize is True
            return [1, 2, 3]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            encoded_reasoning.append(text)
            return [7, 8]

    monkeypatch.setattr("plap.responses.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[
                ChatMessage(role="developer", content="system prompt"),
                ChatMessage(
                    role="assistant",
                    content="answer",
                    tool_calls=[ChatToolCall(id="call_1", name="search", arguments='{"query":"cats"}')],
                    reasoning_content="kept thinking",
                ),
                ChatMessage(role="tool", content="cats found", tool_call_id="call_1"),
            ],
            tools=[
                ChatTool(
                    function=ChatFunctionTool(
                        name="search",
                        description="Search docs",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                )
            ],
        ),
        actor_config=RuntimeActorConfig(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
    )

    assert count == 5
    assert captured_messages == [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": "answer",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": {"query": "cats"}}}],
        },
        {"role": "tool", "content": "cats found", "tool_call_id": "call_1"},
    ]
    assert captured_tools == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search docs",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                "strict": None,
            },
        }
    ]
    assert [json.loads(text) for text in encoded_reasoning] == [{"reasoning_content": "kept thinking"}]


def test_hf_tokenizer_uses_direct_fast_loader_for_step3p5(monkeypatch) -> None:
    tokens_module._hf_tokenizer.cache_clear()
    calls: list[tuple[str, str | None]] = []

    class _FakeTokenizer:
        pass

    def fake_fast_from_pretrained(cls, repo, revision=None, **kwargs):
        calls.append((repo, revision))
        assert kwargs == {}
        return _FakeTokenizer()

    def fake_auto_from_pretrained(cls, *args, **kwargs):
        raise AssertionError("AutoTokenizer should not be used for Step-3.5")

    monkeypatch.setattr(
        transformers.PreTrainedTokenizerFast,
        "from_pretrained",
        classmethod(fake_fast_from_pretrained),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(fake_auto_from_pretrained),
    )

    tokenizer = tokens_module._hf_tokenizer(
        "stepfun-ai/Step-3.5-Flash",
        "ab446a3de5e171ea341227e24bb1f090e1b771f7",
        False,
    )

    assert isinstance(tokenizer, _FakeTokenizer)
    assert calls == [("stepfun-ai/Step-3.5-Flash", "ab446a3de5e171ea341227e24bb1f090e1b771f7")]
    tokens_module._hf_tokenizer.cache_clear()


def test_measure_request_tokens_uses_dsv4_encoder_for_flash(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append(
            {
                "messages": messages,
                "thinking_mode": thinking_mode,
                "drop_thinking": drop_thinking,
                "reasoning_effort": reasoning_effort,
                "add_default_bos_token": add_default_bos_token,
            }
        )
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "dsv4-prompt"
            assert add_special_tokens is False
            return [1, 2, 3, 4]

    monkeypatch.setattr("plap.responses.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.responses.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="deepseek/deepseek-v4-flash",
            messages=[
                ChatMessage(role="developer", content="system prompt"),
                ChatMessage(
                    role="assistant",
                    content="answer",
                    tool_calls=[ChatToolCall(id="call_1", name="search", arguments='{"query":"cats"}')],
                    reasoning_content="kept thinking",
                ),
                ChatMessage(role="tool", content="cats found", tool_call_id="call_1"),
            ],
            tools=[
                ChatTool(
                    function=ChatFunctionTool(
                        name="search",
                        description="Search docs",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                )
            ],
        ),
        actor_config=RuntimeActorConfig(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
    )

    assert count == 4
    assert encoded == [
        {
            "messages": [
                {
                    "role": "developer",
                    "content": "system prompt",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "search",
                                "description": "Search docs",
                                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                                "strict": None,
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "answer",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"cats"}'},
                        }
                    ],
                    "reasoning_content": "kept thinking",
                },
                {"role": "tool", "content": "cats found", "tool_call_id": "call_1"},
            ],
            "thinking_mode": "thinking",
            "drop_thinking": False,
            "reasoning_effort": None,
            "add_default_bos_token": True,
        }
    ]


def test_measure_request_tokens_uses_dsv4_encoder_for_pro_and_maps_effort(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append((messages, thinking_mode, drop_thinking, reasoning_effort))
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

    monkeypatch.setattr("plap.responses.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.responses.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="deepseek/deepseek-v4-pro",
            messages=[ChatMessage(role="user", content="hello")],
            reasoning_effort="xhigh",
        ),
        actor_config=RuntimeActorConfig(model="deepseek/deepseek-v4-pro", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Pro"),
    )

    assert count == 1
    assert encoded == [([{"role": "user", "content": "hello"}], "thinking", True, "max")]


def test_measure_prompt_tokens_keeps_dsv4_thinking_for_tool_history_without_request_tools(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append((messages, thinking_mode, drop_thinking, reasoning_effort))
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

    monkeypatch.setattr("plap.responses.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.responses.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_prompt_tokens(
        [
            ChatMessage(role="user", content="hello"),
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ChatToolCall(id="call_1", name="search", arguments='{"query":"cats"}')],
                reasoning_content="kept thinking",
            ),
            ChatMessage(role="tool", content="cats found", tool_call_id="call_1"),
            ChatMessage(role="user", content="thanks"),
        ],
        actor_config=RuntimeActorConfig(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
    )

    assert count == 1
    assert encoded == [
        (
            [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"query":"cats"}'},
                        }
                    ],
                    "reasoning_content": "kept thinking",
                },
                {"role": "tool", "content": "cats found", "tool_call_id": "call_1"},
                {"role": "user", "content": "thanks"},
            ],
            "thinking",
            False,
            None,
        )
    ]


def test_measure_request_tokens_injects_system_for_user_only_dsv4_tools(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append(messages)
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

    monkeypatch.setattr("plap.responses.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.responses.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="deepseek/deepseek-v4-flash",
            messages=[ChatMessage(role="user", content="hello")],
            tools=[
                ChatTool(
                    function=ChatFunctionTool(
                        name="search",
                        description="Search docs",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                )
            ],
        ),
        actor_config=RuntimeActorConfig(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
    )

    assert count == 1
    assert encoded == [
        [
            {
                "role": "system",
                "content": "",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search docs",
                            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                            "strict": None,
                        },
                    }
                ],
            },
            {"role": "user", "content": "hello"},
        ]
    ]


def test_measure_request_tokens_injects_system_for_user_only_dsv4_response_format(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append(messages)
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

    monkeypatch.setattr("plap.responses.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.responses.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="deepseek/deepseek-v4-flash",
            messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
            response_format=ChatResponseFormat(
                type="json_schema",
                name="ok_response",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                strict=True,
            ),
        ),
        actor_config=RuntimeActorConfig(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
    )

    assert count == 1
    assert encoded == [
        [
            {
                "role": "system",
                "content": "",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ok_response",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            },
            {"role": "user", "content": 'Return {"ok": true}.'},
        ]
    ]


def test_measure_request_tokens_falls_back_when_dsv4_only_has_reasoning_details(monkeypatch) -> None:
    monkeypatch.setattr("plap.responses.tokens.encode_dsv4_messages", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    def fake_tokenize(text: str, *, actor_config):
        assert json.loads(text) == {
            "messages": [
                {
                    "role": "assistant",
                    "reasoning_details": [{"type": "reasoning.summary", "text": "hi"}],
                }
            ]
        }
        return 17

    monkeypatch.setattr("plap.responses.tokens._tokenize_text_with_actor", fake_tokenize)

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="deepseek/deepseek-v4-flash",
            messages=[ChatMessage(role="assistant", reasoning_details=[{"type": "reasoning.summary", "text": "hi"}])],
        ),
        actor_config=RuntimeActorConfig(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
    )

    assert count == 17


def test_render_budgeted_spans_tries_other_candidates_when_recount_rejects_best_heuristic_fit() -> None:
    first_leaf = ChatMessageSpan(
        start=0,
        end=0,
        message=StateMessage(role="user", content="alpha"),
        token_count=3,
    )
    second_leaf = ChatMessageSpan(
        start=1,
        end=1,
        message=StateMessage(role="assistant", content="beta"),
        token_count=3,
    )
    third_leaf = ChatMessageSpan(
        start=2,
        end=2,
        message=StateMessage(role="user", content="gamma"),
        token_count=2,
    )
    fourth_leaf = ChatMessageSpan(
        start=3,
        end=3,
        message=StateMessage(role="assistant", content="delta"),
        token_count=2,
    )
    first_summary = ChatMessageSpan(
        start=0,
        end=1,
        message=StateMessage(role="assistant", content="alpha beta"),
        token_count=3,
        children=(first_leaf, second_leaf),
        summary_fidelity=1,
    )
    second_summary = ChatMessageSpan(
        start=2,
        end=3,
        message=StateMessage(role="assistant", content="gamma delta"),
        token_count=3,
        children=(third_leaf, fourth_leaf),
        summary_fidelity=5,
    )

    def measure(spans: tuple[ChatMessageSpan, ...]) -> int:
        bounds = [(span.start, span.end) for span in spans]
        if bounds == [(0, 0), (1, 1), (2, 3)]:
            return 10
        if bounds == [(0, 1), (2, 2), (3, 3)]:
            return 9
        raise AssertionError(f"unexpected span layout: {bounds}")

    rendered = render_budgeted_spans(
        (first_summary, second_summary),
        measure=measure,
        recount_margin=3,
        token_budget=9,
    )

    assert [(span.start, span.end) for span in rendered] == [(0, 1), (2, 2), (3, 3)]


def test_render_budgeted_spans_expands_tool_output_summary_with_shared_segment_ordinal() -> None:
    prep_leaf = ChatMessageSpan(
        start=0,
        end=0,
        message=StateMessage(role="user", content="prep"),
        token_count=1,
    )
    assistant_leaf = ChatMessageSpan(
        start=1,
        end=1,
        message=StateMessage(
            role="assistant",
            content="",
            tool_calls=[StateToolCall(id="call_1", name="search", arguments='{"query":"cats"}')],
        ),
        token_count=1,
    )
    tool_leaf = ChatMessageSpan(
        start=1,
        end=1,
        message=StateMessage(role="tool", tool_call_id="call_1", content="cats found"),
        token_count=1,
    )
    after_leaf = ChatMessageSpan(
        start=2,
        end=2,
        message=StateMessage(role="user", content="after"),
        token_count=1,
    )
    call_summary = ChatMessageSpan(
        start=0,
        end=1,
        message=StateMessage(role="assistant", content="prep and tool call"),
        token_count=1,
        children=(prep_leaf, assistant_leaf),
        summary_fidelity=1,
    )
    output_summary = ChatMessageSpan(
        start=1,
        end=2,
        message=StateMessage(role="assistant", content="tool output and after"),
        token_count=1,
        children=(tool_leaf, after_leaf),
        summary_fidelity=5,
    )

    rendered = render_budgeted_spans((call_summary, output_summary), token_budget=4)
    transcript = compact_transcript(rendered)

    assert [(span.start, span.end) for span in rendered] == [(0, 0), (1, 1), (1, 1), (2, 2)]
    assert [message.to_primitive() for message in transcript] == [
        {"role": "user", "content": "prep"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "name": "search",
                    "arguments": {"query": "cats"},
                    "output": "cats found",
                }
            ],
        },
        {"role": "user", "content": "after"},
    ]
