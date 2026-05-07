from __future__ import annotations

from plap.llms.chat import ChatCompletionRequest, ChatFunctionTool, ChatMessage, ChatTool, ChatToolCall
from plap.responses.ingest.render import compact_transcript
from plap.responses.ingest.render import render_budgeted_spans
from plap.responses.models import ChatMessageSpan, StateMessage, StateToolCall, measure_prompt_tokens, measure_request_tokens
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

    monkeypatch.setattr("plap.responses.models._estimate_text_tokens", fake_estimate_text_tokens)

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

    assert captured_payloads == [
        'message[0]\n\nrole: assistant\ntool_call_id: tool_output_1\ncontent:\nvisible content\nreasoning_content:\nkept thinking\nreasoning_details:\n[{"a":1,"b":2}]\ntool_calls:\n[0]\nid: tool_call_1\nname: search\narguments:\n{"a":1,"b":2}'
    ]


def test_measure_request_tokens_uses_hf_chat_template_when_available(monkeypatch) -> None:
    captured_messages = None
    captured_tools = None
    encoded_reasoning: list[str] = []

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, add_generation_prompt, tokenize):
            nonlocal captured_messages, captured_tools
            captured_messages = messages
            captured_tools = tools
            assert add_generation_prompt is False
            assert tokenize is True
            return [1, 2, 3]

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            encoded_reasoning.append(text)
            return [7, 8]

    monkeypatch.setattr("plap.responses.models._hf_tokenizer", lambda *args: _FakeTokenizer())

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
    assert encoded_reasoning == ["reasoning_content:\nkept thinking"]


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


def test_render_budgeted_spans_expands_dependent_tool_output_summary() -> None:
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
        start=2,
        end=2,
        message=StateMessage(role="tool", tool_call_id="call_1", content="cats found"),
        token_count=1,
    )
    after_leaf = ChatMessageSpan(
        start=3,
        end=3,
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
        start=2,
        end=3,
        message=StateMessage(role="assistant", content="tool output and after"),
        token_count=1,
        children=(tool_leaf, after_leaf),
        summary_fidelity=5,
    )

    rendered = render_budgeted_spans((call_summary, output_summary), token_budget=4)
    transcript = compact_transcript(rendered)

    assert [(span.start, span.end) for span in rendered] == [(0, 0), (1, 1), (2, 2), (3, 3)]
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
