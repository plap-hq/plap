from __future__ import annotations

import json
from types import SimpleNamespace

import transformers

import plap.llms.completions.tokens as tokens_module
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatContentFile,
    ChatContentImage,
    ChatContentText,
    ChatFile,
    ChatFunctionTool,
    ChatImageURL,
    ChatMessage,
    ChatResponseFormat,
    ChatTool,
    ChatToolCall,
)
from plap.llms.completions.tokens import measure_prompt_tokens, measure_request_tokens


def _tokenizer_config(**kw: object) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "model": "unknown",
            "tokenizer_hf_repo": None,
            "tokenizer_revision": None,
            "tokenizer_trust_remote_code": False,
            **kw,
        }
    )


def test_measure_prompt_tokens_uses_model_visible_surface(monkeypatch) -> None:
    captured_payloads: list[str] = []

    def fake_estimate_text_tokens(text: str | None) -> int:
        assert text is not None
        captured_payloads.append(text)
        return len(text)

    monkeypatch.setattr("plap.llms.completions.tokens.estimate_text_tokens", fake_estimate_text_tokens)

    measure_prompt_tokens(
        [
            ChatMessage(
                role="assistant",
                content="visible content",
                tool_call_id="tool_output_1",
                tool_calls=[ChatToolCall(id="tool_call_1", name="search", arguments='{"b":2,"a":1}')],
                reasoning_content="kept thinking",
                memory={"advisor": {"call_id": "call_1"}},
            )
        ],
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b"),
    )

    assert len(captured_payloads) == 1
    assert json.loads(captured_payloads[0]) == {
        "messages": [
            {
                "role": "assistant",
                "content": "visible content",
                "tool_call_id": "tool_output_1",
                "reasoning_content": "kept thinking",
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


def test_token_surface_schema_values_place_required_immediately_before_properties() -> None:
    tool = ChatTool(
        function=ChatFunctionTool(
            name="apply_change",
            description="Apply a structured change.",
            parameters={
                "type": "object",
                "properties": {
                    "change": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                        },
                        "required": ["file_path", "old_string", "new_string"],
                    }
                },
                "required": ["change"],
            },
        )
    )
    response_format = ChatResponseFormat(
        type="json_schema",
        name="answer",
        schema={
            "type": "object",
            "properties": {
                "result": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                }
            },
            "required": ["result"],
        },
    )

    tool_schema = tokens_module._tool_definition_value(tool)["function"]["parameters"]
    assert list(tool_schema) == ["type", "required", "properties"]
    assert list(tool_schema["properties"]["change"]) == ["type", "required", "properties"]

    response_schema = tokens_module._response_format_value(response_format)["json_schema"]["schema"]
    assert list(response_schema) == ["type", "required", "properties"]
    assert list(response_schema["properties"]["result"]) == ["type", "required", "properties"]


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

    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

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
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
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


def test_measure_request_tokens_uses_native_template_for_structured_content_when_supported(monkeypatch) -> None:
    captured_messages = None

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, response_format, add_generation_prompt, tokenize):
            nonlocal captured_messages
            captured_messages = messages
            assert tools is None
            assert response_format is None
            assert add_generation_prompt is False
            assert tokenize is True
            return [1, 2, 3]

        def encode(self, text, add_special_tokens=False):
            raise AssertionError("reasoning encoding should not run")

    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        ChatContentText(text="look"),
                        ChatContentImage(
                            image_url=ChatImageURL(
                                url="https://example.com/image.png",
                                detail="original",
                                file_id="file_img_1",
                            )
                        ),
                    ],
                )
            ],
        ),
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
    )

    assert count == 3
    assert captured_messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png",
                        "detail": "high",
                        "file_id": "file_img_1",
                    },
                },
            ],
        }
    ]


def test_measure_request_tokens_lowers_developer_multimodal_content_before_template(monkeypatch) -> None:
    captured_messages = None

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, response_format, add_generation_prompt, tokenize):
            nonlocal captured_messages
            captured_messages = messages
            assert tools is None
            assert response_format is None
            assert add_generation_prompt is False
            assert tokenize is True
            return [1, 2, 3]

        def encode(self, text, add_special_tokens=False):
            raise AssertionError("reasoning encoding should not run")

    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[
                ChatMessage(
                    role="developer",
                    content=[
                        ChatContentText(text="preface"),
                        ChatContentImage(image_url=ChatImageURL(url="https://example.com/image.png", detail="original")),
                        ChatContentText(text="suffix"),
                    ],
                    name="planner",
                )
            ],
        ),
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
    )

    assert count == 3
    assert captured_messages == [
        {"role": "system", "content": [{"type": "text", "text": "preface"}], "name": "planner"},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png",
                        "detail": "high",
                    },
                }
            ],
        },
        {"role": "system", "content": [{"type": "text", "text": "suffix"}], "name": "planner"},
    ]


def test_measure_request_tokens_retries_template_with_projected_structured_content(monkeypatch) -> None:
    captured_messages = []

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, response_format, add_generation_prompt, tokenize):
            captured_messages.append(messages)
            assert tools is None
            assert response_format is None
            assert add_generation_prompt is False
            assert tokenize is True
            if isinstance(messages[0]["content"], list):
                raise TypeError("template cannot handle structured content")
            return [1, 2]

        def encode(self, text, add_special_tokens=False):
            raise AssertionError("reasoning encoding should not run")

    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        ChatContentText(text="look"),
                        ChatContentImage(
                            image_url=ChatImageURL(
                                url="https://example.com/image.png",
                                detail="original",
                                file_id="file_img_1",
                            )
                        ),
                        ChatContentFile(
                            file=ChatFile(
                                file_url="https://example.com/doc.pdf",
                                filename="doc.pdf",
                                detail="high",
                            )
                        ),
                    ],
                )
            ],
        ),
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
    )

    assert count == 2
    assert len(captured_messages) == 2
    assert isinstance(captured_messages[0][0]["content"], list)
    assert captured_messages[1] == [
        {
            "role": "user",
            "content": (
                "look\n\n"
                '{"image_url":{"detail":"high","file_id":"file_img_1","url":"https://example.com/image.png"},"type":"image_url"}'
                "\n\n"
                '{"file":{"file_url":"https://example.com/doc.pdf","filename":"doc.pdf"},"type":"file"}'
            ),
        }
    ]


def test_measure_request_tokens_falls_back_to_json_when_projected_template_still_fails(monkeypatch) -> None:
    captured_payloads: list[str] = []

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, response_format, add_generation_prompt, tokenize):
            _ = messages, tools, response_format, add_generation_prompt, tokenize
            raise TypeError("template cannot handle this content")

        def encode(self, text, add_special_tokens=False):
            raise AssertionError("template path should not succeed")

    def fake_tokenize_text(text: str, *, tokenizer_config) -> int:
        _ = tokenizer_config
        captured_payloads.append(text)
        return len(text)

    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())
    monkeypatch.setattr("plap.llms.completions.tokens._tokenize_text_with_tokenizer_config", fake_tokenize_text)

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        ChatContentText(text="look"),
                        ChatContentImage(image_url=ChatImageURL(url="https://example.com/image.png", detail="original")),
                    ],
                )
            ],
        ),
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
    )

    assert count == len(captured_payloads[0])
    assert json.loads(captured_payloads[0]) == {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.png",
                            "detail": "high",
                        },
                    },
                ],
            }
        ]
    }


def test_measure_request_tokens_keeps_invalid_template_tool_arguments_raw(monkeypatch) -> None:
    captured_messages = None

    class _FakeTokenizer:
        chat_template = "fake-template"

        def apply_chat_template(self, messages, *, tools, response_format, add_generation_prompt, tokenize):
            nonlocal captured_messages
            captured_messages = messages
            assert tools is None
            assert response_format is None
            assert add_generation_prompt is False
            assert tokenize is True
            return [1]

        def encode(self, text, add_special_tokens=False):
            raise AssertionError("reasoning encoding should not run")

    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[
                ChatMessage(
                    role="assistant", content="answer", tool_calls=[ChatToolCall(id="call_1", name="search", arguments="{'query':'cats'}")]
                )
            ],
        ),
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b", tokenizer_hf_repo="fake/repo"),
    )

    assert count == 1
    assert captured_messages == [
        {
            "role": "assistant",
            "content": "answer",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{'query':'cats'}"}}],
        }
    ]


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

    tokenizer = tokens_module._hf_tokenizer("stepfun-ai/Step-3.5-Flash", "ab446a3de5e171ea341227e24bb1f090e1b771f7", False)

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

    monkeypatch.setattr("plap.llms.completions.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

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
        tokenizer_config=_tokenizer_config(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
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


def test_measure_request_tokens_keeps_dsv4_path_for_structured_user_content(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append(messages)
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "dsv4-prompt"
            assert add_special_tokens is False
            return [1, 2]

    monkeypatch.setattr("plap.llms.completions.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    count = measure_request_tokens(
        ChatCompletionRequest(
            model="deepseek/deepseek-v4-flash",
            messages=[
                ChatMessage(role="developer", content="system prompt"),
                ChatMessage(
                    role="user",
                    content=[
                        ChatContentText(text="look"),
                        ChatContentImage(
                            image_url=ChatImageURL(
                                url="https://example.com/image.png",
                                detail="original",
                                file_id="file_img_1",
                            )
                        ),
                        ChatContentFile(
                            file=ChatFile(
                                file_url="https://example.com/doc.pdf",
                                filename="doc.pdf",
                                detail="high",
                            )
                        ),
                    ],
                ),
            ],
        ),
        tokenizer_config=_tokenizer_config(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
    )

    assert count == 2
    assert len(encoded) == 1
    assert encoded[0][0] == {"role": "developer", "content": "system prompt"}
    assert encoded[0][1]["role"] == "user"
    assert encoded[0][1]["content"] == (
        "look\n\n"
        '{"image_url":{"detail":"high","file_id":"file_img_1","url":"https://example.com/image.png"},"type":"image_url"}'
        "\n\n"
        '{"file":{"file_url":"https://example.com/doc.pdf","filename":"doc.pdf"},"type":"file"}'
    )


def test_measure_prompt_tokens_keeps_invalid_fallback_tool_arguments_raw(monkeypatch) -> None:
    captured_payloads: list[str] = []

    def fake_estimate_text_tokens(text: str | None) -> int:
        assert text is not None
        captured_payloads.append(text)
        return len(text)

    monkeypatch.setattr("plap.llms.completions.tokens.estimate_text_tokens", fake_estimate_text_tokens)

    measure_prompt_tokens(
        [
            ChatMessage(
                role="assistant",
                content="visible content",
                tool_calls=[ChatToolCall(id="tool_call_1", name="search", arguments="{'b':2,'a':1}")],
            )
        ],
        tokenizer_config=_tokenizer_config(model="crof/qwen3.5-9b"),
    )

    assert len(captured_payloads) == 1
    assert json.loads(captured_payloads[0]) == {
        "messages": [
            {
                "role": "assistant",
                "content": "visible content",
                "tool_calls": [
                    {
                        "id": "tool_call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{'b':2,'a':1}",
                        },
                    }
                ],
            }
        ]
    }


def test_measure_request_tokens_uses_dsv4_max_for_max_and_xhigh_effort(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append((messages, thinking_mode, drop_thinking, reasoning_effort))
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

    monkeypatch.setattr("plap.llms.completions.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

    for effort in ("max", "xhigh"):
        count = measure_request_tokens(
            ChatCompletionRequest(
                model="deepseek/deepseek-v4-pro", messages=[ChatMessage(role="user", content="hello")], reasoning_effort=effort
            ),
            tokenizer_config=_tokenizer_config(model="deepseek/deepseek-v4-pro", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Pro"),
        )

        assert count == 1
    assert encoded == [
        ([{"role": "user", "content": "hello"}], "thinking", True, "max"),
        ([{"role": "user", "content": "hello"}], "thinking", True, "max"),
    ]


def test_measure_prompt_tokens_keeps_dsv4_thinking_for_tool_history_without_request_tools(monkeypatch) -> None:
    encoded = []

    def fake_encode_messages(messages, thinking_mode, context=None, drop_thinking=True, add_default_bos_token=True, reasoning_effort=None):
        encoded.append((messages, thinking_mode, drop_thinking, reasoning_effort))
        return "dsv4-prompt"

    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

    monkeypatch.setattr("plap.llms.completions.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

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
        tokenizer_config=_tokenizer_config(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
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

    monkeypatch.setattr("plap.llms.completions.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

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
        tokenizer_config=_tokenizer_config(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
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

    monkeypatch.setattr("plap.llms.completions.tokens.encode_dsv4_messages", fake_encode_messages)
    monkeypatch.setattr("plap.llms.completions.tokens._hf_tokenizer", lambda *args: _FakeTokenizer())

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
        tokenizer_config=_tokenizer_config(model="deepseek/deepseek-v4-flash", tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash"),
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
