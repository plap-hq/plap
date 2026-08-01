from __future__ import annotations

import pytest

from scripts.dev import _completion_fields, _missing_provider_fields


def _field(model: str) -> dict[str, object]:
    return {
        "model": model,
        "sampling": {},
        "output_equivalence": {},
    }


def test_completion_field_discovery_is_structural() -> None:
    resolved = {
        "main": _field("openrouter/main"),
        "critic": _field("groq/critic"),
        "model_info": {"model": "display-only"},
        "other": {"model": "not-a-completion", "sampling": {}},
    }

    fields = _completion_fields(resolved)

    assert fields == {
        "main": resolved["main"],
        "critic": resolved["critic"],
    }


def test_one_provider_can_satisfy_multiple_discovered_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fields = _completion_fields(
        {
            "main": _field("openrouter/main"),
            "critic": _field("groq/critic,openrouter/critic"),
        }
    )

    assert _missing_provider_fields(fields) == {}


def test_missing_provider_fields_identifies_plugin_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    fields = _completion_fields(
        {
            "main": _field("openrouter/main"),
            "critic": _field("groq/critic"),
        }
    )

    assert _missing_provider_fields(fields) == {"critic": ("GROQ_API_KEY",)}
