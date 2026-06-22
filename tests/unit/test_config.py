from __future__ import annotations

from pathlib import Path

import pytest

import plap.config as config_mod
from plap.config import CueBox, PackageLess, load


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_cue_export(calls: list[tuple[tuple[str, ...], str, str | None]]):
    def run(files: list[Path], expr: str | None = None, stdin: str | None = None) -> object:
        names = tuple(path.name for path in files)
        calls.append((names, expr, stdin))

        if expr is None:
            if "alpha1.cue" in names:
                return {
                    "config": {
                        "name": "alpha",
                        "overlays": {
                            "reasoning_effort": {
                                "high": {
                                    "main": {"reasoning_effort": "high"},
                                }
                            },
                            "model": {
                                "plap-ai/wisp": {
                                    "main": {"model": "alpha-model"},
                                    "overlays": {
                                        "reasoning_effort": {
                                            "high": {
                                                "main": {"max_completion_tokens": 123},
                                            }
                                        },
                                        "service_tier": {
                                            "priority": {
                                                "main": {"service_tier": "priority"},
                                            }
                                        },
                                    },
                                },
                                "plap-ai/wisp-mini": {
                                    "main": {"model": "mini-model"},
                                    "overlays": {
                                        "reasoning_effort": {
                                            "high": {
                                                "main": {"max_completion_tokens": 999},
                                            }
                                        }
                                    },
                                },
                            },
                        },
                    },
                }
            if "beta1.cue" in names:
                return {
                    "config": {
                        "name": "beta",
                        "overlays": {
                            "model": {
                                "plap-ai/wisp": {
                                    "main": {"model": "beta-model"},
                                }
                            }
                        },
                    },
                }
            raise AssertionError(f"unexpected config export files: {names!r}")

        if expr == "resolved":
            return {"stdin": stdin}

        raise AssertionError(f"unexpected expr: {expr}")

    return run


def _reset_config_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _ = monkeypatch


def test_load_groups_files_by_package_and_applies_global_package_less(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "alpha1.cue", "package alpha\n")
    _write(tmp_path / "alpha2.cue", "package alpha\n")
    _write(tmp_path / "beta1.cue", "package beta\n")
    _write(tmp_path / "shared.cue", "x: 1\n")
    _write(tmp_path / "ignored_tool.cue", "package alpha\n")
    _write(tmp_path / "cue.mod" / "ignored.cue", "package alpha\n")

    calls: list[tuple[tuple[str, ...], str, str | None]] = []
    _reset_config_state(monkeypatch)
    monkeypatch.setattr(config_mod, "_cue_export", _fake_cue_export(calls))

    loaded = load(tmp_path, package_less=PackageLess.GLOBAL)

    assert set(loaded) == {"alpha", "beta"}
    assert isinstance(loaded, CueBox)
    assert loaded.alpha.config.name == "alpha"

    config_calls = [call for call in calls if call[1] is None]
    assert config_calls == [
        (("alpha1.cue", "alpha2.cue", "shared.cue"), None, None),
        (("beta1.cue", "shared.cue"), None, None),
    ]


def test_load_errors_on_package_less_files_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "alpha1.cue", "package alpha\n")
    _write(tmp_path / "shared.cue", "x: 1\n")

    _reset_config_state(monkeypatch)

    with pytest.raises(ValueError, match="package-less"):
        load(tmp_path, package_less=PackageLess.ERROR)


def test_resolve_uses_selected_package_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "alpha1.cue", "package alpha\n")
    _write(tmp_path / "beta1.cue", "package beta\n")

    calls: list[tuple[tuple[str, ...], str, str | None]] = []
    _reset_config_state(monkeypatch)
    monkeypatch.setattr(config_mod, "_cue_export", _fake_cue_export(calls))

    loaded = load(tmp_path)

    result = loaded.alpha.config.resolve({"model": "plap-ai/wisp", "reasoning_effort": "high"})

    assert result.stdin == (
        "package alpha\n\nresolved: config & "
        'config.overlays["model"]["plap-ai/wisp"] & '
        'config.overlays["reasoning_effort"]["high"] & '
        'config.overlays["model"]["plap-ai/wisp"].overlays["reasoning_effort"]["high"]\n'
    )


def test_loaded_package_resolve_accepts_keyword_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "alpha1.cue", "package alpha\n")

    calls: list[tuple[tuple[str, ...], str, str | None]] = []
    _reset_config_state(monkeypatch)
    monkeypatch.setattr(config_mod, "_cue_export", _fake_cue_export(calls))

    loaded = load(tmp_path)

    result = loaded.alpha.config.resolve(model="plap-ai/wisp")

    assert result.stdin == 'package alpha\n\nresolved: config & config.overlays["model"]["plap-ai/wisp"]\n'


def test_precompute_skips_unreachable_nested_only_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(tmp_path / "alpha1.cue", "package alpha\n")

    calls: list[tuple[tuple[str, ...], str, str | None]] = []
    _reset_config_state(monkeypatch)
    monkeypatch.setattr(config_mod, "_cue_export", _fake_cue_export(calls))

    load(tmp_path)

    resolved_calls = [call for call in calls if call[1] == "resolved"]
    assert any(
        call[2]
        == (
            "package alpha\n\nresolved: config & "
            'config.overlays["model"]["plap-ai/wisp"] & '
            'config.overlays["model"]["plap-ai/wisp"].overlays["service_tier"]["priority"]\n'
        )
        for call in resolved_calls
    )
    assert not any(
        call[2] == ('package alpha\n\nresolved: config & config.overlays["model"]["plap-ai/wisp"].overlays["service_tier"]["priority"]\n')
        for call in resolved_calls
    )
