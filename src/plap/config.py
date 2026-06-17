"""Nickel-based config loading and per-request resolution.

Usage::

    from plap.config import load, resolve

    config = load("config.ncl")
    result = resolve(config, {"model": "plap-ai/wisp", "reasoning_effort": "high"})
"""

from __future__ import annotations

import os
import re
from itertools import combinations, product
from pathlib import Path
from typing import Any

import msgspec
import nickel

_ENV_VAR = re.compile(r"\$\{(\w+)\}")
_IMPORT = re.compile(r'import\s+"([^"]+)"(?:\s+as\s+\'(\w+))?')

_cached_source: list[str | None] = [None]
_resolve_cache: dict[tuple[bytes, bytes], dict[str, Any]] = {}


def _resolve_env(value: object) -> object:
    if isinstance(value, str):
        return _ENV_VAR.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _nickel(value: object) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, dict):
        items = ", ".join(f"{k} = {_nickel(v)}" for k, v in value.items())
        return f"{{{items}}}"
    if isinstance(value, list):
        items = ", ".join(_nickel(v) for v in value)
        return f"[{items}]"
    return "null"


def _resolve_imports(content: str, base: Path, *, _depth: int = 0) -> str:
    if _depth > 10:
        return content

    def _replace(m: re.Match[str]) -> str:
        target = (base / m.group(1)).resolve()
        if not target.exists():
            return m.group(0)
        resolved = _resolve_imports(target.read_text(), target.parent, _depth=_depth + 1)
        return f"({resolved})"

    return _IMPORT.sub(_replace, content)


def _read_sources(paths: list[Path]) -> str:
    parts = []
    for p in paths:
        raw = p.read_text()
        resolved = _resolve_imports(raw, p.parent)
        parts.append(f"({resolved})")
    return " & ".join(parts)


def _get_by_dims(config: dict[str, Any]) -> dict[str, list[str]]:
    dims = {}
    for key, val in config.items():
        if key.startswith("by_") and isinstance(val, dict):
            dims[key[3:]] = sorted(val.keys())
    return dims


def _flatten_paths(obj: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for key, val in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.extend(_flatten_paths(val, path))
        else:
            result.append((path, val))
    return result


def _build_program(paths: list[tuple[str, Any]]) -> str:
    source = _cached_source[0]
    if not paths:
        return f"(let acc = {source} in acc)"

    lines: list[str] = [f"let acc = {source} in"]
    for i, (path, val) in enumerate(paths):
        parts = path.split(".")
        nick_val = _nickel(val)
        prev = "acc" if i == 0 else f"s{i - 1}"
        if len(parts) == 1:
            lines.append(f"let s{i} = std.record.update {_nickel(parts[0])} {nick_val} {prev} in")
        else:
            parent_key = parts[0]
            sub_key = parts[1]
            lines.append(
                f"let t{i} = std.record.update {_nickel(sub_key)} {nick_val} (std.record.get_or {_nickel(parent_key)} {{}} {prev}) in"
            )
            lines.append(f"let s{i} = std.record.update {_nickel(parent_key)} t{i} {prev} in")

    last = f"s{len(paths) - 1}"
    lines.append(last)
    return f"({chr(10).join(' ' + line for line in lines)} {chr(10)})"


def _resolve(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    pkey = msgspec.json.encode(config, order="deterministic")
    rkey = msgspec.json.encode(request, order="deterministic")
    key = (pkey, rkey)

    cached = _resolve_cache.get(key)
    if cached is not None:
        return cached

    paths: list[tuple[str, Any]] = []
    for key_name, val in request.items():
        dim = f"by_{key_name}"
        if dim in config and val in config[dim]:
            paths.extend(_flatten_paths(config[dim][val]))

    program = _build_program(paths)

    result = msgspec.json.decode(nickel.run(program))
    _resolve_cache[key] = result
    return result


def _precompute_all(config: dict[str, Any]) -> None:
    dims = _get_by_dims(config)
    dim_names = sorted(dims)
    if not dim_names:
        return

    for r in range(len(dim_names) + 1):
        for subset in combinations(dim_names, r):
            values = [dims[n] for n in subset]
            for combo in product(*values):
                _resolve(config, dict(zip(subset, combo, strict=True)))


def load(*paths: str | Path) -> dict[str, Any]:
    resolved = [Path(p).resolve() for p in paths]
    sources = _read_sources(resolved)
    result = _resolve_env(msgspec.json.decode(nickel.run(sources)))
    _cached_source[0] = sources
    _precompute_all(result)
    return result


def resolve(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    return _resolve(config, request)
