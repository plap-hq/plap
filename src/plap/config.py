"""CUE-based config loading and per-request resolution.

Usage::

    from plap.config import load, resolve

    config = load("config.cue", "schema.cue")
    result = resolve(config, {"model": "plap-ai/wisp", "reasoning_effort": "high"})
"""

from __future__ import annotations

import json
import os
import subprocess
from itertools import combinations, product
from pathlib import Path
from typing import Any

import msgspec

_resolve_cache: dict[tuple[bytes, bytes], dict[str, Any]] = {}


def _cue_eval(paths: list[str], stdin: str | None = None) -> dict[str, Any]:
    args = ["cue", "eval", "--out", "json", *paths]
    if stdin is not None:
        args.append("-")
    proc = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = f"cue eval failed:\n{proc.stderr}"
        raise RuntimeError(msg)
    return json.loads(proc.stdout)


def _walk_overrides(dims: dict, prefix: str) -> dict[str, list[tuple[str, list[str]]]]:
    result: dict[str, list[tuple[str, list[str]]]] = {}
    for key, val in dims.items():
        if not isinstance(val, dict):
            continue
        entry_keys = sorted(val.keys())
        if entry_keys:
            path = f"{prefix}[{json.dumps(key)}]"
            result[key] = [*result.get(key, []), (path, entry_keys)]
        for entry_key, entry_val in val.items():
            if isinstance(entry_val, dict) and "overrides" in entry_val and isinstance(entry_val["overrides"], dict):
                nested_path = f"{prefix}[{json.dumps(key)}][{json.dumps(entry_key)}].overrides"
                nested = _walk_overrides(entry_val["overrides"], nested_path)
                for nk, nv in nested.items():
                    result[nk] = result.get(nk, []) + nv
    return result


def _discover_dims(config: dict[str, Any]) -> dict[str, list[tuple[str, list[str]]]]:
    return _walk_overrides(config.get("overrides", {}), "config.overrides")


def _build_selections(request: dict[str, str], dims: dict[str, list[tuple[str, list[str]]]]) -> str:
    items: list[str] = []
    for key, val in request.items():
        if key not in dims:
            continue
        sources = dims[key]
        scored: list[tuple[str, int]] = []
        for path, values in sources:
            if val in values:
                depth = path.count("[")
                matches_model = False
                if "model" in request:
                    model_path = f'config.overrides["model"][{json.dumps(request["model"])}]'
                    if model_path in path:
                        matches_model = True
                scored.append((path, depth * 10 + (1 if matches_model else 0)))
        if scored:
            best = max(scored, key=lambda x: x[1])
            items.append(f"{best[0]}[{json.dumps(val)}]")
    return " & ".join(items)


def load(*paths: str | Path) -> dict[str, Any]:
    def _expand(value: object) -> object:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {k: _expand(val) for k, val in value.items()}
        if isinstance(value, list):
            return [_expand(val) for val in value]
        return value

    resolved = [str(Path(p).resolve()) for p in paths]
    output = _cue_eval(resolved)
    result = _expand(output.get("config", output))
    _precompute_all(result, resolved)
    return result


def _precompute_all(config: dict[str, Any], base_args: list[str]) -> None:
    dims = _discover_dims(config)
    dim_names = sorted(dims)
    if not dim_names:
        return

    pkey = msgspec.json.encode(config, order="deterministic")

    for r in range(len(dim_names) + 1):
        for subset in combinations(dim_names, r):
            values = []
            for name in subset:
                sources = dims[name]
                seen: set[str] = set()
                for _, vals in sources:
                    seen.update(vals)
                values.append(sorted(seen))
            for combo in product(*values):
                request = dict(zip(subset, combo, strict=True))
                rkey = msgspec.json.encode(request, order="deterministic")
                key = (pkey, rkey)
                if key in _resolve_cache:
                    continue

                selections = _build_selections(request, dims)
                stdin = f"package plap\n\nresolved: config & {selections}\n" if selections else "package plap\n\nresolved: config\n"
                output = _cue_eval(base_args, stdin=stdin)
                _resolve_cache[key] = output["resolved"]


def resolve(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    pkey = msgspec.json.encode(config, order="deterministic")
    rkey = msgspec.json.encode(request, order="deterministic")
    key = (pkey, rkey)

    cached = _resolve_cache.get(key)
    if cached is not None:
        return cached

    raise RuntimeError("resolve miss — call load() before resolve()")
