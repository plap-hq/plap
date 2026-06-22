"""CUE-based config loading and per-request resolution.

Usage::

    from plap.config import load

    loaded = load("config.cue", "schema.cue")
    result = loaded.plap.config.resolve(model="plap-ai/wisp", reasoning_effort="high")
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any

import msgspec
from box import Box


class PackageLess(Enum):
    ERROR = "error"
    GLOBAL = "global"
    IGNORE = "ignore"


class CueBox(Box):
    def resolve(self, request: Mapping[str, Any] | None = None, /, **kwargs: Any) -> Any:
        if request is not None and kwargs:
            raise TypeError("resolve accepts either one mapping or keyword arguments, not both")
        resolved_request = dict(kwargs if request is None else request)

        root = object.__getattribute__(self, "_cue_root")
        if root is None:
            raise TypeError("resolve is only available on top-level exported fields and their descendants")

        resolved_root = _resolve_root(root, resolved_request)
        path = object.__getattribute__(self, "_cue_path")
        value: Any = resolved_root
        for step in path:
            value = value[step]
        return value


_ENV_VAR = re.compile(r"\$\{(\w+)\}")
_PACKAGE_RE = re.compile(r"(?m)^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)\b")


def _resolve_env(value: object) -> object:
    if isinstance(value, str):
        return _ENV_VAR.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


def _package_name(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    match = _PACKAGE_RE.search(text)
    if match is None:
        return None
    name = match.group(1)
    return None if name == "_" else name


def _expand_cue_files(inputs: Iterable[str | Path]) -> list[Path]:
    files: set[Path] = set()
    for item in inputs:
        spec = str(item)
        if spec == "..." or spec.endswith("/..."):
            root = Path("." if spec == "..." else spec[:-4])
            candidates = root.rglob("*.cue")
        elif any(char in spec for char in "*?["):
            raw_path = Path(spec)
            if raw_path.is_absolute():
                root = Path(raw_path.anchor)
                pattern = str(raw_path.relative_to(root))
            else:
                root = Path()
                pattern = spec
            candidates = root.glob(pattern)
        else:
            path = Path(spec)
            candidates = path.rglob("*.cue") if path.is_dir() else [path]
        for path in candidates:
            if (
                path.is_file()
                and path.suffix == ".cue"
                and "cue.mod" not in path.parts
                and not path.name.endswith(("_tool.cue", "_test.cue"))
            ):
                files.add(path.resolve())
    return sorted(files)


def _cue_export(files: list[Path], expr: str | None = None, stdin: str | None = None) -> object:
    args = ["cue", "export", *map(str, files)]
    if expr is not None:
        args.extend(["-e", expr])
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
        raise RuntimeError(f"cue export failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def _box(value: dict[str, Any]) -> CueBox:
    return CueBox(value, frozen_box=True)


def _merge_requests(left: dict[str, str], right: dict[str, str]) -> dict[str, str] | None:
    merged = dict(left)
    for key, value in right.items():
        existing = merged.get(key)
        if existing is not None and existing != value:
            return None
        merged[key] = value
    return merged


def _reachable_requests(overlays: dict[str, Any]) -> list[dict[str, str]]:
    if not isinstance(overlays, dict) or not overlays:
        return [{}]

    axes: list[tuple[str, list[tuple[str, dict[str, Any]] | None]]] = []
    for axis in sorted(overlays):
        branches = overlays.get(axis)
        if not isinstance(branches, dict):
            continue
        options: list[tuple[str, dict[str, Any]] | None] = [None]
        for value, branch in sorted(branches.items()):
            if isinstance(value, str) and isinstance(branch, dict):
                options.append((value, branch))
        axes.append((axis, options))

    if not axes:
        return [{}]

    requests: dict[bytes, dict[str, str]] = {msgspec.json.encode({}, order="deterministic"): {}}
    for choices in product(*(options for _, options in axes)):
        selected: list[tuple[dict[str, str], dict[str, Any]]] = []
        for (axis, _), choice in zip(axes, choices, strict=True):
            if choice is None:
                continue
            value, branch = choice
            selected.append(({axis: value}, branch))

        request_variants: list[dict[str, str]] = [{}]
        for base_request, branch in selected:
            nested = branch.get("overlays")
            nested_variants = _reachable_requests(nested) if isinstance(nested, dict) else [{}]
            next_variants: list[dict[str, str]] = []
            for variant in request_variants:
                merged_base = _merge_requests(variant, base_request)
                if merged_base is None:
                    continue
                for nested_variant in nested_variants:
                    merged = _merge_requests(merged_base, nested_variant)
                    if merged is not None:
                        next_variants.append(merged)
            request_variants = next_variants

        for request in request_variants:
            requests.setdefault(msgspec.json.encode(request, order="deterministic"), request)
    return list(requests.values())


def _build_selections(request: dict[str, Any], overlays: dict[str, Any], expr: str) -> str:
    def collect(node: dict[str, Any], prefix: str) -> list[str]:
        selected: list[tuple[str, dict[str, Any]]] = []
        for axis in sorted(node):
            value = request.get(axis)
            branches = node.get(axis)
            if not isinstance(value, str) or not isinstance(branches, dict):
                continue
            branch = branches.get(value)
            if not isinstance(branch, dict):
                continue
            selected.append((f"{prefix}[{json.dumps(axis)}][{json.dumps(value)}]", branch))

        results = [branch_path for branch_path, _ in selected]
        for branch_path, branch in selected:
            nested = branch.get("overlays")
            if isinstance(nested, dict):
                results.extend(collect(nested, f"{branch_path}.overlays"))
        return results

    return " & ".join(collect(overlays, f"{expr}.overlays"))


def _annotate_descendants(value: object, *, root: CueBox, path: tuple[Any, ...]) -> None:
    if isinstance(value, CueBox):
        object.__setattr__(value, "_cue_root", root)
        object.__setattr__(value, "_cue_path", path)
        for key, child in value.items():
            _annotate_descendants(child, root=root, path=(*path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _annotate_descendants(child, root=root, path=(*path, index))


def _attach_resolve_root(root: CueBox, *, package: str, files: tuple[Path, ...], expr: str) -> None:
    object.__setattr__(root, "_cue_root", root)
    object.__setattr__(root, "_cue_path", ())
    object.__setattr__(root, "_cue_package", package)
    object.__setattr__(root, "_cue_files", files)
    object.__setattr__(root, "_cue_expr", expr)
    object.__setattr__(root, "_cue_cache", {})
    for key, child in root.items():
        _annotate_descendants(child, root=root, path=(key,))


def _attach_package(package_box: CueBox, *, package: str, files: tuple[Path, ...]) -> None:
    object.__setattr__(package_box, "_cue_root", None)
    object.__setattr__(package_box, "_cue_path", ())
    object.__setattr__(package_box, "_cue_package", package)
    object.__setattr__(package_box, "_cue_files", files)
    object.__setattr__(package_box, "_cue_expr", None)
    for key, child in package_box.items():
        if isinstance(child, CueBox):
            _attach_resolve_root(child, package=package, files=files, expr=str(key))


def _resolve_root(root: CueBox, request: dict[str, Any]) -> CueBox:
    cache = object.__getattribute__(root, "_cue_cache")
    request_key = msgspec.json.encode(request, order="deterministic")
    cached = cache.get(request_key)
    if cached is not None:
        return cached

    expr = object.__getattribute__(root, "_cue_expr")
    package = object.__getattribute__(root, "_cue_package")
    files = object.__getattribute__(root, "_cue_files")

    selections = _build_selections(request, root.get("overlays", {}), expr)
    prefix = f"package {package}\n\n"
    stdin = f"{prefix}resolved: {expr} & {selections}\n" if selections else f"{prefix}resolved: {expr}\n"
    result = _resolve_env(_cue_export(list(files), "resolved", stdin=stdin))
    if not isinstance(result, dict):
        raise TypeError(f"resolved config must be an object, got {type(result).__name__}")

    boxed = _box(result)
    _attach_resolve_root(boxed, package=package, files=files, expr=expr)
    cache[request_key] = boxed
    return boxed


def _precompute_all(root: CueBox) -> None:
    overlays = root.get("overlays", {})
    if not isinstance(overlays, dict) or not overlays:
        return

    for request in _reachable_requests(overlays):
        root.resolve(request)


def load(*paths: str | Path, package_less: PackageLess = PackageLess.ERROR) -> CueBox:
    files = _expand_cue_files(paths)

    grouped: dict[str, list[Path]] = defaultdict(list)
    package_less_files: list[Path] = []
    for path in files:
        package = _package_name(path)
        if package is None:
            package_less_files.append(path)
            continue
        grouped[package].append(path)

    if package_less_files and package_less is PackageLess.ERROR:
        names = "\n".join(f"  {path}" for path in package_less_files)
        raise ValueError(f"Found package-less CUE files:\n{names}")

    if not grouped:
        if package_less_files and package_less is PackageLess.GLOBAL:
            raise ValueError("package-less CUE files cannot be loaded globally without at least one named package")
        return _box({})

    shared = package_less_files if package_less is PackageLess.GLOBAL else []
    packages: dict[str, dict[str, Any]] = {}
    package_files_by_name: dict[str, tuple[Path, ...]] = {}

    for package, package_files in sorted(grouped.items()):
        selected_files = tuple(sorted([*shared, *package_files]))
        exported = _resolve_env(_cue_export(list(selected_files)))
        if not isinstance(exported, dict):
            raise TypeError(f"exported package {package!r} must be an object, got {type(exported).__name__}")
        packages[package] = exported
        package_files_by_name[package] = selected_files

    loaded = _box(packages)
    object.__setattr__(loaded, "_cue_root", None)
    object.__setattr__(loaded, "_cue_path", ())
    object.__setattr__(loaded, "_cue_package", None)
    object.__setattr__(loaded, "_cue_files", ())
    object.__setattr__(loaded, "_cue_expr", None)

    for package, package_box in loaded.items():
        if isinstance(package_box, CueBox):
            _attach_package(package_box, package=package, files=package_files_by_name[package])
            for child in package_box.values():
                if isinstance(child, CueBox) and object.__getattribute__(child, "_cue_root") is child:
                    _precompute_all(child)

    return loaded
