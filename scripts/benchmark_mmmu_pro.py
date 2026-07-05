#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import structlog
from datasets import load_dataset

from plap.app import _import_plugin, _plugin_names, _plugins
from plap.bus import bus
from plap.config import CueBox, load
from plap.llms.accumulator import Accumulator
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatContentImage,
    ChatContentPart,
    ChatContentText,
    ChatImageURL,
    ChatMessage,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatUsage,
    IChatCompletionClient,
)
from plap.llms.completions.client import ChatCompletionClient
from plap.llms.completions.providers import build_providers
from plap.llms.completions.providers.openai import OpenAIProvider
from plap.llms.completions.quirks import Drop, ExtraBodyIf, MoveMessageField, MoveOutput, SystemRole
from plap.llms.completions.router import ModelRoute, RoutingChatCompletionClient
from plap.llms.retry import (
    RetryValidator,
    retry_message,
    retry_on_tool_choice_mismatch,
    retry_on_unusable_tool_calls,
)
from plap.llms.retry import (
    complete as retry_complete,
)
from plap.plugins.core.request import apply_float_transform, apply_int_transform
from plap.plugins.vision import (
    VISION_PROMPT,
    VISION_TOOL_NAME,
    _image_id,
    _normalize_image_ids,
    _rewrite_request,
    _tool_output_text,
    _vision_history_messages,
    _vision_request,
)
from plap.responses.contracts import ResponseCreateRequest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / ".dev" / "mmmu-pro-bench"
DATASET_NAME = "MMMU/MMMU_Pro"
DATASET_CONFIG = "standard (10 options)"
DIRECT_INSTRUCTION = "Answer with the option letter from the given choices directly."

logger = structlog.stdlib.get_logger(__name__)
_CONFIG_CACHE: CueBox | None = None


@dataclass(frozen=True, slots=True)
class EvalExample:
    id: str
    subject: str
    topic_difficulty: str | None
    question: str
    options: list[str]
    answer: str
    images: list[bytes] = field(repr=False)

    @property
    def image_count(self) -> int:
        return len(self.images)


@dataclass(frozen=True, slots=True)
class VisionOverride:
    base_url: str
    model: str
    api_key: str


def _options_formatted(options: list[str]) -> str:
    labels = [chr(ord("A") + index) for index in range(len(options))]
    return "\n".join(f"{label}. {option}" for label, option in zip(labels, options, strict=True))


def _choice_letters(options: list[str]) -> list[str]:
    return [chr(ord("A") + index) for index in range(len(options))]


def _index_to_answer(options: list[str]) -> dict[str, str]:
    return dict(zip(_choice_letters(options), options, strict=True))


def _parse_multi_choice_response(
    response: str | None,
    options: list[str],
    *,
    rng: random.Random | None = None,
) -> str | None:
    if response is None:
        return None

    index_to_answer = _index_to_answer(options)
    all_choices = list(index_to_answer.keys())

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    normalized = response
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        normalized = normalized.strip(char)
    normalized = f" {normalized} "

    index_answer = True
    answer_with_brackets = False
    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in normalized:
            candidates.append(choice)
            answer_with_brackets = True

    if not candidates:
        candidates.extend(choice for choice in all_choices if f"{choice} " in normalized)

    if not candidates:
        candidates.extend(choice for choice in all_choices if f"{choice}." in normalized)

    if not candidates and len(normalized.split()) > 5:
        for index, answer in index_to_answer.items():
            if answer.lower() in normalized.lower():
                candidates.append(index)
                index_answer = False

    if not candidates:
        chooser = random.choice if rng is None else rng.choice
        return chooser(all_choices)

    if len(candidates) == 1:
        return candidates[0]

    start_indexes: list[int] = []
    if index_answer:
        if answer_with_brackets:
            start_indexes.extend(normalized.rfind(f"({candidate})") for candidate in candidates)
        else:
            start_indexes.extend(normalized.rfind(f" {candidate} ") for candidate in candidates)
    else:
        lowered = normalized.lower()
        start_indexes.extend(lowered.rfind(index_to_answer[candidate].lower()) for candidate in candidates)
    best_index = max(range(len(start_indexes)), key=start_indexes.__getitem__)
    return candidates[best_index]


def _pil_to_data_uri(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _vision_turn_prompt(ids: list[str], prompt: str) -> str:
    return f"Selected image ids: {', '.join(ids)}\nQuestion: {prompt}"


def _serialize_content(content: str | list[ChatContentPart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, ChatContentText):
            parts.append(part.text)
            continue
        if isinstance(part, ChatContentImage):
            url = part.image_url.url or ""
            parts.append(f"<image url={url[:120]}>")
            continue
        parts.append(repr(part))
    return "\n".join(parts)


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": message.role,
        "content": _serialize_content(message.content),
    }
    if message.name is not None:
        record["name"] = message.name
    if message.refusal is not None:
        record["refusal"] = message.refusal
    if message.tool_call_id is not None:
        record["tool_call_id"] = message.tool_call_id
    if message.reasoning_content is not None:
        record["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        record["tool_calls"] = [{"id": call.id, "name": call.name, "arguments": call.arguments} for call in message.tool_calls]
    return record


def _message_text(message: ChatMessage) -> str | None:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = [part.text for part in message.content if isinstance(part, ChatContentText) and part.text]
        if parts:
            return "".join(parts)
    if message.refusal is not None and message.refusal.strip():
        return message.refusal
    return None


def _usage_dict(usage: ChatUsage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    payload = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cached_tokens is not None:
        payload["cached_tokens"] = usage.cached_tokens
    if usage.reasoning_tokens is not None:
        payload["reasoning_tokens"] = usage.reasoning_tokens
    return payload


def _merge_usage(total: dict[str, int], usage: ChatUsage | None) -> dict[str, int]:
    payload = _usage_dict(usage)
    if payload is None:
        return total
    merged = dict(total)
    for key, value in payload.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
        file.write("\n")


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL in {path} at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"expected JSON object in {path} at line {line_number}")
            records.append(payload)
    return records


def _completed_example_ids(path: Path) -> set[str]:
    completed: set[str] = set()
    for record in _read_jsonl_records(path):
        example_id = record.get("example_id")
        if isinstance(example_id, str) and example_id:
            completed.add(example_id)
    return completed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _manifest_payload(
    *,
    args: argparse.Namespace,
    total_examples: int,
    resolved_main_model: str,
    resolved_vision_model: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "concurrency": args.concurrency,
        "dataset_config": DATASET_CONFIG,
        "dataset_name": DATASET_NAME,
        "max_examples": args.max_examples,
        "model": args.model,
        "resolved_main_model": resolved_main_model,
        "resolved_vision_model": resolved_vision_model,
        "save_transcripts": args.save_transcripts,
        "seed": args.seed,
        "split": args.split,
        "total_examples": total_examples,
    }
    if args.vision_base_url:
        payload["vision_override"] = {
            "base_url": args.vision_base_url,
            "model": args.vision_wire_model,
        }
    return payload


def _validate_or_write_manifest(path: Path, *, payload: dict[str, Any], resume: bool) -> None:
    if not path.exists():
        _write_json(path, payload)
        return

    with path.open(encoding="utf-8") as file:
        existing = json.load(file)
    if not isinstance(existing, dict):
        raise TypeError(f"manifest at {path} must be a JSON object")

    if not resume:
        raise SystemExit(f"run manifest already exists at {path}; use --resume or choose a new --run-dir")

    keys_to_match = (
        "dataset_name",
        "dataset_config",
        "split",
        "max_examples",
        "model",
        "resolved_main_model",
        "resolved_vision_model",
        "seed",
    )
    mismatches = [key for key in keys_to_match if existing.get(key) != payload.get(key)]
    if mismatches:
        mismatch_text = ", ".join(mismatches)
        raise SystemExit(f"existing manifest at {path} does not match this run for: {mismatch_text}")


def _summaries_by_key(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, int]] = {}
    for record in records:
        raw_value = record.get(key)
        value = raw_value if isinstance(raw_value, str) and raw_value else "unknown"
        bucket = grouped.setdefault(value, {"correct": 0, "errors": 0, "total": 0})
        bucket["total"] += 1
        if bool(record.get("correct")):
            bucket["correct"] += 1
        if isinstance(record.get("error"), str) and record["error"]:
            bucket["errors"] += 1
    return [
        {
            key: value,
            "accuracy": (bucket["correct"] / bucket["total"]) if bucket["total"] else 0.0,
            "correct": bucket["correct"],
            "errors": bucket["errors"],
            "total": bucket["total"],
        }
        for value, bucket in sorted(grouped.items())
    ]


def _build_summary(
    *,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    total_examples: int,
    resolved_main_model: str,
    resolved_vision_model: str,
) -> dict[str, Any]:
    correct = sum(1 for record in records if bool(record.get("correct")))
    errors = sum(1 for record in records if isinstance(record.get("error"), str) and record["error"])
    turns = [float(record.get("turns", 0)) for record in records if isinstance(record.get("turns"), int | float)]
    elapsed = [float(record.get("elapsed_seconds", 0.0)) for record in records if isinstance(record.get("elapsed_seconds"), int | float)]

    main_usage_total: dict[str, int] = {}
    vision_usage_total: dict[str, int] = {}
    for record in records:
        main_usage = record.get("main_usage")
        if isinstance(main_usage, dict):
            for key, value in main_usage.items():
                if isinstance(value, int):
                    main_usage_total[key] = main_usage_total.get(key, 0) + value
        vision_usage = record.get("vision_usage")
        if isinstance(vision_usage, dict):
            for key, value in vision_usage.items():
                if isinstance(value, int):
                    vision_usage_total[key] = vision_usage_total.get(key, 0) + value

    return {
        "accuracy": (correct / len(records)) if records else 0.0,
        "completed_examples": len(records),
        "concurrency": args.concurrency,
        "correct_examples": correct,
        "dataset_config": DATASET_CONFIG,
        "dataset_name": DATASET_NAME,
        "error_examples": errors,
        "main_usage": main_usage_total,
        "max_examples": args.max_examples,
        "mean_elapsed_seconds": _mean(elapsed),
        "mean_turns": _mean(turns),
        "model": args.model,
        "records_path": str((args.run_dir / "records.jsonl").resolve()),
        "resolved_main_model": resolved_main_model,
        "resolved_vision_model": resolved_vision_model,
        "save_transcripts": args.save_transcripts,
        "seed": args.seed,
        "split": args.split,
        "subjects": _summaries_by_key(records, "subject"),
        "difficulties": _summaries_by_key(records, "topic_difficulty"),
        "total_examples": total_examples,
        "vision_usage": vision_usage_total,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Model: {summary['model']}")
    print(f"Resolved main model: {summary['resolved_main_model']}")
    print(f"Resolved vision model: {summary['resolved_vision_model']}")
    print(f"Split: {summary['split']}")
    print(f"Concurrency: {summary['concurrency']}")
    print(f"Accuracy: {summary['correct_examples']}/{summary['completed_examples']} = {summary['accuracy']:.4f}")
    print(f"Errors: {summary['error_examples']}")
    print(f"Mean turns: {summary['mean_turns']:.3f}")
    print(f"Mean elapsed seconds: {summary['mean_elapsed_seconds']:.3f}")


def _load_mmmu_pro_examples(split: str, max_examples: int | None) -> list[EvalExample]:
    cache_key = hashlib.sha256(f"{DATASET_CONFIG}:full:v1:{split}:{max_examples}".encode()).hexdigest()[:16]
    cache_name = f"dataset_cache_{cache_key}.pkl"
    cache_path = DEFAULT_RUN_DIR / cache_name
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with cache_path.open("rb") as file:
            cached = pickle.load(file)
        if not isinstance(cached, list):
            raise TypeError(f"cached dataset at {cache_path} must be a list")
        return cached

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split)
    rows: list[EvalExample] = []
    for row in dataset:
        images: list[bytes] = []
        for key in ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"):
            image = row.get(key)
            if image is None:
                continue
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images.append(buffer.getvalue())
        rows.append(
            EvalExample(
                id=str(row["id"]),
                subject=str(row["subject"]),
                topic_difficulty=str(row.get("topic_difficulty")) if row.get("topic_difficulty") is not None else None,
                question=str(row["question"]),
                options=ast.literal_eval(str(row["options"])),
                answer=str(row["answer"]).strip().upper(),
                images=images,
            )
        )

    if max_examples is not None:
        rows = rows[:max_examples]

    with cache_path.open("wb") as file:
        pickle.dump(rows, file)
    return rows


def _build_user_content(question: str, options: list[str], images: list[bytes]) -> list[ChatContentPart]:
    full_text = f"{question}\n{_options_formatted(options)}\n{DIRECT_INSTRUCTION}"
    ordered_indexes = [int(value) for value in re.findall(r"<image\s+(\d+)>", full_text)]
    text = re.sub(r"<image\s+\d+>", "<image>", full_text)
    parts: list[ChatContentPart] = [ChatContentText(text=text)]
    for image_index in ordered_indexes:
        zero_index = image_index - 1
        if zero_index < 0 or zero_index >= len(images):
            raise ValueError(f"question referenced image {image_index}, but only {len(images)} images were available")
        parts.append(ChatContentImage(image_url=ChatImageURL(url=_pil_to_data_uri(images[zero_index]))))
    return parts


def _main_messages_for_example(example: EvalExample, *, display_name: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="developer", content=f"You are {display_name}, an AI assistant."),
        ChatMessage(role="user", content=_build_user_content(example.question, example.options, example.images)),
    ]


def _sampling_kwargs(field_config: CueBox) -> dict[str, object]:
    sampling = field_config.get("sampling")
    if sampling is None:
        return {}
    return {
        "temperature": apply_float_transform(None, sampling.get("temperature"), minimum=0, maximum=2),
        "top_p": apply_float_transform(None, sampling.get("top_p"), minimum=0, maximum=1),
        "min_p": apply_float_transform(None, sampling.get("min_p"), minimum=0, maximum=1),
        "top_k": apply_int_transform(None, sampling.get("top_k"), minimum=0),
        "frequency_penalty": apply_float_transform(None, sampling.get("frequency_penalty"), minimum=-2, maximum=2),
        "presence_penalty": apply_float_transform(None, sampling.get("presence_penalty"), minimum=-2, maximum=2),
        "repetition_penalty": apply_float_transform(None, sampling.get("repetition_penalty"), minimum=0, maximum=2),
        "seed": apply_int_transform(None, sampling.get("seed")),
    }


def _build_main_request(resolved: CueBox, *, messages: list[ChatMessage], tools: list[ChatTool]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=str(resolved.main.model),
        messages=messages,
        tools=tools,
        tool_choice=None,
        parallel_tool_calls=None,
        max_completion_tokens=int(resolved.main.max_completion_tokens),
        reasoning_effort=str(resolved.main.reasoning_effort),
        service_tier=resolved.main.get("service_tier"),
        stream_options=ChatStreamOptions(include_usage=True),
        **_sampling_kwargs(resolved.main),
    )


def _build_vision_request(
    resolved: CueBox,
    *,
    response_request: ResponseCreateRequest,
    history_messages: list[ChatMessage],
    ids: list[str],
    prompt: str,
    vision_model: str,
) -> ChatCompletionRequest:
    request = _vision_request(
        resolved,
        response_request,
        _vision_history_messages(history_messages),
        ids,
        prompt,
    )
    return replace(request, model=vision_model, stream_options=ChatStreamOptions(include_usage=True))


def _tool_args_or_none(call: ChatToolCall) -> tuple[list[str], str] | None:
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(arguments, dict):
        return None
    ids = _normalize_image_ids(arguments.get("ids"))
    prompt = arguments.get("prompt")
    if ids is None:
        return None
    if not isinstance(prompt, str) or not prompt:
        return None
    return ids, prompt


def _tool_args(call: ChatToolCall) -> tuple[list[str], str]:
    arguments = _tool_args_or_none(call)
    if arguments is None:
        raise RuntimeError(f"vision tool call {call.id} has invalid arguments: {call.arguments}")
    return arguments


def _vision_retry_validator(known_image_ids: set[str]) -> RetryValidator:
    async def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
        for call in result.message.tool_calls:
            if call.name != VISION_TOOL_NAME:
                continue
            arguments = _tool_args_or_none(call)
            if arguments is None:
                return retry_message(
                    problems=("The vision tool call must include image ids and a non-empty prompt.",),
                    rules=("Use the ids field to select one or more image ids and the prompt field to describe what to inspect.",),
                )
            ids, _prompt = arguments
            missing = [image_id for image_id in ids if image_id not in known_image_ids]
            if missing:
                joined = ", ".join(sorted(missing))
                return retry_message(
                    problems=(f"You requested unknown image ids: {joined}.",),
                    rules=("Use only image ids that already appear in the conversation.",),
                )
        return None

    return validate


async def _complete_with_retries(
    client: IChatCompletionClient,
    request: ChatCompletionRequest,
    *,
    validators: tuple[RetryValidator, ...],
):
    def next_request(history) -> ChatCompletionRequest:
        return replace(request, messages=[*request.messages, *history.messages])

    snapshot = await retry_complete(client, next_request=next_request, validators=validators)
    if not snapshot.results:
        raise RuntimeError("retry-complete ended without a final result")
    return snapshot


async def _stream_complete(client: IChatCompletionClient, request: ChatCompletionRequest) -> ChatCompletionResult:
    accumulator = Accumulator(tools=tuple(request.tools))
    result: ChatCompletionResult | None = None
    async for delta in client.stream(request):
        snapshot = accumulator.apply(delta)
        if snapshot.results:
            result = snapshot.results[-1]
    if result is None:
        raise RuntimeError("stream ended without final result")
    return result


def _load_config() -> CueBox:
    global _CONFIG_CACHE  # noqa: PLW0603
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    bus.reset()

    @bus.emit("config.collect")
    async def collect(paths: tuple[str, ...]) -> tuple[str, ...]:
        return paths

    discovered = _plugins()
    for name in _plugin_names():
        entry = discovered.get(name)
        if entry is not None:
            _import_plugin(entry)

    paths = anyio.run(partial(collect, paths=()))
    _CONFIG_CACHE = load(*paths)
    return _CONFIG_CACHE


def _build_local_openai_client(*, base_url: str, api_key: str, model: str) -> ChatCompletionClient:
    provider = OpenAIProvider(
        name="local-vision",
        api_key=api_key,
        base_url=base_url,
        quirks=(
            SystemRole(),
            MoveMessageField("reasoning_content", "reasoning", role="assistant"),
            MoveOutput("reasoning", "reasoning_content"),
        ),
        models={
            model: (
                ExtraBodyIf(
                    "reasoning_effort",
                    ("none",),
                    {
                        "reasoning": {"enabled": False},
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                ),
                ExtraBodyIf(
                    "reasoning_effort",
                    (None, "none"),
                    {"reasoning": {"enabled": True}},
                    negate=True,
                ),
                Drop("reasoning_effort"),
            )
        },
    )
    return ChatCompletionClient(provider)


def _build_routed_client(loaded: CueBox) -> RoutingChatCompletionClient:
    providers = build_providers(loaded)
    if not providers:
        raise SystemExit("no LLM providers are available; export the relevant provider API key environment variable first")
    routes = [ModelRoute(prefix=prefix, client=ChatCompletionClient(provider)) for prefix, provider in providers.items()]
    return RoutingChatCompletionClient(routes)


async def _evaluate_example(
    example: EvalExample,
    *,
    parser_seed: int,
    resolved: CueBox,
    response_request: ResponseCreateRequest,
    main_client: IChatCompletionClient,
    vision_client: IChatCompletionClient,
    vision_model: str,
    save_transcripts: bool,
) -> dict[str, Any]:
    def _tool_call_error(message: str) -> None:
        raise RuntimeError(message)

    started_at = time.perf_counter()
    base_messages = _main_messages_for_example(example, display_name=str(resolved.display_name))
    rewritten_request = _rewrite_request(_build_main_request(resolved, messages=base_messages, tools=[]))
    rewritten_messages = list(rewritten_request.messages)
    history_messages = list(base_messages)
    content = base_messages[-1].content
    if not isinstance(content, list):
        raise TypeError("MMMU-Pro benchmark expected user content to be multimodal")

    chat_images = [part for part in content if isinstance(part, ChatContentImage)]
    correct_ids = [_image_id(image) for image in chat_images]
    validators = (
        retry_on_tool_choice_mismatch,
        retry_on_unusable_tool_calls,
        _vision_retry_validator(set(correct_ids)),
    )
    transcript: list[dict[str, Any]] = []
    if save_transcripts:
        transcript = [_serialize_message(message) for message in base_messages]

    main_request = replace(rewritten_request, messages=rewritten_messages)
    parser_rng = random.Random(f"{parser_seed}:{example.id}")
    main_usage: dict[str, int] = {}
    vision_usage: dict[str, int] = {}
    main_calls = 0
    vision_calls = 0
    total_turns = 0
    raw_response_text: str | None = None

    try:
        while True:
            snapshot = await _complete_with_retries(main_client, main_request, validators=validators)
            main_result = snapshot.results[-1]
            main_usage = _merge_usage(main_usage, main_result.usage)
            main_calls += 1
            total_turns += 1

            new_turn_messages = list(snapshot.messages)
            rewritten_messages = [*rewritten_messages, *new_turn_messages]
            history_messages = [*history_messages, *new_turn_messages]

            if save_transcripts:
                transcript.extend(_serialize_message(message) for message in new_turn_messages)

            tool_outputs: list[ChatMessage] = []
            for call in main_result.message.tool_calls:
                if call.name != VISION_TOOL_NAME:
                    _tool_call_error(f"unexpected tool call name: {call.name}")
                requested_ids, prompt = _tool_args(call)
                unknown = [image_id for image_id in requested_ids if image_id not in correct_ids]
                if unknown:
                    _tool_call_error(f"model requested unknown ids: {unknown}, available: {correct_ids}")

                vision_request = _build_vision_request(
                    resolved,
                    response_request=response_request,
                    history_messages=history_messages,
                    ids=requested_ids,
                    prompt=prompt,
                    vision_model=vision_model,
                )
                vision_result = await _stream_complete(vision_client, vision_request)
                vision_usage = _merge_usage(vision_usage, vision_result.usage)
                vision_calls += 1
                tool_output_text = _tool_output_text(vision_result)
                tool_message = ChatMessage(
                    role="tool",
                    tool_call_id=call.id,
                    content=tool_output_text,
                    reasoning_content=vision_result.message.reasoning_content,
                )
                tool_outputs.append(tool_message)
                rewritten_messages.append(tool_message)
                history_messages.append(tool_message)

                if save_transcripts:
                    transcript.append(
                        {
                            "role": "vision_model",
                            "messages": [
                                {"role": "developer", "content": VISION_PROMPT},
                                *[_serialize_message(message) for message in _vision_history_messages(history_messages[:-1])],
                                {"role": "user", "content": _vision_turn_prompt(requested_ids, prompt)},
                                _serialize_message(vision_result.message),
                            ],
                        }
                    )
                    transcript.append(_serialize_message(tool_message))

            if not main_result.message.tool_calls:
                raw_response_text = _message_text(main_result.message)
                parsed_answer = _parse_multi_choice_response(raw_response_text, example.options, rng=parser_rng)
                correct = parsed_answer == example.answer if parsed_answer is not None else False
                return {
                    "correct": correct,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "error": None,
                    "example_id": example.id,
                    "expected_answer": example.answer,
                    "image_count": example.image_count,
                    "main_calls": main_calls,
                    "main_usage": main_usage,
                    "prediction": parsed_answer,
                    "raw_response": raw_response_text,
                    "score": 1.0 if correct else 0.0,
                    "subject": example.subject,
                    "topic_difficulty": example.topic_difficulty,
                    "transcript": transcript if save_transcripts else None,
                    "turns": total_turns,
                    "vision_calls": vision_calls,
                    "vision_usage": vision_usage,
                }

            main_request = _build_main_request(resolved, messages=rewritten_messages, tools=rewritten_request.tools)

    except Exception as exc:
        return {
            "correct": False,
            "elapsed_seconds": time.perf_counter() - started_at,
            "error": f"{type(exc).__name__}: {exc}",
            "example_id": example.id,
            "expected_answer": example.answer,
            "image_count": example.image_count,
            "main_calls": main_calls,
            "main_usage": main_usage,
            "prediction": None,
            "raw_response": raw_response_text,
            "score": 0.0,
            "subject": example.subject,
            "topic_difficulty": example.topic_difficulty,
            "transcript": transcript if save_transcripts else None,
            "turns": total_turns,
            "vision_calls": vision_calls,
            "vision_usage": vision_usage,
        }


async def _run_benchmark(
    examples: list[EvalExample],
    *,
    args: argparse.Namespace,
    resolved: CueBox,
    records_path: Path,
    vision_override: VisionOverride | None,
) -> None:
    response_request = ResponseCreateRequest(model=args.model)
    main_client = _build_routed_client(_load_config().plap.config)
    vision_client: IChatCompletionClient = main_client
    vision_model = str(resolved.vision.model)
    if vision_override is not None:
        vision_client = _build_local_openai_client(
            base_url=vision_override.base_url,
            api_key=vision_override.api_key,
            model=vision_override.model,
        )
        vision_model = vision_override.model

    try:
        correct = 0
        completed = 0
        errors = 0
        started_at = time.perf_counter()
        total_examples = len(examples)
        progress_lock = anyio.Lock()
        concurrency_limiter = anyio.Semaphore(args.concurrency)

        async def run_one(example_index: int, example: EvalExample) -> None:
            nonlocal completed, correct, errors

            async with concurrency_limiter:
                record = await _evaluate_example(
                    example,
                    parser_seed=args.seed,
                    resolved=resolved,
                    response_request=response_request,
                    main_client=main_client,
                    vision_client=vision_client,
                    vision_model=vision_model,
                    save_transcripts=args.save_transcripts,
                )

            async with progress_lock:
                _append_jsonl(records_path, record)
                completed += 1
                if bool(record.get("correct")):
                    correct += 1
                if isinstance(record.get("error"), str) and record["error"]:
                    errors += 1
                    logger.warning(
                        "mmmu_pro_bench.example_failed",
                        error=record["error"],
                        example_id=example.id,
                        index=example_index,
                        total=total_examples,
                    )
                if completed == total_examples or completed % args.progress_every == 0:
                    accuracy = correct / completed
                    logger.info(
                        "mmmu_pro_bench.progress",
                        accuracy=accuracy,
                        completed=completed,
                        elapsed_seconds=time.perf_counter() - started_at,
                        errors=errors,
                        total=total_examples,
                    )

        async with anyio.create_task_group() as task_group:
            for index, example in enumerate(examples, start=1):
                task_group.start_soon(run_one, index, example)
    finally:
        if vision_client is not main_client:
            await vision_client.aclose()
        await main_client.aclose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark plap's routed main+vision model loop on the full MMMU-Pro split. "
            "This exercises the main model, vision tool calls, and plap-style tool-call retries."
        )
    )
    parser.add_argument("--model", default="plap-ai/wisp", help="plap model alias to benchmark")
    parser.add_argument("--split", default="test", help="MMMU-Pro split to evaluate")
    parser.add_argument("--max-examples", default=None, type=int, help="Optional cap for smoke tests")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path, help="Directory for records, manifest, and summary")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing records.jsonl in --run-dir")
    parser.add_argument("--save-transcripts", action="store_true", help="Store full per-example transcripts in records.jsonl")
    parser.add_argument("--progress-every", default=25, type=int, help="Log progress every N completed examples")
    parser.add_argument("--concurrency", default=8, type=int, help="Maximum number of MMMU-Pro examples to evaluate concurrently")
    parser.add_argument("--seed", default=0, type=int, help="Random seed used for parser fallback determinism")
    parser.add_argument(
        "--vision-base-url",
        default=os.environ.get("VISION_BASE_URL"),
        help="Override the resolved vision model with a local OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--vision-wire-model",
        default=os.environ.get("VISION_WIRE_MODEL", "google/gemma-4-31b-it"),
        help="Actual model id exposed by --vision-base-url",
    )
    parser.add_argument(
        "--vision-api-key",
        default=os.environ.get("VISION_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy",
        help="API key for --vision-base-url",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")
    if args.progress_every <= 0:
        raise SystemExit("--progress-every must be positive")

    random.seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.run_dir / "records.jsonl"
    summary_path = args.run_dir / "summary.json"
    manifest_path = args.run_dir / "manifest.json"

    if records_path.exists() and not args.resume:
        raise SystemExit(f"{records_path} already exists; use --resume or choose a new --run-dir")

    logger.info("mmmu_pro_bench.loading_dataset", split=args.split, max_examples=args.max_examples)
    examples = _load_mmmu_pro_examples(args.split, args.max_examples)
    loaded = _load_config()
    resolved = loaded.plap.config.resolve({"model": args.model})

    vision_override = None
    resolved_vision_model = str(resolved.vision.model)
    if args.vision_base_url:
        vision_override = VisionOverride(
            base_url=args.vision_base_url,
            model=args.vision_wire_model,
            api_key=args.vision_api_key,
        )
        resolved_vision_model = vision_override.model

    manifest = _manifest_payload(
        args=args,
        total_examples=len(examples),
        resolved_main_model=str(resolved.main.model),
        resolved_vision_model=resolved_vision_model,
    )
    _validate_or_write_manifest(manifest_path, payload=manifest, resume=args.resume)

    pending_examples = examples
    if args.resume:
        completed = _completed_example_ids(records_path)
        pending_examples = [example for example in examples if example.id not in completed]
        logger.info(
            "mmmu_pro_bench.resume",
            completed=len(completed),
            pending=len(pending_examples),
            total=len(examples),
        )

    if pending_examples:
        anyio.run(
            partial(
                _run_benchmark,
                pending_examples,
                args=args,
                resolved=resolved,
                records_path=records_path,
                vision_override=vision_override,
            )
        )

    records = _read_jsonl_records(records_path)
    summary = _build_summary(
        args=args,
        records=records,
        total_examples=len(examples),
        resolved_main_model=str(resolved.main.model),
        resolved_vision_model=resolved_vision_model,
    )
    _write_json(summary_path, summary)
    _print_summary(summary)
    logger.info("mmmu_pro_bench.summary_written", path=str(summary_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
