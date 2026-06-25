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
import threading
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import gepa.optimize_anything as oa
import litellm
import structlog
from datasets import load_dataset
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig

import plap.llms.accumulator as accumulator_module
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
    ChatFunctionTool,
    ChatImageURL,
    ChatMessage,
    ChatTool,
    IChatCompletionClient,
    ReasoningEffort,
)
from plap.llms.completions.client import ChatCompletionClient
from plap.llms.completions.providers import build_providers
from plap.llms.completions.providers.openai import OpenAIProvider
from plap.llms.completions.quirks import Drop, ExtraBodyIf, MoveOutput, SystemRole
from plap.llms.completions.router import ModelRoute, RoutingChatCompletionClient
from plap.plugins.vision import (
    VISION_PROMPT,
    VISION_TOOL,
    VISION_TOOL_NAME,
    _image_id,
    _rewrite_request,
    _tool_output_text,
    _vision_content,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_DIRECT_INSTRUCTION = "Answer with the option letter from the given choices directly."
logger = structlog.stdlib.get_logger(__name__)

# Match mimo_gepa: suppress noisy payload-level tool call repair logs.
accumulator_module._log_tool_call_repair = lambda *args, **kwargs: None


def _make_reflection_lm_with_reasoning(model_name: str, *, reasoning_effort: str):
    def _lm(prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        completion = litellm.completion(
            model=model_name,
            messages=messages,
            reasoning_effort=reasoning_effort,
        )
        return completion.choices[0].message.content  # type: ignore[union-attr]

    return _lm


# ---------------------------------------------------------------------------
# Tunable seed candidate (3 parameters)
# ---------------------------------------------------------------------------

VISION_TOOL_DESCRIPTION = VISION_TOOL.function.description or ""
VISION_TOOL_PROMPT_DESCRIPTION = str(VISION_TOOL.function.parameters["properties"]["prompt"]["description"])

# SEED: dict[str, str] = {
#     "vision_prompt": VISION_PROMPT,
#     "tool_description": VISION_TOOL_DESCRIPTION,
#     "prompt_description": VISION_TOOL_PROMPT_DESCRIPTION,
# }

SEED: dict[str, str] = {
    "vision_prompt": VISION_PROMPT,
    "tool_description": VISION_TOOL_DESCRIPTION,
    "prompt_description": VISION_TOOL_PROMPT_DESCRIPTION,
}

# ---------------------------------------------------------------------------
# Dataset (MMMU Pro via HuggingFace)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MMMUExample:
    id: str
    subject: str
    question: str
    options: list[str]
    answer: str
    images: list[bytes] = field(repr=False)

    @property
    def image_count(self) -> int:
        return len(self.images)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"MMMUExample(id={self.id!r})"


@dataclass(frozen=True, slots=True)
class VisionOverride:
    base_url: str
    model: str
    api_key: str


def _options_formatted(options: list[str]) -> str:
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    return "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(options))


def _load_mmmu_pro_hard(
    split: str = "test",
    max_examples: int | None = None,
) -> list[MMMUExample]:
    cache_key = hashlib.sha256(f"standard (10 options):hard:v2:{split}:{max_examples}".encode()).hexdigest()[:16]
    cache_dir = REPO_ROOT / ".dev" / "vision-gepa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"dataset_cache_{cache_key}.pkl"

    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split=split)
    rows: list[MMMUExample] = []
    for row in ds:
        if row.get("topic_difficulty") != "Hard":
            continue
        images: list[bytes] = []
        for key in ("image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"):
            img = row.get(key)
            if img is None:
                continue
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(buf.getvalue())
        if not images:
            continue
        rows.append(
            MMMUExample(
                id=str(row["id"]),
                subject=str(row["subject"]),
                question=str(row["question"]),
                options=ast.literal_eval(str(row["options"])),
                answer=str(row["answer"]).strip().upper(),
                images=images,
            )
        )
    if max_examples is not None:
        rows = rows[:max_examples]

    with cache_path.open("wb") as f:
        pickle.dump(rows, f)
    return rows


def _pil_to_data_uri(data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _grouped_subject_split(
    examples: list[MMMUExample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[MMMUExample], list[MMMUExample]]:
    if not examples or val_fraction <= 0:
        return list(examples), []

    by_subject: dict[str, list[MMMUExample]] = {}
    for example in examples:
        by_subject.setdefault(example.subject, []).append(example)

    subjects = list(by_subject)
    rng = random.Random(seed)
    rng.shuffle(subjects)

    target_val_examples = max(1, round(len(examples) * val_fraction))
    val_subjects: set[str] = set()
    val_subject_order: list[str] = []
    val_examples_count = 0
    for subject in subjects:
        if val_examples_count >= target_val_examples and val_subjects:
            break
        val_subjects.add(subject)
        val_subject_order.append(subject)
        val_examples_count += len(by_subject[subject])

    while len(val_subjects) > 1:
        train = [example for example in examples if example.subject not in val_subjects]
        if train:
            break
        last_subject = val_subject_order.pop()
        val_subjects.remove(last_subject)

    train = [example for example in examples if example.subject not in val_subjects]
    val = [example for example in examples if example.subject in val_subjects]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Vision tool builder (parameterized by candidate)
# ---------------------------------------------------------------------------


def _build_tool(tool_desc: str, prompt_desc: str) -> ChatTool:
    return ChatTool(
        function=ChatFunctionTool(
            name=VISION_TOOL_NAME,
            description=tool_desc,
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": prompt_desc,
                    },
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
            strict=True,
        )
    )


def _referenced_image_ids(text: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for image_id in re.findall(r"\bimage-[A-Z2-7]{4}-[A-Z2-7]{4}-[A-Z2-7]{4}-[A-Z2-7]{4}\b", text):
        if image_id in seen:
            continue
        seen.add(image_id)
        ids.append(image_id)
    return ids


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _choice_letters(options: list[str]) -> list[str]:
    return [chr(ord("A") + i) for i in range(len(options))]


def _index_to_answer(options: list[str]) -> dict[str, str]:
    return dict(zip(_choice_letters(options), options, strict=True))


def _parse_multi_choice_response(response: str | None, options: list[str]) -> str | None:
    if response is None:
        return None

    index2ans = _index_to_answer(options)
    all_choices = list(index2ans.keys())

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

    index_ans = True
    ans_with_brack = False
    candidates: list[str] = []
    for choice in all_choices:
        if f"({choice})" in normalized:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        candidates.extend(choice for choice in all_choices if f"{choice} " in normalized)

    if not candidates:
        candidates.extend(choice for choice in all_choices if f"{choice}." in normalized)

    if not candidates and len(normalized.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in normalized.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        return random.choice(all_choices)

    if len(candidates) == 1:
        return candidates[0]

    start_indexes: list[int] = []
    if index_ans:
        if ans_with_brack:
            start_indexes.extend(normalized.rfind(f"({candidate})") for candidate in candidates)
        else:
            start_indexes.extend(normalized.rfind(f" {candidate} ") for candidate in candidates)
    else:
        lowered = normalized.lower()
        start_indexes.extend(lowered.rfind(index2ans[candidate].lower()) for candidate in candidates)
    best_index = max(range(len(start_indexes)), key=start_indexes.__getitem__)
    return candidates[best_index]


# ---------------------------------------------------------------------------
# Transcript serialization
# ---------------------------------------------------------------------------


def _serialize(content: str | list[ChatContentText | ChatContentImage] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, ChatContentText):
            parts.append(part.text)
        else:
            parts.append(f"<image url={part.image_url.url[:120]}>")
    return "\n".join(parts)


def _append_jsonl(path: Path, *, lock: threading.Lock, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _candidate_prompt_lengths(candidate: dict[str, str]) -> dict[str, int]:
    return {name: len(value) for name, value in candidate.items()}


def _candidate_prompt_length_score(candidate: dict[str, str]) -> tuple[int, float]:
    total_length = sum(_candidate_prompt_lengths(candidate).values())
    return total_length, 1.0 / max(total_length, 1)


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


# ---------------------------------------------------------------------------
# Interleaved content builder (standard config: question text + images)
# ---------------------------------------------------------------------------


def _build_user_content(question: str, options: list[str], images: list[bytes]) -> list[ChatContentPart]:
    """Build content matching MMMU-Pro format: text with <image> markers + images appended at end."""
    label = _options_formatted(options)
    full_text = f"{question}\n{label}\n{_DIRECT_INSTRUCTION}"
    ordered = [int(num) for num in re.findall(r"<image\s+(\d+)>", full_text)]
    text = re.sub(r"<image\s+\d+>", "<image>", full_text)
    parts: list[ChatContentPart] = [ChatContentText(text=text)]
    for idx in ordered:
        data_uri = _pil_to_data_uri(images[idx - 1])
        parts.append(ChatContentImage(image_url=ChatImageURL(url=data_uri)))
    return parts


# ---------------------------------------------------------------------------
# Per-example evaluation (full main → vision-tool → vision pipeline)
# ---------------------------------------------------------------------------


async def _evaluate_one(
    candidate: dict[str, str],
    example: MMMUExample,
    *,
    main_client: IChatCompletionClient,
    vision_client: IChatCompletionClient,
    main_model: str,
    vision_model: str,
    vision_reasoning_effort: ReasoningEffort | None,
) -> tuple[float, dict[str, Any]]:
    candidate_lengths = _candidate_prompt_lengths(candidate)
    total_prompt_length, prompt_length_score = _candidate_prompt_length_score(candidate)
    anti_overfit_instruction = (
        "Do not optimize by hardcoding MMMU-Pro specific wording, answer-letter conventions, subject-specific heuristics, "
        "or specific examples. Prefer generic image-inspection, transcription, and iterative clarification behavior "
        "that transfers across domains."
    )
    content = _build_user_content(example.question, example.options, example.images)
    raw_main = ChatCompletionRequest(
        model=main_model,
        messages=[
            ChatMessage(role="developer", content="You are Wisp, an AI assistant."),
            ChatMessage(role="user", content=content),
        ],
        tools=[],
        tool_choice="auto",
        max_completion_tokens=4096,
        reasoning_effort=ReasoningEffort.HIGH,
        temperature=0,
    )

    # Pre-compute image IDs for validation
    chat_images: list[ChatContentImage] = [part for part in content if isinstance(part, ChatContentImage)]
    correct_ids = [_image_id(img) for img in chat_images]

    # 2. Apply the vision plugin's rewrite: image → text id, inject VISION_TOOL
    rewritten = _rewrite_request(raw_main)
    # Replace with the tuned tool description from this candidate
    tuned_tool = _build_tool(candidate["tool_description"], candidate["prompt_description"])
    main_request = replace(rewritten, tools=[tuned_tool])

    messages = list(main_request.messages)
    vision_messages = [ChatMessage(role="user", content=_vision_content(correct_ids, chat_images))]
    raw_answer: str | None = None
    total_turns = 1
    error: str | None = None
    transcript: list[dict[str, Any]] = [{"role": m.role, "content": _serialize(m.content)} for m in messages]

    try:
        main_result = await _stream_complete(main_client, main_request)
    except Exception as exc:
        return 0.0, {
            "error": f"main model call failed: {exc}",
            "transcript": transcript,
            "anti_overfit_instruction": anti_overfit_instruction,
            "candidate_component_lengths": candidate_lengths,
            "total_prompt_length": total_prompt_length,
            "scores": {
                "efficiency": 1.0,
                "prompt_compactness": prompt_length_score,
            },
        }

    while True:
        tool_calls = main_result.message.tool_calls
        tool_outputs: list[ChatMessage] = []

        transcript.append(
            {
                "role": "assistant",
                "content": _serialize(main_result.message.content),
                "reasoning_content": main_result.message.reasoning_content,
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
            }
        )

        for call in tool_calls:
            if call.name == VISION_TOOL_NAME:
                try:
                    args = json.loads(call.arguments)
                except json.JSONDecodeError as err:
                    raise RuntimeError(f"vision tool call {call.id} has invalid JSON arguments: {call.arguments}") from err
                prompt = args["prompt"]

                requested_ids = _referenced_image_ids(prompt)
                if not requested_ids:
                    error = "vision tool prompt must reference one or more known image ids"
                    break
                unknown = [iid for iid in requested_ids if iid not in correct_ids]
                if unknown:
                    error = f"model requested unknown ids: {unknown}, available: {correct_ids}"
                    break

                vision_request = ChatCompletionRequest(
                    model=vision_model,
                    messages=[
                        ChatMessage(role="developer", content=candidate["vision_prompt"]),
                        *vision_messages,
                        ChatMessage(role="user", content=prompt),
                    ],
                    max_completion_tokens=8192,
                    reasoning_effort=vision_reasoning_effort,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=20,
                    repetition_penalty=1.15,
                    presence_penalty=1.5,
                )
                vision_result = None
                try:
                    vision_result = await _stream_complete(vision_client, vision_request)
                    vision_output_text = _tool_output_text(vision_result)
                except Exception as exc:
                    assistant_message = None
                    if vision_result is not None:
                        assistant_message = {
                            "role": "assistant",
                            "content": _serialize(vision_result.message.content),
                            "reasoning_content": vision_result.message.reasoning_content,
                            "refusal": vision_result.message.refusal,
                            "tool_calls": [
                                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in vision_result.message.tool_calls
                            ],
                        }
                    transcript.append(
                        {
                            "role": "vision_model",
                            "messages": [
                                {"role": "developer", "content": candidate["vision_prompt"]},
                                *[{"role": message.role, "content": _serialize(message.content)} for message in vision_messages],
                                {"role": "user", "content": prompt},
                                *([] if assistant_message is None else [assistant_message]),
                            ],
                            "error": f"vision model failed: {exc}",
                        }
                    )
                    error = f"vision model failed: {exc}"
                    break
                tool_output = ChatMessage(
                    role="tool",
                    tool_call_id=call.id,
                    content=vision_output_text,
                )
                tool_outputs.append(tool_output)
                transcript.append({"role": "tool", "tool_call_id": call.id, "content": vision_output_text})
                prior_vision_messages = list(vision_messages)
                vision_messages.append(ChatMessage(role="user", content=prompt))
                vision_messages.append(ChatMessage(role="assistant", content=vision_output_text))
                transcript.append(
                    {
                        "role": "vision_model",
                        "messages": [
                            {"role": "developer", "content": candidate["vision_prompt"]},
                            *[{"role": message.role, "content": _serialize(message.content)} for message in prior_vision_messages],
                            {"role": "user", "content": prompt},
                            {
                                "role": "assistant",
                                "content": vision_output_text,
                                "reasoning_content": vision_result.message.reasoning_content,
                            },
                        ],
                    }
                )
            else:
                tool_outputs.append(ChatMessage(role="tool", tool_call_id=call.id, content=""))
                transcript.append({"role": "tool", "tool_call_id": call.id, "content": ""})

        if error is not None:
            break

        if not tool_calls:
            answer_text = main_result.message.content
            if isinstance(answer_text, str):
                raw_answer = _parse_multi_choice_response(answer_text, example.options)
            elif isinstance(answer_text, list):
                raw_answer = _parse_multi_choice_response(
                    "".join(part.text for part in answer_text if isinstance(part, ChatContentText)),
                    example.options,
                )
            correct = raw_answer == example.answer if raw_answer is not None else False
            score = 1.0 if correct else 0.0
            side_info: dict[str, Any] = {
                "expected": example.answer,
                "got": raw_answer,
                "turns": total_turns,
                "transcript": transcript,
                "anti_overfit_instruction": anti_overfit_instruction,
                "candidate_component_lengths": candidate_lengths,
                "total_prompt_length": total_prompt_length,
                "scores": {
                    "efficiency": 1.0 / total_turns,
                    "prompt_compactness": prompt_length_score,
                },
            }
            return score, side_info

        messages.append(main_result.message)
        messages.extend(tool_outputs)
        total_turns += 1

        try:
            main_request = ChatCompletionRequest(
                model=main_model,
                messages=messages,
                tools=main_request.tools,
                tool_choice="auto",
                max_completion_tokens=8192,
                reasoning_effort=ReasoningEffort.HIGH,
                temperature=0,
            )
            main_result = await _stream_complete(main_client, main_request)
        except Exception as exc:
            return 0.0, {
                "error": f"main model continuation call failed: {exc}",
                "transcript": transcript,
                "anti_overfit_instruction": anti_overfit_instruction,
                "candidate_component_lengths": candidate_lengths,
                "total_prompt_length": total_prompt_length,
                "scores": {
                    "efficiency": 1.0 / max(total_turns, 1),
                    "prompt_compactness": prompt_length_score,
                },
            }

    side_info = {
        "expected": example.answer,
        "got": None,
        "turns": total_turns,
        "error": error,
        "transcript": transcript,
        "anti_overfit_instruction": anti_overfit_instruction,
        "candidate_component_lengths": candidate_lengths,
        "total_prompt_length": total_prompt_length,
        "scores": {
            "efficiency": 1.0 / max(total_turns, 1),
            "prompt_compactness": prompt_length_score,
        },
    }
    return 0.0, side_info


# ---------------------------------------------------------------------------
# Evaluator builder
# ---------------------------------------------------------------------------


def _build_evaluator(
    *,
    main_model_str: str,
    vision_model_str: str,
    transcript_path: Path,
    vision_override: VisionOverride | None,
):
    transcript_lock = threading.Lock()

    async def _run(candidate: dict[str, str], example: MMMUExample) -> tuple[float, dict[str, Any]]:
        _main_model, _vision_model, client = _build_client_in_loop(main_model_str)
        vision_client: IChatCompletionClient = client
        vision_model = _vision_model
        if vision_override is not None:
            vision_client = _build_local_openai_client(
                base_url=vision_override.base_url,
                api_key=vision_override.api_key,
                model=vision_override.model,
            )
            vision_model = vision_override.model
        try:
            return await _evaluate_one(
                candidate,
                example,
                main_client=client,
                vision_client=vision_client,
                main_model=_main_model,
                vision_model=vision_model,
                vision_reasoning_effort=ReasoningEffort.MEDIUM,
            )
        finally:
            if vision_client is not client:
                await vision_client.aclose()
            await client.aclose()

    def evaluate(candidate: dict[str, str], example: MMMUExample, **kwargs) -> tuple[float, dict[str, Any]]:
        _ = kwargs
        c_hash = hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest()[:8]
        logger.info("eval.start", candidate=c_hash, example=example.id, thread=threading.current_thread().name)
        result = anyio.run(_run, candidate, example)
        score, info = result
        _append_jsonl(
            transcript_path,
            lock=transcript_lock,
            record={
                "candidate": candidate,
                "candidate_hash": c_hash,
                "error": info.get("error"),
                "example_id": example.id,
                "expected": info.get("expected"),
                "got": info.get("got"),
                "score": score,
                "scores": info.get("scores"),
                "thread": threading.current_thread().name,
                "transcript": info.get("transcript"),
                "turns": info.get("turns"),
            },
        )
        logger.info(
            "eval.done",
            candidate=c_hash,
            example=example.id,
            score=score,
            turns=info.get("turns"),
            error=info.get("error"),
        )
        return result

    return evaluate


# ---------------------------------------------------------------------------
# Config / provider setup
# ---------------------------------------------------------------------------


_CONFIG_CACHE: CueBox | None = None


def _load_config():
    global _CONFIG_CACHE  # noqa: PLW0603
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    bus.reset()

    @bus.emit("config.collect")
    async def _collect(paths: tuple[str, ...]) -> tuple[str, ...]:
        return paths

    for name in _plugin_names():
        entry = _plugins().get(name)
        if entry is not None:
            _import_plugin(entry)

    paths = anyio.run(partial(_collect, paths=()))
    _CONFIG_CACHE = load(*paths)
    return _CONFIG_CACHE


def _resolve_model_name(model_str: str) -> str:
    config = _load_config()
    loaded = config.plap.config
    providers = build_providers(loaded)
    if not providers:
        raise SystemExit(f"no LLM providers available for {model_str!r}. Check the relevant env var is set and exported.")
    resolved = loaded.resolve({"model": model_str})
    logger.info("resolve_model", model=model_str, resolved=resolved.main.model)
    return resolved.main.model


def _build_local_openai_client(*, base_url: str, api_key: str, model: str) -> ChatCompletionClient:
    provider = OpenAIProvider(
        name="local-vision",
        api_key=api_key,
        base_url=base_url,
        quirks=(
            SystemRole(),
            MoveOutput("reasoning", "reasoning_content"),
        ),
        models={
            model: (
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


def _build_client_in_loop(model_str: str) -> tuple[str, str, ChatCompletionClient]:
    config = _load_config()
    loaded = config.plap.config
    resolved = loaded.resolve({"model": model_str})
    providers = build_providers(loaded)
    routes = [ModelRoute(prefix=prefix, client=ChatCompletionClient(provider)) for prefix, provider in providers.items()]
    client: IChatCompletionClient = RoutingChatCompletionClient(routes)
    return resolved.main.model, resolved.vision.model, client


# ---------------------------------------------------------------------------
# GEPA runner
# ---------------------------------------------------------------------------


def _run_gepa(
    *,
    train: list[MMMUExample],
    val: list[MMMUExample],
    evaluator: Any,
    run_dir: Path,
    reflection_model: str,
    max_metric_calls: int,
    seed: int,
) -> oa.GEPAResult:
    cfg = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir.resolve()),
            seed=seed,
            max_metric_calls=max_metric_calls,
            candidate_selection_strategy="pareto",
            parallel=True,
            max_workers=50,
            cache_evaluation=True,
            use_cloudpickle=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=_make_reflection_lm_with_reasoning(
                reflection_model,
                reasoning_effort=ReasoningEffort.XHIGH.value,
            ),
            skip_perfect_score=True,
            perfect_score=1.0,
            reflection_minibatch_size=5,
        ),
        merge=None,
    )
    return oa.optimize_anything(
        seed_candidate=SEED,
        evaluator=evaluator,
        dataset=train,
        valset=val,
        config=cfg,
        objective=(
            "Optimize 3 parameters (vision_prompt, tool_description, prompt_description) for a generic image-inspection "
            "tool used inside a multimodal question-answering system. Improve image-grounded usefulness, iterative clarification, "
            "and transfer across domains while avoiding benchmark-specific tricks or memorized phrasing."
        ),
        background=(
            "Parameters:\n"
            "- vision_prompt: developer prompt for the vision model; keep it general and image-grounded\n"
            "- tool_description: description of the vision tool shown to the main model\n"
            "- prompt_description: description of the prompt parameter the main model sends to the vision tool\n\n"
            "The benchmark adapter uses MMMU-Pro standard-format academic/technical multiple-choice questions, "
            "but the optimized fields should not hardcode multiple-choice wording, answer-letter conventions, "
            "subject-specific heuristics, or specific examples. "
            "Optimize generic image reading, transcription, interpretation, and iterative follow-up behavior. "
            "side_info includes full transcripts, turns, anti-overfit guidance, and a prompt-compactness objective."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize vision prompt/tool descriptions on MMMU Pro.")
    parser.add_argument("--main-model", default="plap-ai/wisp", help="Main model name from config")
    parser.add_argument("--vision-model", default="plap-ai/wisp", help="Vision model name from config")
    parser.add_argument(
        "--vision-base-url",
        default=os.environ.get("VISION_BASE_URL"),
        help="Override vision model with a local OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--vision-wire-model",
        default=os.environ.get("VISION_WIRE_MODEL", "google/gemma-4-31b-it"),
        help="Actual model id exposed by the overridden vision endpoint",
    )
    parser.add_argument(
        "--vision-api-key",
        default=os.environ.get("VISION_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy",
        help="API key for overridden vision endpoint",
    )
    parser.add_argument("--reflection-model", default="openai/gpt-5.5", help="Reflection LM for GEPA")
    parser.add_argument("--run-dir", default=REPO_ROOT / ".dev" / "vision-gepa", type=Path)
    parser.add_argument("--max-examples", default=None, type=int, help="Limit dataset size for testing")
    parser.add_argument("--max-metric-calls", default=200, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--val-split", default=0.1, type=float)
    parser.add_argument("--no-val", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logger.info("loading dataset")
    examples = _load_mmmu_pro_hard(max_examples=args.max_examples)
    if args.no_val:
        train = list(examples)
        val = []
    else:
        train, val = _grouped_subject_split(
            examples,
            val_fraction=args.val_split,
            seed=args.seed,
        )
    logger.info("dataset", total=len(examples), train=len(train), val=len(val))
    if val:
        logger.info(
            "dataset.grouped_split",
            train_subjects=sorted({example.subject for example in train}),
            val_subjects=sorted({example.subject for example in val}),
        )

    main_str = _resolve_model_name(args.main_model)
    vision_str = _resolve_model_name(args.vision_model)
    logger.info("models", task=main_str, vision=vision_str, reflection=args.reflection_model)

    vision_override = None
    if args.vision_base_url:
        vision_override = VisionOverride(
            base_url=args.vision_base_url,
            model=args.vision_wire_model,
            api_key=args.vision_api_key,
        )
        logger.info(
            "vision.override",
            base_url=vision_override.base_url,
            model=vision_override.model,
        )

    evaluator = _build_evaluator(
        main_model_str=args.main_model,
        vision_model_str=args.vision_model,
        transcript_path=args.run_dir / "evaluation_transcripts.jsonl",
        vision_override=vision_override,
    )

    _run_gepa(
        train=train,
        val=val,
        evaluator=evaluator,
        run_dir=args.run_dir,
        reflection_model=args.reflection_model,
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
