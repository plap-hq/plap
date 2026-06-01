#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = REPO_ROOT / "tests" / ".env"
DEFAULT_REPORT_FILE = REPO_ROOT / ".dev" / "openai_responses_item_ordering_probe.json"
MODEL = "gpt-5.4-mini"
INCLUDE = ["reasoning.encrypted_content"]
TOOL_NAME = "record_step"
MAX_CASE_ATTEMPTS = 2
TOOL_SCHEMA = {
    "type": "function",
    "name": TOOL_NAME,
    "description": "Record a probe step number and return a short acknowledgement.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "step": {"type": "integer"},
        },
        "required": ["step"],
        "additionalProperties": False,
    },
}

type JSONValue = object
type JSONObject = dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class Baseline:
    name: str
    input_items: list[JSONObject]
    input_sequence: list[str]
    replay_start_index: int
    response_id: str | None
    output_items: list[JSONObject]
    output_digest: JSONObject


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    input_items: list[JSONObject]
    input_sequence: list[str]
    replay_start_index: int
    request_kwargs: JSONObject
    baseline_response_id: str | None
    baseline_output_digest: JSONObject
    notes: str


@dataclass(frozen=True, slots=True)
class InsertionVariant:
    name: str
    items: list[JSONObject]


def main() -> int:
    args = _parse_args()
    _load_env_file(args.env_file.resolve())
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set; put it in tests/.env or your shell before running this probe.")
    report = asyncio.run(
        _run_probe(
            api_key=api_key,
            model=args.model,
            max_case_attempts=args.max_case_attempts,
            max_tool_attempts=args.max_tool_attempts,
        )
    )
    report_path = args.report_file.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report, report_path=report_path)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenAI Responses manual-item ordering constraints.")
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-case-attempts", default=MAX_CASE_ATTEMPTS, type=int)
    parser.add_argument("--max-tool-attempts", default=3, type=int)
    parser.add_argument("--report-file", default=DEFAULT_REPORT_FILE, type=Path)
    return parser.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        os.environ.setdefault(key.strip(), _unquote(value.strip()))


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


async def _run_probe(*, api_key: str, model: str, max_case_attempts: int, max_tool_attempts: int) -> JSONObject:
    client = AsyncOpenAI(api_key=api_key, max_retries=0, timeout=300)
    try:
        answer_baseline, answer_scenario = await _build_answer_pair_scenario(client, model=model)
        tool_baseline, tool_scenarios = await _build_tool_scenarios(
            client,
            model=model,
            max_tool_attempts=max_tool_attempts,
        )
        scenarios = [answer_scenario, *tool_scenarios]
        scenario_reports: list[JSONObject] = []
        for scenario in scenarios:
            scenario_reports.append(
                await _probe_scenario(
                    client,
                    model=model,
                    max_case_attempts=max_case_attempts,
                    scenario=scenario,
                )
            )
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "max_case_attempts": max_case_attempts,
            "max_tool_attempts": max_tool_attempts,
            "model": model,
            "answer_pair_baseline": asdict(answer_baseline),
            "tool_first_turn_baseline": asdict(tool_baseline),
            "scenarios": scenario_reports,
        }
    finally:
        await client.close()


async def _build_answer_pair_scenario(client: AsyncOpenAI, *, model: str) -> tuple[Baseline, Scenario]:
    initial_user = _message(role="user", content="Reply with exactly BASELINE_OK and nothing else.")
    response = await _create_response(
        client,
        model=model,
        include=INCLUDE,
        input=[initial_user],
        max_output_tokens=256,
        reasoning={"effort": "high"},
        store=False,
    )
    output_items = _output_items(response)
    output_digest = _response_digest(output_items)
    if output_digest["output_sequence"] != ["R0", "M0"]:
        raise RuntimeError(f"answer baseline did not return the expected reasoning/message pair: {output_digest['output_sequence']}")
    baseline = Baseline(
        name="answer_pair_baseline",
        input_items=[initial_user],
        input_sequence=_sequence_labels([initial_user]),
        replay_start_index=1,
        response_id=getattr(response, "id", None),
        output_items=output_items,
        output_digest=output_digest,
    )
    followup_user = _message(role="user", content="Reply with exactly FOLLOWUP_OK and nothing else.")
    scenario_input = [deepcopy(initial_user), *_clone_items(output_items), followup_user]
    followup = await _create_response(
        client,
        model=model,
        include=INCLUDE,
        input=_clone_items(scenario_input),
        max_output_tokens=256,
        reasoning={"effort": "high"},
        store=False,
    )
    return baseline, Scenario(
        name="answer_pair_followup",
        input_items=scenario_input,
        input_sequence=_sequence_labels(scenario_input),
        replay_start_index=1,
        request_kwargs={},
        baseline_response_id=getattr(followup, "id", None),
        baseline_output_digest=_response_digest(_output_items(followup)),
        notes="Manual replay of a simple R M answer pair followed by a fresh user turn.",
    )


async def _build_tool_scenarios(
    client: AsyncOpenAI,
    *,
    model: str,
    max_tool_attempts: int,
) -> tuple[Baseline, list[Scenario]]:
    prompt = (
        "Use the record_step function exactly once now. After you receive the function output, continue and call "
        "record_step exactly once more. Do not provide a final answer before the second function call. After you "
        "receive the second function output, reply with exactly TOOL_CHAIN_DONE and nothing else."
    )
    initial_user = _message(role="user", content=prompt)
    request_kwargs: JSONObject = {
        "max_output_tokens": 512,
        "parallel_tool_calls": False,
        "reasoning": {"effort": "high"},
        "store": False,
        "tool_choice": "required",
        "tools": [deepcopy(TOOL_SCHEMA)],
    }
    last_error: str | None = None
    for attempt in range(1, max_tool_attempts + 1):
        try:
            first = await _create_response(
                client,
                model=model,
                include=INCLUDE,
                input=[deepcopy(initial_user)],
                **request_kwargs,
            )
        except APIStatusError as exc:
            last_error = f"attempt {attempt}: first turn failed with {exc.status_code}"
            continue
        first_output = _output_items(first)
        first_digest = _response_digest(first_output)
        first_call = _first_function_call(first_output)
        if first_call is None:
            last_error = f"attempt {attempt}: first turn had no function_call ({first_digest['output_sequence']})"
            continue
        if not _has_reasoning(first_output):
            last_error = f"attempt {attempt}: first turn had no reasoning item ({first_digest['output_sequence']})"
            continue
        first_output_item = _function_call_output(first_call, suffix="real_1")
        continuation_input = [deepcopy(initial_user), *_clone_items(first_output), first_output_item]
        try:
            second = await _create_response(
                client,
                model=model,
                include=INCLUDE,
                input=_clone_items(continuation_input),
                **request_kwargs,
            )
        except APIStatusError as exc:
            last_error = f"attempt {attempt}: second turn failed with {exc.status_code}"
            continue
        second_output = _output_items(second)
        second_digest = _response_digest(second_output)
        second_call = _first_function_call(second_output)
        if second_call is None:
            last_error = f"attempt {attempt}: continuation had no function_call ({second_digest['output_sequence']})"
            continue
        if not _has_reasoning(second_output):
            last_error = f"attempt {attempt}: continuation had no reasoning item ({second_digest['output_sequence']})"
            continue
        second_output_item = _function_call_output(second_call, suffix="real_2")
        third_input = [*continuation_input, *_clone_items(second_output), second_output_item]
        try:
            third = await _create_response(
                client,
                model=model,
                include=INCLUDE,
                input=_clone_items(third_input),
                **request_kwargs,
            )
        except APIStatusError as exc:
            last_error = f"attempt {attempt}: third turn failed with {exc.status_code}"
            continue
        third_output = _output_items(third)
        third_digest = _response_digest(third_output)
        baseline = Baseline(
            name="tool_first_turn_baseline",
            input_items=[initial_user],
            input_sequence=_sequence_labels([initial_user]),
            replay_start_index=1,
            response_id=getattr(first, "id", None),
            output_items=first_output,
            output_digest=first_digest,
        )
        second_chain_start = len(continuation_input)
        return baseline, [
            Scenario(
                name="tool_first_continuation",
                input_items=continuation_input,
                input_sequence=_sequence_labels(continuation_input),
                replay_start_index=1,
                request_kwargs=request_kwargs,
                baseline_response_id=getattr(second, "id", None),
                baseline_output_digest=second_digest,
                notes="Manual replay of the first tool turn plus the real first function_call_output before the second reasoning step.",
            ),
            Scenario(
                name="tool_second_continuation",
                input_items=third_input,
                input_sequence=_sequence_labels(third_input),
                replay_start_index=second_chain_start,
                request_kwargs=request_kwargs,
                baseline_response_id=getattr(third, "id", None),
                baseline_output_digest=third_digest,
                notes="Manual replay of two full tool subchains before the third response, focused on the second chain boundary and onward.",
            ),
        ]
    raise RuntimeError(f"could not build a usable two-step tool chain after {max_tool_attempts} attempts: {last_error}")


async def _probe_scenario(
    client: AsyncOpenAI,
    *,
    model: str,
    max_case_attempts: int,
    scenario: Scenario,
) -> JSONObject:
    mutations: list[JSONObject] = []
    insertion_variants = _insertion_variants(scenario)
    for insert_index, gap_label in _gap_labels(scenario.input_sequence, replay_start_index=scenario.replay_start_index):
        for variant in insertion_variants:
            mutated = _insert_items(scenario.input_items, insert_index=insert_index, inserted=variant.items)
            result = await _execute_probe_case(
                client,
                model=model,
                max_case_attempts=max_case_attempts,
                input_items=mutated,
                request_kwargs=scenario.request_kwargs,
                baseline_output_digest=scenario.baseline_output_digest,
            )
            mutations.append(
                {
                    "category": "insert",
                    "gap": gap_label,
                    "insert_index": insert_index,
                    "input_sequence": _sequence_labels(mutated),
                    "inserted_sequence": _sequence_labels(variant.items),
                    "mutation": variant.name,
                    "result": result,
                }
            )
    for removal in _omission_cases(scenario):
        input_items = removal.pop("input_items")
        result = await _execute_probe_case(
            client,
            model=model,
            max_case_attempts=max_case_attempts,
            input_items=input_items,
            request_kwargs=scenario.request_kwargs,
            baseline_output_digest=scenario.baseline_output_digest,
        )
        mutations.append({**removal, "result": result})
    for swap in _swap_cases(scenario):
        input_items = swap.pop("input_items")
        result = await _execute_probe_case(
            client,
            model=model,
            max_case_attempts=max_case_attempts,
            input_items=input_items,
            request_kwargs=scenario.request_kwargs,
            baseline_output_digest=scenario.baseline_output_digest,
        )
        mutations.append({**swap, "result": result})
    for move in _move_cases(scenario):
        input_items = move.pop("input_items")
        result = await _execute_probe_case(
            client,
            model=model,
            max_case_attempts=max_case_attempts,
            input_items=input_items,
            request_kwargs=scenario.request_kwargs,
            baseline_output_digest=scenario.baseline_output_digest,
        )
        mutations.append({**move, "result": result})
    for edit in _edit_cases(scenario):
        input_items = edit.pop("input_items")
        result = await _execute_probe_case(
            client,
            model=model,
            max_case_attempts=max_case_attempts,
            input_items=input_items,
            request_kwargs=scenario.request_kwargs,
            baseline_output_digest=scenario.baseline_output_digest,
        )
        mutations.append({**edit, "result": result})
    return {
        **asdict(scenario),
        "mutations": mutations,
        "summary": _scenario_summary(mutations),
    }


def _insertion_variants(scenario: Scenario) -> list[InsertionVariant]:
    variants = [
        InsertionVariant(name="fabricated_assistant_commentary", items=[_message(role="assistant", content="fabricated assistant commentary")]),
        InsertionVariant(
            name="fabricated_assistant_final_answer",
            items=[_message(role="assistant", content="fabricated assistant final", phase="final_answer")],
        ),
        InsertionVariant(name="fabricated_user", items=[_message(role="user", content="fabricated user")]),
        InsertionVariant(name="fabricated_tool_call", items=[_fabricated_tool_call(9001)]),
        InsertionVariant(name="fabricated_tool_output", items=[_fabricated_tool_output("call_fake_probe_output_only", "fabricated tool output only")]),
        InsertionVariant(
            name="fabricated_tool_call_pair",
            items=[
                _fabricated_tool_call(9002, call_id="call_fake_probe_pair"),
                _fabricated_tool_output("call_fake_probe_pair", "fabricated tool output pair"),
            ],
        ),
    ]
    labeled_items = _labeled_items(scenario.input_items)
    seen_bases: set[str] = set()
    for index, label, item in labeled_items:
        if index < scenario.replay_start_index:
            continue
        base = _item_label(item)
        if base not in {"R", "M", "F", "FO"} or base in seen_bases:
            continue
        variants.append(InsertionVariant(name=f"duplicate_real_{label}", items=[deepcopy(item)]))
        seen_bases.add(base)
    adjacent_pairs = _adjacent_pairs(labeled_items, start_index=scenario.replay_start_index)
    for left_label, left_item, right_label, right_item in adjacent_pairs:
        if { _item_label(left_item), _item_label(right_item) } <= {"R", "M", "F", "FO"}:
            variants.append(
                InsertionVariant(
                    name=f"duplicate_real_pair_{left_label}_{right_label}",
                    items=[deepcopy(left_item), deepcopy(right_item)],
                )
            )
    return variants


def _omission_cases(scenario: Scenario) -> list[JSONObject]:
    cases: list[JSONObject] = []
    for index, label, item in _labeled_items(scenario.input_items):
        if index < scenario.replay_start_index:
            continue
        if _item_label(item) not in {"R", "M", "F", "FO"}:
            continue
        mutated = _clone_items(scenario.input_items[:index]) + _clone_items(scenario.input_items[index + 1 :])
        cases.append(
            {
                "category": "omit",
                "input_items": mutated,
                "input_sequence": _sequence_labels(mutated),
                "mutation": f"omit_{label}",
                "omitted_sequence": [label],
            }
        )
    return cases


def _swap_cases(scenario: Scenario) -> list[JSONObject]:
    cases: list[JSONObject] = []
    labeled_items = _labeled_items(scenario.input_items)
    swap_start = max(0, scenario.replay_start_index - 1)
    for index in range(swap_start, len(labeled_items) - 1):
        left_index, left_label, _left_item = labeled_items[index]
        right_index, right_label, _right_item = labeled_items[index + 1]
        mutated = _clone_items(scenario.input_items)
        mutated[left_index], mutated[right_index] = mutated[right_index], mutated[left_index]
        cases.append(
            {
                "category": "swap",
                "input_items": mutated,
                "input_sequence": _sequence_labels(mutated),
                "mutation": f"swap_{left_label}_{right_label}",
                "swapped_sequence": [left_label, right_label],
            }
        )
    return cases


def _move_cases(scenario: Scenario) -> list[JSONObject]:
    cases: list[JSONObject] = []
    gaps = _gap_labels(scenario.input_sequence, replay_start_index=scenario.replay_start_index)
    labeled_items = _labeled_items(scenario.input_items)
    for index, label, item in labeled_items:
        if _item_label(item) not in {"F", "FO"}:
            continue
        for insert_index, gap_label in gaps:
            mutated = _move_span(scenario.input_items, start=index, end=index + 1, target_insert_index=insert_index)
            if mutated is None:
                continue
            cases.append(
                {
                    "category": "move",
                    "gap": gap_label,
                    "input_items": mutated,
                    "input_sequence": _sequence_labels(mutated),
                    "moved_sequence": [label],
                    "mutation": f"move_{label}",
                    "source_label": label,
                }
            )
    for index in range(len(labeled_items) - 1):
        left_index, left_label, left_item = labeled_items[index]
        right_index, right_label, right_item = labeled_items[index + 1]
        if _item_label(left_item) != "F" or _item_label(right_item) != "FO":
            continue
        for insert_index, gap_label in gaps:
            mutated = _move_span(scenario.input_items, start=left_index, end=right_index + 1, target_insert_index=insert_index)
            if mutated is None:
                continue
            cases.append(
                {
                    "category": "move",
                    "gap": gap_label,
                    "input_items": mutated,
                    "input_sequence": _sequence_labels(mutated),
                    "moved_sequence": [left_label, right_label],
                    "mutation": f"move_{left_label}_{right_label}",
                    "source_label": f"{left_label}_{right_label}",
                }
            )
    return cases


def _edit_cases(scenario: Scenario) -> list[JSONObject]:
    cases: list[JSONObject] = []
    for index, label, item in _labeled_items(scenario.input_items):
        if index < scenario.replay_start_index:
            continue
        if _item_label(item) != "M":
            continue
        edits = [
            (
                "edit_message_text",
                _edited_output_message(item, text="HISTORICAL_ASSISTANT_EDIT"),
            ),
            (
                "edit_message_phase_commentary",
                _edited_output_message(item, phase="commentary"),
            ),
            (
                "edit_message_phase_removed",
                _edited_output_message(item, phase=None, remove_phase=True),
            ),
            (
                "edit_message_status_incomplete",
                _edited_output_message(item, status="incomplete"),
            ),
            (
                "edit_message_status_in_progress",
                _edited_output_message(item, status="in_progress"),
            ),
        ]
        for mutation_name, edited in edits:
            if edited == item:
                continue
            mutated = _clone_items(scenario.input_items)
            mutated[index] = edited
            cases.append(
                {
                    "category": "edit",
                    "input_items": mutated,
                    "input_sequence": _sequence_labels(mutated),
                    "edited_sequence": [label],
                    "mutation": f"{mutation_name}_{label}",
                    "target_label": label,
                }
            )
    return cases


def _move_span(items: list[JSONObject], *, start: int, end: int, target_insert_index: int) -> list[JSONObject] | None:
    if target_insert_index >= start and target_insert_index <= end:
        return None
    moved = _clone_items(items[start:end])
    remaining = [*_clone_items(items[:start]), *_clone_items(items[end:])]
    adjusted_index = target_insert_index
    if target_insert_index > end:
        adjusted_index -= end - start
    mutated = [*remaining[:adjusted_index], *moved, *remaining[adjusted_index:]]
    if mutated == items:
        return None
    return mutated


def _gap_labels(sequence: list[str], *, replay_start_index: int) -> list[tuple[int, str]]:
    labels: list[tuple[int, str]] = []
    for insert_index in range(replay_start_index, len(sequence) + 1):
        left = sequence[insert_index - 1]
        right = "$" if insert_index == len(sequence) else sequence[insert_index]
        labels.append((insert_index, f"{left}|{right}"))
    return labels


async def _execute_probe_case(
    client: AsyncOpenAI,
    *,
    model: str,
    max_case_attempts: int,
    input_items: list[JSONObject],
    request_kwargs: JSONObject,
    baseline_output_digest: JSONObject,
) -> JSONObject:
    attempts: list[JSONObject] = []
    for attempt in range(1, max_case_attempts + 1):
        try:
            response = await client.responses.create(
                model=model,
                include=INCLUDE,
                input=_clone_items(input_items),
                **request_kwargs,
            )
        except APIStatusError as exc:
            error = _status_error_body(exc)
            attempts.append({"attempt": attempt, "accepted": False, "error": error, "status_code": exc.status_code})
            if not _is_retryable_status(exc.status_code) or attempt == max_case_attempts:
                return {
                    "accepted": False,
                    "attempts": attempts,
                    "error": error,
                    "status_code": exc.status_code,
                }
            await asyncio.sleep(float(attempt))
            continue
        except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
            error = {"class": type(exc).__name__, "message": str(exc)}
            attempts.append({"attempt": attempt, "accepted": False, "error": error, "status_code": None})
            if attempt == max_case_attempts:
                return {
                    "accepted": False,
                    "attempts": attempts,
                    "error": error,
                    "status_code": None,
                }
            await asyncio.sleep(float(attempt))
            continue
        except Exception as exc:
            error = {"class": type(exc).__name__, "message": str(exc)}
            attempts.append({"attempt": attempt, "accepted": False, "error": error, "status_code": None})
            return {
                "accepted": False,
                "attempts": attempts,
                "error": error,
                "status_code": None,
            }
        output_items = _output_items(response)
        output_digest = _response_digest(output_items)
        comparison = _compare_output_digests(output_digest, baseline_output_digest)
        attempts.append(
            {
                "attempt": attempt,
                "accepted": True,
                "output_digest": output_digest,
                "response_id": getattr(response, "id", None),
            }
        )
        return {
            "accepted": True,
            "attempts": attempts,
            "comparison": comparison,
            "output_digest": output_digest,
            "response_id": getattr(response, "id", None),
        }
    raise AssertionError("probe case loop exhausted unexpectedly")


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 429, 500, 502, 503, 504}


async def _create_response(client: AsyncOpenAI, **kwargs: object) -> object:
    last_error: APIStatusError | None = None
    for attempt in range(1, 4):
        try:
            return await client.responses.create(**kwargs)
        except APIStatusError as exc:
            last_error = exc
            if not _is_retryable_status(exc.status_code) or attempt == 3:
                raise
            await asyncio.sleep(float(attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("unreachable response creation state")


def _status_error_body(exc: APIStatusError) -> JSONObject:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        payload = deepcopy(body)
    else:
        response = getattr(exc, "response", None)
        payload = {
            "class": type(exc).__name__,
            "message": str(exc),
            "text": getattr(response, "text", None),
        }
    request_id = getattr(exc, "request_id", None)
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def _insert_items(items: list[JSONObject], *, insert_index: int, inserted: list[JSONObject]) -> list[JSONObject]:
    return [*_clone_items(items[:insert_index]), *_clone_items(inserted), *_clone_items(items[insert_index:])]


def _message(*, role: str, content: str, phase: str | None = None) -> JSONObject:
    item: JSONObject = {
        "type": "message",
        "role": role,
        "content": content,
    }
    if phase is not None:
        item["phase"] = phase
    return item


def _edited_output_message(
    item: JSONObject,
    *,
    phase: str | None | object = ..., 
    remove_phase: bool = False,
    status: str | None = None,
    text: str | None = None,
) -> JSONObject:
    edited = deepcopy(item)
    if remove_phase:
        edited.pop("phase", None)
    elif phase is not ...:
        if phase is None:
            edited.pop("phase", None)
        else:
            edited["phase"] = phase
    if status is not None:
        edited["status"] = status
    if text is not None:
        content = edited.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                first["text"] = text
        elif isinstance(content, str):
            edited["content"] = text
    return edited


def _fabricated_tool_call(step: int, *, call_id: str | None = None) -> JSONObject:
    return {
        "arguments": json.dumps({"step": step}),
        "call_id": call_id or f"call_fake_probe_{step}",
        "id": f"fc_fake_{step}",
        "name": TOOL_NAME,
        "status": "completed",
        "type": "function_call",
    }


def _fabricated_tool_output(call_id: str, output: str) -> JSONObject:
    return {
        "call_id": call_id,
        "id": f"fco_fake_{call_id}",
        "output": output,
        "status": "completed",
        "type": "function_call_output",
    }


def _function_call_output(function_call: JSONObject, *, suffix: str) -> JSONObject:
    call_id = function_call.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError(f"function_call item is missing call_id: {function_call}")
    return {
        "call_id": call_id,
        "id": f"fco_{suffix}",
        "output": json.dumps({"ok": True, "source": suffix}),
        "status": "completed",
        "type": "function_call_output",
    }


def _first_function_call(items: list[JSONObject]) -> JSONObject | None:
    for item in items:
        if item.get("type") == "function_call":
            return item
    return None


def _has_reasoning(items: list[JSONObject]) -> bool:
    return any(item.get("type") == "reasoning" for item in items)


def _output_items(response: object) -> list[JSONObject]:
    raw_output = getattr(response, "output", None)
    if not isinstance(raw_output, list):
        raise RuntimeError(f"response.output is not a list: {type(raw_output).__name__}")
    return [_dump_item(item) for item in raw_output]


def _dump_item(item: object) -> JSONObject:
    if hasattr(item, "model_dump"):
        value = item.model_dump(mode="json", exclude_none=True)
    elif hasattr(item, "to_dict"):
        value = item.to_dict()
    else:
        value = item
    if not isinstance(value, dict):
        raise RuntimeError(f"response item is not a JSON object: {value!r}")
    return deepcopy(value)


def _clone_items(items: list[JSONObject]) -> list[JSONObject]:
    return deepcopy(items)


def _response_digest(items: list[JSONObject]) -> JSONObject:
    assistant_texts = [_message_text(item) for item in items if item.get("type") == "message" and item.get("role") == "assistant"]
    function_calls = [
        {
            "arguments": item.get("arguments"),
            "name": item.get("name"),
        }
        for item in items
        if item.get("type") == "function_call"
    ]
    function_call_outputs = [item.get("output") for item in items if item.get("type") == "function_call_output"]
    return {
        "assistant_text": "\n\n".join(text for text in assistant_texts if text),
        "assistant_texts": assistant_texts,
        "function_call_outputs": function_call_outputs,
        "function_calls": function_calls,
        "output_sequence": _sequence_labels(items),
        "reasoning_count": sum(1 for item in items if item.get("type") == "reasoning"),
    }


def _compare_output_digests(actual: JSONObject, baseline: JSONObject) -> JSONObject:
    same_output_sequence = actual.get("output_sequence") == baseline.get("output_sequence")
    same_assistant_text = actual.get("assistant_text") == baseline.get("assistant_text")
    same_function_calls = actual.get("function_calls") == baseline.get("function_calls")
    same_function_call_outputs = actual.get("function_call_outputs") == baseline.get("function_call_outputs")
    return {
        "all_key_fields_match": all(
            (
                same_output_sequence,
                same_assistant_text,
                same_function_calls,
                same_function_call_outputs,
            )
        ),
        "same_assistant_text": same_assistant_text,
        "same_function_call_outputs": same_function_call_outputs,
        "same_function_calls": same_function_calls,
        "same_output_sequence": same_output_sequence,
    }


def _message_text(item: JSONObject) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return " ".join(parts)


def _sequence_labels(items: list[JSONObject]) -> list[str]:
    counts: dict[str, int] = {}
    labels: list[str] = []
    for item in items:
        base = _item_label(item)
        index = counts.get(base, 0)
        counts[base] = index + 1
        labels.append(f"{base}{index}")
    return labels


def _labeled_items(items: list[JSONObject]) -> list[tuple[int, str, JSONObject]]:
    return [(index, label, item) for index, (label, item) in enumerate(zip(_sequence_labels(items), items, strict=True))]


def _adjacent_pairs(
    labeled_items: list[tuple[int, str, JSONObject]],
    *,
    start_index: int,
) -> list[tuple[str, JSONObject, str, JSONObject]]:
    pairs: list[tuple[str, JSONObject, str, JSONObject]] = []
    for index in range(start_index, len(labeled_items) - 1):
        _, left_label, left_item = labeled_items[index]
        _, right_label, right_item = labeled_items[index + 1]
        pairs.append((left_label, left_item, right_label, right_item))
    return pairs


def _item_label(item: JSONObject) -> str:
    item_type = item.get("type")
    if item_type == "reasoning":
        return "R"
    if item_type == "message":
        role = item.get("role")
        if role == "assistant":
            return "M"
        if role == "user":
            return "U"
        if isinstance(role, str):
            return role.upper()
        return "MSG"
    if item_type == "function_call":
        return "F"
    if item_type == "function_call_output":
        return "FO"
    if item_type == "item_reference":
        return "REF"
    if item_type == "compaction":
        return "C"
    if isinstance(item_type, str):
        return item_type.upper()
    return "ITEM"


def _scenario_summary(mutations: list[JSONObject]) -> JSONObject:
    accepted = 0
    changed = 0
    rejected = 0
    retried = 0
    statuses: dict[str, int] = {}
    for mutation in mutations:
        result = mutation["result"]
        if result["accepted"]:
            accepted += 1
            if not result["comparison"]["all_key_fields_match"]:
                changed += 1
        else:
            rejected += 1
            statuses[str(result["status_code"])] = statuses.get(str(result["status_code"]), 0) + 1
        if len(result["attempts"]) > 1:
            retried += 1
    return {
        "accepted": accepted,
        "accepted_but_changed": changed,
        "rejected": rejected,
        "rejected_by_status": statuses,
        "retried": retried,
        "total": len(mutations),
    }


def _print_summary(report: JSONObject, *, report_path: Path) -> None:
    print(f"Wrote report to {report_path}")
    print(f"Model: {report['model']}")
    print(f"Answer baseline: {report['answer_pair_baseline']['output_digest']['output_sequence']}")
    print(f"Tool first turn baseline: {report['tool_first_turn_baseline']['output_digest']['output_sequence']}")
    for scenario in report["scenarios"]:
        summary = scenario["summary"]
        print(f"\nScenario: {scenario['name']}")
        print(f"  Notes: {scenario['notes']}")
        print(f"  Input: {scenario['input_sequence']}")
        print(f"  Baseline response: {scenario['baseline_output_digest']['output_sequence']}")
        print(
            "  Summary: "
            f"accepted={summary['accepted']} "
            f"changed={summary['accepted_but_changed']} "
            f"rejected={summary['rejected']} "
            f"retried={summary['retried']}"
        )
        for mutation in scenario["mutations"]:
            result = mutation["result"]
            if result["accepted"]:
                changed = " changed" if not result["comparison"]["all_key_fields_match"] else ""
                outcome = f"ACCEPT{changed}"
            else:
                outcome = f"REJECT {result['status_code']}"
            location = mutation.get("gap") or "/".join(mutation.get("swapped_sequence") or mutation.get("omitted_sequence") or [])
            print(f"  {mutation['category']:>6} {mutation['mutation']:<36} @ {location:<16} -> {outcome}")


if __name__ == "__main__":
    raise SystemExit(main())
