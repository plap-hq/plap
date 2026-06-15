#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging as stdlib_logging
import os
import random
import re
import statistics
import sys
import threading
import warnings
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import duckdb  # noqa: E402
import gepa.optimize_anything as oa  # noqa: E402
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig  # noqa: E402

import plap.llms.accumulator as accumulator_module  # noqa: E402
from plap.llms.accumulator import Accumulator  # noqa: E402
from plap.llms.completions.chat import (  # noqa: E402
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatResponseFormat,
    ChatTool,
    ChatToolCall,
    ChatToolChoiceFunction,
)
from plap.llms.completions.client import ChatCompletionClient, Provider  # noqa: E402
from plap.llms.completions.errors import ChatCompletionUnsupportedRequestError  # noqa: E402
from plap.llms.completions.providers import PROVIDER_BUILDERS  # noqa: E402
from plap.llms.completions.providers.openai import OpenAIProvider  # noqa: E402
from plap.llms.completions.router import ModelRoute, RoutingChatCompletionClient  # noqa: E402
from plap.llms.completions.tokens import measure_request_tokens  # noqa: E402
from plap.llms.json import Outcome, recover  # noqa: E402
from plap.llms.retry import RetryLimitExceededError  # noqa: E402
from plap.llms.retry import complete as retry_complete  # noqa: E402
from plap.settings import Settings  # noqa: E402

DEFAULT_DATABASE = REPO_ROOT / "archive" / "deepswe.duckdb"
DEFAULT_ENV_FILE = REPO_ROOT / "tests" / ".env"
DEFAULT_RUN_DIR = REPO_ROOT / ".dev" / "mimo-gepa"
DEFAULT_TASK_MODEL = "crof/mimo-v2.5-pro,crof/mimo-v2.5-pro"
DEFAULT_JUDGE_MODEL = "crof/deepseek-v4-pro,crof/deepseek-v4-pro"
DEFAULT_REFLECTION_MODEL = "crof/deepseek-v4-pro,crof/deepseek-v4-pro"
DEFAULT_CODEX_LB_BASE_URL = "http://127.0.0.1:2455/v1"
DEFAULT_TEST_TASK_COUNT = 15
DEFAULT_MAX_STAGES = 4
DEFAULT_MAX_METRIC_CALLS = 150
DEFAULT_REFLECTION_MINIBATCH_SIZE = 6
DEFAULT_MAX_WORKERS = 100
DEFAULT_OPTIMIZATION_TRAIN_RATIO = 0.8
DEFAULT_TASK_REASONING_EFFORT = "high"
DEFAULT_JUDGE_REASONING_EFFORT = "xhigh"
DEFAULT_REFLECTOR_REASONING_EFFORT = "xhigh"
DEFAULT_STAGE_MAX_COMPLETION_TOKENS = 4096
DEFAULT_REPLAY_MAX_COMPLETION_TOKENS = 16384
DEFAULT_JUDGE_MAX_COMPLETION_TOKENS = 4096
DEFAULT_CANDIDATE_OVERFIT_MAX_COMPLETION_TOKENS = 1024
DEFAULT_STREAM_FIRST_DELTA_TIMEOUT_SECONDS = 180.0
DEFAULT_JUDGE_STABLE_PROMPT_TOKEN_BUDGET = 850_000
DEFAULT_EVALUATION_TRACE_FILENAME = "evaluation_traces.jsonl"
HIGH_MATCH_QUALITIES = frozenset({"high", "medium"})
SINGLE_TOOL_NAME = "bash"
CODEX_LB_PROVIDER_SLUG = "codex-lb"
CURRENT_CANDIDATE_PARAMETER_NAME = "current_candidate"
FINAL_STAGE_REASONING_PLACEHOLDER = "{{ORIGINAL_REASONING_CONTENT}}"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
SLICE_NO_ISSUE = "no_issue"
SLICE_ISSUE = "issue"
SLICE_ESCALATION = "escalation"
REFERENCE_SCOPE_SAME_MODEL = "same_model_same_task"
REFERENCE_SCOPE_OTHER_MODEL = "other_model_same_task"
REFERENCE_STATUS_FOUND = "reference_found"

WRITE_REDIRECT_RE = re.compile(r"(^|[^0-9])>>?\s*[^&| ]")
WRITE_SED_I_RE = re.compile(r"(^|[\s;|&(])sed\s+-i(\s|$)")
WRITE_PERL_PI_RE = re.compile(r"(^|[\s;|&(])perl\s+-pi(\s|$)")
WRITE_TEE_RE = re.compile(r"(^|[\s;|&(])tee(\s|$)")
WRITE_MUTATION_RE = re.compile(r"(^|[\s;|&(])(mv|cp|rm|patch|git apply)\s")
WRITE_SCRIPT_RE = re.compile(r"write_text|\.write_text\(|json\.dump\(|yaml\.dump\(|open\([^\n]+,[^\n]*[\"\']w")

SEED_CANDIDATE = """<<<TOOL_CALL_STUB>>>
Tool call interrupted by the user.  You are now in a reasoning-only phase. DO NOT TRY AND CALL ANY TOOLS. PLAIN ENGLISH ONLY.
<<<END TOOL_CALL_STUB>>>

<<<STAGE>>>
Think about whether the action you were about to run is safe and correct given the context.  If
it is, state that. If you are uncertain or see a risk, explain what you
should check first. Again, plain text only.
<<<END STAGE>>>

<<<STAGE>>>
Current reasoning trace:

{{ORIGINAL_REASONING_CONTENT}}

Write only the text that should immediately follow the trace above,
continuing the assistant's internal monologue right before acting.  Do NOT
output any tool call, bash command, JSON, or code fence. Do NOT recap the
task or mention the interruption. Output a single paragraph of plain English
reasoning. That is your entire output.

Good examples:
- Wait, I should verify that this file actually exists before trying to edit it.
- Hmm, the safer next move is to cat the current config first.
- Actually, this approach might break the existing tests; I should check.

Bad examples (these will be rejected):
- [Tool call or bash command]
- "Continue from where you left off..."
- "The assistant should now..."
- Any recap of the full task or previous reasoning.
<<<END STAGE>>>"""

GEPA_REFLECTION_PROMPT_TEMPLATE = """You are optimizing a linear conversation-stage program for interrupted coding-agent trajectories.

Goal: preserve correct next moves on no-issue slices, redirect bad next moves on
issue slices, and avoid overclaiming that everything is broken.

Candidate format and execution semantics:
- exactly one <<<TOOL_CALL_STUB>>> ... <<<END TOOL_CALL_STUB>>> block
- one or more <<<STAGE>>> ... <<<END STAGE>>> blocks
- the tool-call stub becomes the synthetic role=tool interruption message
- each stage prompt becomes a new role=user message and gets one role=assistant continuation
- the evaluator runs the stages first, then takes only the final stage
  assistant output text and treats it as the reasoning-trace extension
- for replay, the evaluator appends that extension to the intercepted
  assistant turn's reasoning_content, removes the intercepted tool call,
  resumes the original conversation from that assistant turn, and lets the
  base model generate the replacement bash tool call

The current candidate is:
```
<curr_param>
```

Evaluation data from recent examples:
```
<side_info>
```

Analyze the evaluation data carefully. Pay special attention to:
- failure patterns: what specific errors, edge cases, or bad intervention patterns appear?
- successful patterns: what behaviors or prompts worked well and should be preserved?
- root causes: what underlying issues explain the observed failures?

Do not hardcode task-specific APIs, filenames, function names, parameter names,
exact error strings, SQL fragments, numeric edge cases, or repository-specific
details from a single example unless the same pattern clearly recurs across
multiple examples in the evaluation data. Optimize for reusable intervention
prompts that generalize across unrelated tasks and repositories.

Then write a full improved replacement candidate that:
- addresses the identified failure patterns and root causes
- preserves successful behaviors from the current candidate
- makes meaningful improvements rather than superficial changes

You may:
- rewrite the tool-call stub
- rewrite any stage prompt
- add stages
- remove stages
- reorder stages

Hard constraints:
- preserve the marker syntax exactly
- keep exactly one <<<TOOL_CALL_STUB>>> block
- keep one or more <<<STAGE>>> blocks
- only the final stage may contain {{ORIGINAL_REASONING_CONTENT}}
- stages must not introduce a structured JSON handoff schema between stages

NOTE: When keywords such as "Wait," "Hmm," or "Alternatively," start the final stage output,
it generally manages to steer the assistant harder than not having them.

Return ONLY the complete replacement candidate inside ``` blocks. The output
must be a complete, drop-in replacement for the current candidate. Do not
include explanations, commentary, or markdown outside the ``` blocks."""

JUDGE_SYSTEM_PROMPT = """You are grading a trajectory-intervention pipeline for a coding agent.

You will be given raw chat-completions message arrays describing:
- the original prefix before an intercepted assistant tool-call turn
- the original intercepted assistant turn
- one synthetic replacement assistant turn
- the original tail that followed in the source run
- and, on issue slices, a reference passing transcript for the same task

Judge only the resulting replacement turn.

Don't expect it to know information in the tail. Remember, the assistant has only seen the prefix - only you (the judge) have seen the tail.

Rules:
- Use the raw evidence provided. Do not invent hidden information.
- Reward preserving the original good move on no-issue slices.
- Reward steering away from the failed path on issue slices.
- Penalize weakly justified, generic, misfocused, or obviously bad next moves.
- Reading, inspecting, or verifying can be an excellent move when the visible
  context makes that the right next step.
- Penalize unnecessary detours only when the prefix already supports a more
  direct next move.
- Judge appropriateness relative to the assistant's actual epistemic state
  from the prefix, not by hindsight from the tail.
- Output valid JSON only.

Scoring guide:
- 1.0: Excellent replacement turn. Given only the prefix evidence, this is a
  clearly appropriate next move. It preserves a good original move on preserve
  slices, or productively redirects away from the failed path on recover
  slices.
- 0.75: Strong replacement turn. Safe and well-justified, but slightly less
  direct, less complete, or less targeted than the best available move.
- 0.5: Plausible but limited. Not clearly wrong, but weakly justified, overly
  generic, or only modestly helpful.
- 0.25: Weak. Misfocused, poorly justified by the visible context, or likely
  to waste time.
- 0.0: Bad. Repeats the failed path, abandons an obviously good move,
  contradicts the visible evidence, or is not a sensible next step.

Return:
{
  "overall_score": number from 0.0 to 1.0,
  "feedback": "short but specific explanation"
}
"""

OVERFIT_AUDIT_SYSTEM_PROMPT = """
You are grading whether a candidate interruption-stage program is overfit as a reusable coding-agent intervention prompt.

Judge whether the candidate program itself encodes a broadly reusable
interruption policy, or whether it hardcodes a narrow diagnostic framework
mined from prior examples.

High overfit signals:
- exact filenames, APIs, functions, classes, parameters, command flags, code
  fragments, regexes, or SQL snippets used as operative guidance
- literal error strings or stack-trace fragments
- numeric thresholds or formatting trivia presented as universal rules
- repository-specific assumptions
- fixed numbered checklists or canned "check these N semantics" structures
- preselected failure-family lenses that are not universally appropriate
- abstract but narrow heuristics that look mined from prior examples
- long semantic laundry lists that sound benchmark-shaped rather than
  generally useful
- broad-sounding instructions that actually smuggle in a specific diagnostic
  recipe

Examples of overfit even without repo-specific names:
- "Check these three specific semantics..."
- defaulting to abstraction-boundary checks, case-normalization checks, and
  cross-line or compound-edge-case checks as a bundled policy
- any fixed diagnostic triage template that would be irrelevant on many
  unrelated tasks

Low overfit signals:
- generic safety checks before editing or executing
- generic reminders about prior normalization or transformed inputs
- generic reminders to finish imports, wiring, data updates, and verification

Scoring guide:
- 0.0: strongly generic, minimal, broadly reusable
- 0.25: mostly generic, only mild unnecessary steering
- 0.5: materially shaped around one family of failures
- 0.75: clearly overfit through fixed diagnostic lenses or mined checklists
- 1.0: explicit task memorization or very narrow benchmark-specific prompting

Return valid JSON only.

{
  "overfit_score": number from 0.0 to 1.0,
  "feedback": "short, specific explanation naming the overfit pattern or why it is genuinely generic"
}
"""

BASH_TOOL = ChatTool(
    function=ChatFunctionTool(
        name=SINGLE_TOOL_NAME,
        description="Execute one bash command.",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
        strict=True,
    )
)


@dataclass(frozen=True, slots=True)
class TrialRow:
    trial_name: str
    task_name: str
    passed: bool
    wrong_path_status: str
    first_wrong_path_write_step_ordinal: int | None
    first_wrong_path_write_tool_call_ordinal: int | None
    wrong_path_why: str
    bug_write_status: str
    first_bug_write_step_ordinal: int | None
    first_bug_write_tool_call_ordinal: int | None
    bug_write_why: str
    reference_pass_status: str | None
    reference_pass_trial_name: str | None
    reference_pass_model: str | None
    reference_pass_scope: str | None
    reference_pass_selection_confidence: str | None
    reference_pass_match_quality: str | None
    verifier_stdout_text: str | None
    reference_pass_verifier_stdout_text: str | None


@dataclass(frozen=True, slots=True)
class SliceExample:
    example_id: str
    task_name: str
    trial_name: str
    slice_kind: str
    evaluation_mode: str
    intercepted_step_ordinal: int
    intercepted_tool_call_ordinal: int
    verifier_stdout_text: str | None
    issue_explanation: str | None = None
    reference_pass_trial_name: str | None = None
    reference_pass_verifier_stdout_text: str | None = None
    reference_pass_scope: str | None = None
    reference_pass_match_quality: str | None = None
    reference_pass_selection_confidence: str | None = None


@dataclass(frozen=True, slots=True)
class JudgeOversizeExclusion:
    example_id: str
    task_name: str
    trial_name: str
    slice_kind: str
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    tool_call_stub: str
    stages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class JudgeResult:
    overall_score: float
    feedback: str


@dataclass(frozen=True, slots=True)
class OverfitAuditResult:
    overfit_score: float
    feedback: str
    available: bool = True


@dataclass(frozen=True, slots=True)
class EvalSummary:
    score: float
    side_info: dict[str, Any]


class CandidateExecutionError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    settings: Settings
    database_path: Path
    run_dir: Path
    evaluation_trace_path: Path
    codex_lb_base_url: str
    task_model: str
    judge_model: str
    reflection_model: str
    task_reasoning_effort: str | None
    judge_reasoning_effort: str | None
    reflector_reasoning_effort: str | None
    stage_max_completion_tokens: int
    replay_max_completion_tokens: int
    judge_max_completion_tokens: int
    max_stages: int
    candidate_overfit_results: dict[str, OverfitAuditResult] = field(default_factory=dict, compare=False, repr=False)
    candidate_overfit_futures: dict[str, Future[OverfitAuditResult]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    candidate_overfit_lock: Any = field(default_factory=threading.Lock, compare=False, repr=False)
    evaluation_trace_lock: Any = field(default_factory=threading.Lock, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class _TokenizerConfig:
    tokenizer_hf_repo: str | None
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _judge_tokenizer_config(model: str) -> _TokenizerConfig:
    if "deepseek-v4-pro" in model or "DeepSeek-V4-Pro" in model:
        return _TokenizerConfig(tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Pro")
    return _TokenizerConfig(tokenizer_hf_repo=None)


def _judge_token_cache_dir(runtime: RuntimeConfig) -> Path:
    path = runtime.run_dir / "judge_token_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _judge_token_cache_path(runtime: RuntimeConfig, key: str) -> Path:
    return _judge_token_cache_dir(runtime) / f"{key}.json"


def _candidate_overfit_cache_dir(runtime: RuntimeConfig) -> Path:
    path = runtime.run_dir / "candidate_overfit_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_overfit_cache_path(runtime: RuntimeConfig, key: str) -> Path:
    return _candidate_overfit_cache_dir(runtime) / f"{key}.json"


def _request_cache_key(request: ChatCompletionRequest) -> str:
    payload = {
        "model": request.model,
        "reasoning_effort": request.reasoning_effort,
        "messages": _serialize_messages(list(request.messages)),
        "response_format": None
        if request.response_format is None
        else {
            "type": str(request.response_format.type),
            "name": request.response_format.name,
            "strict": request.response_format.strict,
            "schema": request.response_format.schema,
            "description": request.response_format.description,
        },
    }
    return hashlib.sha256(_json_dumps(payload).encode()).hexdigest()


def _judge_token_cache_key(request: ChatCompletionRequest) -> str:
    return _request_cache_key(request)


def _load_cached_judge_token_count(runtime: RuntimeConfig, key: str) -> int | None:
    path = _judge_token_cache_path(runtime, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    tokens = payload.get("prompt_tokens")
    return tokens if isinstance(tokens, int) else None


def _store_cached_judge_token_count(runtime: RuntimeConfig, key: str, prompt_tokens: int) -> None:
    path = _judge_token_cache_path(runtime, key)
    _write_text_atomic(path, _json_dumps({"prompt_tokens": prompt_tokens}) + "\n")


def _parsed_overfit_audit_value(value: dict[str, Any]) -> OverfitAuditResult:
    feedback = value.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("overfit audit output is missing non-empty feedback")
    overfit_score = value.get("overfit_score")
    if not isinstance(overfit_score, int | float):
        raise TypeError("overfit audit output is missing numeric overfit_score")
    return OverfitAuditResult(
        overfit_score=max(0.0, min(1.0, float(overfit_score))),
        feedback=feedback.strip(),
        available=True,
    )


def _load_cached_candidate_overfit(runtime: RuntimeConfig, key: str) -> OverfitAuditResult | None:
    cached = runtime.candidate_overfit_results.get(key)
    if cached is not None:
        return cached
    path = _candidate_overfit_cache_path(runtime, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError, OSError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        result = _parsed_overfit_audit_value(payload)
    except Exception:
        return None
    runtime.candidate_overfit_results[key] = result
    return result


def _store_cached_candidate_overfit(runtime: RuntimeConfig, key: str, result: OverfitAuditResult) -> None:
    runtime.candidate_overfit_results[key] = result
    path = _candidate_overfit_cache_path(runtime, key)
    with contextlib.suppress(OSError):
        _write_text_atomic(
            path,
            _json_dumps(
                {
                    "overfit_score": result.overfit_score,
                    "feedback": result.feedback,
                }
            )
            + "\n",
        )


def _provider_env_suffix(slug: str) -> str:
    return slug.upper().replace("-", "_")


def _provider_env_var(slug: str) -> str:
    return f"PLAP_LLM_{_provider_env_suffix(slug)}_API_KEY"


def _provider_env_alias(slug: str) -> str:
    return f"{_provider_env_suffix(slug)}_API_KEY"


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


def _populate_provider_aliases() -> None:
    for slug in PROVIDER_BUILDERS:
        target = _provider_env_var(slug)
        source = _provider_env_alias(slug)
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]


def _is_write_like_command(command: str) -> bool:
    if WRITE_SED_I_RE.search(command) or WRITE_PERL_PI_RE.search(command) or WRITE_TEE_RE.search(command):
        return True
    if WRITE_MUTATION_RE.search(command) or WRITE_SCRIPT_RE.search(command):
        return True
    if WRITE_REDIRECT_RE.search(command):
        cleaned = re.sub(r"\b\d>>?\s*/dev/null", "", command)
        cleaned = re.sub(r"\b\d>>?\s*&\d", "", cleaned)
        return bool(WRITE_REDIRECT_RE.search(cleaned))
    return False


def _parse_candidate(candidate_text: str, *, max_stages: int) -> ParsedCandidate:
    stub_match = re.search(
        r"<<<TOOL_CALL_STUB>>>\s*(.*?)\s*<<<END TOOL_CALL_STUB>>>",
        candidate_text,
        flags=re.DOTALL,
    )
    if stub_match is None:
        raise ValueError("candidate is missing a TOOL_CALL_STUB block")
    stages = tuple(
        match.strip()
        for match in re.findall(
            r"<<<STAGE>>>\s*(.*?)\s*<<<END STAGE>>>",
            candidate_text,
            flags=re.DOTALL,
        )
        if match.strip()
    )
    if not stages:
        raise ValueError("candidate must contain at least one STAGE block")
    if len(stages) > max_stages:
        raise ValueError(f"candidate contains {len(stages)} stages, exceeding max_stages={max_stages}")
    return ParsedCandidate(tool_call_stub=stub_match.group(1).strip(), stages=stages)


def _append_reasoning_extension(original: str | None, extension: str) -> str:
    original_text = (original or "").strip()
    extension_text = extension.strip()
    if not original_text:
        return extension_text
    if not extension_text:
        return original_text
    return f"{original_text}\n\n{extension_text}"


def _join_optional_text(*parts: str | None) -> str | None:
    texts = [part.strip() for part in parts if isinstance(part, str) and part.strip()]
    if not texts:
        return None
    return "\n\n".join(texts)


def _render_stage_prompt(
    stage_prompt: str,
    *,
    original_reasoning_content: str | None,
    is_final_stage: bool,
) -> str:
    if not is_final_stage and FINAL_STAGE_REASONING_PLACEHOLDER in stage_prompt:
        raise CandidateExecutionError(f"{FINAL_STAGE_REASONING_PLACEHOLDER} may only appear in the final stage")
    if not is_final_stage:
        return stage_prompt
    return stage_prompt.replace(FINAL_STAGE_REASONING_PLACEHOLDER, original_reasoning_content or "")


def _decode_json_field(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


@lru_cache(maxsize=2048)
def _load_trial_steps(database_path: str, trial_name: str) -> tuple[dict[str, Any], ...]:
    con = duckdb.connect(database_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT step_ordinal, source, message, reasoning_content, tool_calls_json, observation_json
            FROM steps
            WHERE trial_name = ?
            ORDER BY step_ordinal
            """,
            [trial_name],
        ).fetchall()
    finally:
        con.close()

    steps: list[dict[str, Any]] = []
    for step_ordinal, source, message, reasoning_content, tool_calls_json, observation_json in rows:
        steps.append(
            {
                "step_id": int(step_ordinal),
                "source": source,
                "message": message,
                "reasoning_content": reasoning_content,
                "tool_calls": _decode_json_field(tool_calls_json) or [],
                "observation": _decode_json_field(observation_json),
            }
        )
    if not steps:
        raise ValueError(f"no steps found for trial {trial_name!r}")
    return tuple(steps)


def _step_ordinal(step: dict[str, Any]) -> int:
    raw = step.get("step_id")
    if not isinstance(raw, int):
        raise TypeError("trajectory step is missing integer step_id")
    return raw


def _step_tool_calls(step: dict[str, Any]) -> list[dict[str, Any]]:
    value = step.get("tool_calls")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _tool_call_from_step(step: dict[str, Any], tool_call_ordinal: int) -> dict[str, Any]:
    tool_calls = _step_tool_calls(step)
    if tool_call_ordinal < 1 or tool_call_ordinal > len(tool_calls):
        raise ValueError(f"step {_step_ordinal(step)} does not contain tool_call_ordinal={tool_call_ordinal}")
    return tool_calls[tool_call_ordinal - 1]


def _chat_tool_call_from_data(tool_call: dict[str, Any]) -> ChatToolCall:
    return ChatToolCall(
        id=str(tool_call.get("tool_call_id") or "tool_call_missing"),
        name=str(tool_call.get("function_name") or SINGLE_TOOL_NAME),
        arguments=_json_dumps(tool_call.get("arguments") or {}),
    )


def _observation_contents(step: dict[str, Any]) -> list[str]:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return []
    results = observation.get("results")
    if not isinstance(results, list):
        return []
    contents: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        content = result.get("content")
        if isinstance(content, str):
            contents.append(content)
    return contents


def _tool_observation_messages(step: dict[str, Any]) -> list[ChatMessage]:
    tool_calls = _step_tool_calls(step)
    contents = _observation_contents(step)
    if not tool_calls or not contents:
        return []
    if len(tool_calls) == 1:
        return [
            ChatMessage(
                role=ROLE_TOOL,
                content="\n\n".join(contents),
                tool_call_id=str(tool_calls[0].get("tool_call_id") or "tool_call_missing"),
            )
        ]
    if len(contents) == len(tool_calls):
        return [
            ChatMessage(
                role=ROLE_TOOL,
                content=contents[index],
                tool_call_id=str(tool_call.get("tool_call_id") or f"tool_call_{index + 1}"),
            )
            for index, tool_call in enumerate(tool_calls)
        ]
    messages: list[ChatMessage] = []
    for index, tool_call in enumerate(tool_calls):
        if index < len(tool_calls) - 1 and index < len(contents):
            content = contents[index]
        elif index == len(tool_calls) - 1:
            content = "\n\n".join(contents[index:]) if index < len(contents) else ""
        else:
            content = ""
        messages.append(
            ChatMessage(
                role=ROLE_TOOL,
                content=content,
                tool_call_id=str(tool_call.get("tool_call_id") or f"tool_call_{index + 1}"),
            )
        )
    return messages


def _assistant_message_from_step(
    step: dict[str, Any],
    *,
    include_tool_calls: bool,
    reasoning_content: str | None = None,
) -> ChatMessage:
    return ChatMessage(
        role=ROLE_ASSISTANT,
        content=step.get("message") if isinstance(step.get("message"), str) else None,
        reasoning_content=reasoning_content if reasoning_content is not None else step.get("reasoning_content"),
        tool_calls=[_chat_tool_call_from_data(tool_call) for tool_call in _step_tool_calls(step)] if include_tool_calls else [],
    )


def _base_message_from_step(step: dict[str, Any]) -> ChatMessage | None:
    source = step.get("source")
    message = step.get("message")
    if source == ROLE_SYSTEM and isinstance(message, str):
        return ChatMessage(role=ROLE_SYSTEM, content=message)
    if source == ROLE_USER and isinstance(message, str):
        return ChatMessage(role=ROLE_USER, content=message)
    return None


def _messages_for_step(step: dict[str, Any]) -> list[ChatMessage]:
    base = _base_message_from_step(step)
    if base is not None:
        return [base]
    if step.get("source") != "agent":
        return []
    return [
        _assistant_message_from_step(step, include_tool_calls=True),
        *_tool_observation_messages(step),
    ]


def _full_trajectory_messages(steps: tuple[dict[str, Any], ...]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for step in steps:
        messages.extend(_messages_for_step(step))
    return messages


def _prefix_messages_before_intercept(
    steps: tuple[dict[str, Any], ...],
    *,
    intercepted_step_ordinal: int,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for step in steps:
        ordinal = _step_ordinal(step)
        if ordinal < intercepted_step_ordinal:
            messages.extend(_messages_for_step(step))
            continue
        break
    return messages


def _intercepted_stage_context_messages(
    steps: tuple[dict[str, Any], ...],
    *,
    intercepted_step_ordinal: int,
    tool_call_stub: str,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for step in steps:
        ordinal = _step_ordinal(step)
        if ordinal < intercepted_step_ordinal:
            messages.extend(_messages_for_step(step))
            continue
        if ordinal == intercepted_step_ordinal:
            tool_calls = _step_tool_calls(step)
            if len(tool_calls) != 1:
                raise ValueError(f"intercepted step {intercepted_step_ordinal} must contain exactly one tool call in v1")
            messages.append(_assistant_message_from_step(step, include_tool_calls=True))
            messages.append(
                ChatMessage(
                    role=ROLE_TOOL,
                    content=tool_call_stub,
                    tool_call_id=str(tool_calls[0].get("tool_call_id") or "tool_call_missing"),
                )
            )
            return messages
        break
    raise ValueError(f"intercepted step {intercepted_step_ordinal} not found in steps")


def _original_tail_messages(
    steps: tuple[dict[str, Any], ...],
    *,
    intercepted_step_ordinal: int,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for step in steps:
        ordinal = _step_ordinal(step)
        if ordinal < intercepted_step_ordinal:
            continue
        if ordinal == intercepted_step_ordinal:
            messages.extend(_tool_observation_messages(step))
            continue
        messages.extend(_messages_for_step(step))
    return messages


def _synthetic_replacement_turn(
    intercepted_step: dict[str, Any],
    *,
    reasoning_extension: str,
    replay_result_message: ChatMessage,
) -> ChatMessage:
    extended_reasoning = _append_reasoning_extension(intercepted_step.get("reasoning_content"), reasoning_extension)
    return ChatMessage(
        role=ROLE_ASSISTANT,
        content=_join_optional_text(
            intercepted_step.get("message") if isinstance(intercepted_step.get("message"), str) else None,
            replay_result_message.content,
        ),
        reasoning_content=_join_optional_text(extended_reasoning, replay_result_message.reasoning_content),
        tool_calls=list(replay_result_message.tool_calls),
    )


def _serialize_tool_call(tool_call: ChatToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": str(message.role)}
    if message.content is not None:
        payload["content"] = message.content
    if message.reasoning_content is not None:
        payload["reasoning_content"] = message.reasoning_content
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [_serialize_tool_call(tool_call) for tool_call in message.tool_calls]
    return payload


def _serialize_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [_serialize_message(message) for message in messages]


def _candidate_text_hash(candidate_text: str) -> str:
    return hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()


def _append_jsonl_record(path: Path, *, lock: Any, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _evaluation_trace_base_record(
    candidate_text: str,
    *,
    event: str,
    example: SliceExample,
) -> dict[str, Any]:
    return {
        "event": event,
        "candidate_hash": _candidate_text_hash(candidate_text),
        "example_id": example.example_id,
        "task_name": example.task_name,
        "trial_name": example.trial_name,
        "slice_kind": example.slice_kind,
        "evaluation_mode": example.evaluation_mode,
        "intercepted_step_ordinal": example.intercepted_step_ordinal,
        "intercepted_tool_call_ordinal": example.intercepted_tool_call_ordinal,
    }


def _log_evaluation_trace_result(
    runtime: RuntimeConfig,
    *,
    candidate_text: str,
    example: SliceExample,
    original_intercepted_turn: ChatMessage,
    stage_messages: list[ChatMessage],
    synthetic_replacement_turn: ChatMessage,
    judge_result: JudgeResult,
    overfit_audit: OverfitAuditResult,
) -> None:
    record = _evaluation_trace_base_record(candidate_text, event="evaluation_result", example=example)
    record.update(
        {
            "original_intercepted_turn": _serialize_message(original_intercepted_turn),
            "interrupted_stage_messages": _serialize_messages(stage_messages),
            "synthetic_replacement_turn": _serialize_message(synthetic_replacement_turn),
            "judge_score": judge_result.overall_score,
            "judge_feedback": judge_result.feedback,
            "genericity_available": overfit_audit.available,
            "genericity_feedback": overfit_audit.feedback,
            "genericity_score": None if not overfit_audit.available else 1.0 - overfit_audit.overfit_score,
        }
    )
    _append_jsonl_record(runtime.evaluation_trace_path, lock=runtime.evaluation_trace_lock, record=record)


def _log_evaluation_trace_error(
    runtime: RuntimeConfig,
    *,
    candidate_text: str,
    example: SliceExample,
    error: str,
    candidate_prefix: str | None = None,
) -> None:
    record = _evaluation_trace_base_record(candidate_text, event="evaluation_error", example=example)
    record["error"] = error
    if candidate_prefix is not None:
        record["candidate_prefix"] = candidate_prefix
    _append_jsonl_record(runtime.evaluation_trace_path, lock=runtime.evaluation_trace_lock, record=record)


def _reference_rank(row: TrialRow) -> tuple[int, int, int, int, str]:
    quality_rank = {"high": 0, "medium": 1}.get(row.reference_pass_match_quality or "", 2)
    scope_rank = {REFERENCE_SCOPE_SAME_MODEL: 0, REFERENCE_SCOPE_OTHER_MODEL: 1}.get(row.reference_pass_scope or "", 2)
    issue_boundary = _earliest_issue_boundary(row)
    issue_step = issue_boundary[0] if issue_boundary is not None else 10**9
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(row.reference_pass_selection_confidence or "", 3)
    return (quality_rank, scope_rank, issue_step, confidence_rank, row.trial_name)


def _wrong_path_boundary(row: TrialRow) -> tuple[int, int] | None:
    if row.first_wrong_path_write_step_ordinal is None:
        return None
    return row.first_wrong_path_write_step_ordinal, row.first_wrong_path_write_tool_call_ordinal or 1


def _bug_boundary(row: TrialRow) -> tuple[int, int] | None:
    if row.first_bug_write_step_ordinal is None:
        return None
    return row.first_bug_write_step_ordinal, row.first_bug_write_tool_call_ordinal or 1


def _earliest_issue_boundary(row: TrialRow) -> tuple[int, int] | None:
    boundaries = [boundary for boundary in (_bug_boundary(row), _wrong_path_boundary(row)) if boundary is not None]
    if not boundaries:
        return None
    return min(boundaries)


def _later_wrong_path_boundary(row: TrialRow) -> tuple[int, int] | None:
    earliest = _earliest_issue_boundary(row)
    wrong = _wrong_path_boundary(row)
    if earliest is None or wrong is None or wrong <= earliest:
        return None
    return wrong


def _task_bucket(task_examples: list[SliceExample]) -> tuple[bool, bool, bool]:
    kinds = {example.slice_kind for example in task_examples}
    return (SLICE_NO_ISSUE in kinds, SLICE_ISSUE in kinds, SLICE_ESCALATION in kinds)


def _bucket_task_names(task_to_examples: dict[str, list[SliceExample]]) -> dict[tuple[bool, bool, bool], list[str]]:
    buckets: dict[tuple[bool, bool, bool], list[str]] = defaultdict(list)
    for task_name, examples in task_to_examples.items():
        if not examples:
            continue
        buckets[_task_bucket(examples)].append(task_name)
    return buckets


def _test_bucket_allocations(
    bucket_sizes: dict[tuple[bool, bool, bool], int],
    *,
    test_task_count: int,
) -> dict[tuple[bool, bool, bool], int]:
    total_tasks = sum(bucket_sizes.values())
    if test_task_count <= 0 or total_tasks == 0:
        return dict.fromkeys(bucket_sizes, 0)
    ratio = test_task_count / total_tasks
    allocations = {bucket: int(size * ratio) for bucket, size in bucket_sizes.items()}
    remainders = sorted(
        ((size * ratio - allocations[bucket], bucket) for bucket, size in bucket_sizes.items()),
        reverse=True,
    )
    remaining = test_task_count - sum(allocations.values())
    for _, bucket in remainders[:remaining]:
        allocations[bucket] += 1
    return allocations


def _partition_task_names(
    task_to_examples: dict[str, list[SliceExample]],
    *,
    seed: int,
    test_task_count: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)
    buckets = _bucket_task_names(task_to_examples)
    bucket_sizes = {bucket: len(task_names) for bucket, task_names in buckets.items()}
    test_allocations = _test_bucket_allocations(bucket_sizes, test_task_count=test_task_count)
    train_tasks: set[str] = set()
    val_tasks: set[str] = set()
    test_tasks: set[str] = set()

    for bucket, task_names in buckets.items():
        shuffled = list(task_names)
        rng.shuffle(shuffled)
        test_n = test_allocations.get(bucket, 0)
        test_chunk = shuffled[:test_n]
        remainder = shuffled[test_n:]
        if len(remainder) <= 1:
            train_n = len(remainder)
        else:
            train_n = round(len(remainder) * DEFAULT_OPTIMIZATION_TRAIN_RATIO)
            train_n = max(1, min(len(remainder) - 1, train_n))
        train_chunk = remainder[:train_n]
        val_chunk = remainder[train_n:]
        test_tasks.update(test_chunk)
        train_tasks.update(train_chunk)
        val_tasks.update(val_chunk)

    return train_tasks, val_tasks, test_tasks


def _task_to_examples(examples: list[SliceExample]) -> dict[str, list[SliceExample]]:
    grouped: dict[str, list[SliceExample]] = defaultdict(list)
    for example in examples:
        grouped[example.task_name].append(example)
    return grouped


def _filter_examples_for_tasks(examples: list[SliceExample], task_names: set[str]) -> list[SliceExample]:
    return [example for example in examples if example.task_name in task_names]


class _BalancedIssueNoIssueBatchSampler:
    """Keeps each reflection minibatch split evenly between no-issue and recover-like slices."""

    def __init__(self, minibatch_size: int, rng: random.Random | None = None) -> None:
        if minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if minibatch_size % 2 != 0:
            raise ValueError("balanced issue/no_issue minibatches require an even minibatch size")
        self.minibatch_size = minibatch_size
        self._per_bucket = minibatch_size // 2
        self._batches: list[list[int]] = []
        self._epoch = -1
        self._last_trainset_size = 0
        self.rng = rng if rng is not None else random.Random(0)

    def _bucket_ids(self, loader) -> tuple[list[int], list[int]]:
        ids = list(loader.all_ids())
        examples = loader.fetch(ids)
        no_issue_ids: list[int] = []
        recover_ids: list[int] = []
        for data_id, example in zip(ids, examples, strict=True):
            if not isinstance(example, SliceExample):
                raise TypeError("balanced minibatch sampler expects SliceExample items")
            if example.slice_kind == SLICE_NO_ISSUE:
                no_issue_ids.append(int(data_id))
            else:
                recover_ids.append(int(data_id))
        if not no_issue_ids or not recover_ids:
            raise ValueError("balanced minibatch sampler requires both no_issue and recover-like slices")
        return no_issue_ids, recover_ids

    def _padded_bucket(self, ids: list[int], *, target_size: int) -> list[int]:
        padded = list(ids)
        if not padded:
            return padded
        index = 0
        while len(padded) < target_size:
            padded.append(ids[index % len(ids)])
            index += 1
        return padded

    def _refresh_batches(self, loader) -> None:
        trainset_size = len(loader)
        self._last_trainset_size = trainset_size
        no_issue_ids, recover_ids = self._bucket_ids(loader)
        self.rng.shuffle(no_issue_ids)
        self.rng.shuffle(recover_ids)
        batch_count = max(
            (len(no_issue_ids) + self._per_bucket - 1) // self._per_bucket,
            (len(recover_ids) + self._per_bucket - 1) // self._per_bucket,
        )
        target_size = batch_count * self._per_bucket
        padded_no_issue = self._padded_bucket(no_issue_ids, target_size=target_size)
        padded_recover = self._padded_bucket(recover_ids, target_size=target_size)
        batches: list[list[int]] = []
        for index in range(batch_count):
            start = index * self._per_bucket
            batch_ids = [
                *padded_no_issue[start : start + self._per_bucket],
                *padded_recover[start : start + self._per_bucket],
            ]
            self.rng.shuffle(batch_ids)
            batches.append(batch_ids)
        self._batches = batches

    def next_minibatch_ids(self, loader, state) -> list[int]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")
        if not self._batches or trainset_size != self._last_trainset_size:
            self._refresh_batches(loader)
            self._epoch = state.i // len(self._batches)
        assert self._batches, "balanced sampler must have at least one batch"
        curr_epoch = state.i // len(self._batches)
        if curr_epoch > self._epoch:
            self._refresh_batches(loader)
            self._epoch = curr_epoch
        batch_index = state.i % len(self._batches)
        return list(self._batches[batch_index])


def _recover_candidate_rank(example: SliceExample) -> tuple[int, int, int, int, int, str]:
    quality_rank = {"high": 0, "medium": 1}.get(example.reference_pass_match_quality or "", 2)
    scope_rank = {REFERENCE_SCOPE_SAME_MODEL: 0, REFERENCE_SCOPE_OTHER_MODEL: 1}.get(example.reference_pass_scope or "", 2)
    slice_kind_rank = {SLICE_ISSUE: 0, SLICE_ESCALATION: 1}.get(example.slice_kind, 2)
    confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(example.reference_pass_selection_confidence or "", 3)
    return (quality_rank, scope_rank, slice_kind_rank, example.intercepted_step_ordinal, confidence_rank, example.trial_name)


def _dataset_stats(examples: list[SliceExample]) -> dict[str, Any]:
    by_kind: dict[str, int] = defaultdict(int)
    by_task: set[str] = set()
    for example in examples:
        by_kind[example.slice_kind] += 1
        by_task.add(example.task_name)
    return {
        "tasks": len(by_task),
        "examples": len(examples),
        "by_kind": dict(sorted(by_kind.items())),
    }


def _issue_explanation_for_boundary(row: TrialRow, boundary: tuple[int, int] | None) -> str | None:
    if boundary is None:
        return None
    bug_boundary = _bug_boundary(row)
    wrong_boundary = _wrong_path_boundary(row)
    explanations: list[str] = []
    if bug_boundary == boundary and row.bug_write_why.strip():
        explanations.append(row.bug_write_why.strip())
    if wrong_boundary == boundary and row.wrong_path_why.strip() and row.wrong_path_why.strip() not in explanations:
        explanations.append(row.wrong_path_why.strip())
    if not explanations:
        return None
    return "\n\n".join(explanations)


def _choose_no_issue_examples(rows: list[TrialRow], write_like_steps_by_trial: dict[str, list[int]]) -> list[SliceExample]:
    examples: list[SliceExample] = []
    for row in rows:
        write_steps = sorted(write_like_steps_by_trial.get(row.trial_name, []))
        if not write_steps:
            continue
        median_index = len(write_steps) // 2
        intercepted_step_ordinal = write_steps[median_index]
        examples.append(
            SliceExample(
                example_id=f"{SLICE_NO_ISSUE}:{row.trial_name}:{intercepted_step_ordinal}",
                task_name=row.task_name,
                trial_name=row.trial_name,
                slice_kind=SLICE_NO_ISSUE,
                evaluation_mode="preserve",
                intercepted_step_ordinal=intercepted_step_ordinal,
                intercepted_tool_call_ordinal=1,
                verifier_stdout_text=row.verifier_stdout_text,
            )
        )
    return examples


def _choose_issue_examples(rows: list[TrialRow], *, include_escalation_slices: bool) -> list[SliceExample]:
    examples: list[SliceExample] = []
    for row in rows:
        earliest = _earliest_issue_boundary(row)
        if earliest is None:
            continue
        step_ordinal, tool_ordinal = earliest
        examples.append(
            SliceExample(
                example_id=f"{SLICE_ISSUE}:{row.trial_name}:{step_ordinal}",
                task_name=row.task_name,
                trial_name=row.trial_name,
                slice_kind=SLICE_ISSUE,
                evaluation_mode="recover",
                intercepted_step_ordinal=step_ordinal,
                intercepted_tool_call_ordinal=tool_ordinal,
                verifier_stdout_text=row.verifier_stdout_text,
                issue_explanation=_issue_explanation_for_boundary(row, earliest),
                reference_pass_trial_name=row.reference_pass_trial_name,
                reference_pass_verifier_stdout_text=row.reference_pass_verifier_stdout_text,
                reference_pass_scope=row.reference_pass_scope,
                reference_pass_match_quality=row.reference_pass_match_quality,
                reference_pass_selection_confidence=row.reference_pass_selection_confidence,
            )
        )
        if not include_escalation_slices:
            continue
        later_wrong = _later_wrong_path_boundary(row)
        if later_wrong is None:
            continue
        wrong_step_ordinal, wrong_tool_ordinal = later_wrong
        examples.append(
            SliceExample(
                example_id=f"{SLICE_ESCALATION}:{row.trial_name}:{wrong_step_ordinal}",
                task_name=row.task_name,
                trial_name=row.trial_name,
                slice_kind=SLICE_ESCALATION,
                evaluation_mode="recover",
                intercepted_step_ordinal=wrong_step_ordinal,
                intercepted_tool_call_ordinal=wrong_tool_ordinal,
                verifier_stdout_text=row.verifier_stdout_text,
                issue_explanation=_issue_explanation_for_boundary(row, later_wrong),
                reference_pass_trial_name=row.reference_pass_trial_name,
                reference_pass_verifier_stdout_text=row.reference_pass_verifier_stdout_text,
                reference_pass_scope=row.reference_pass_scope,
                reference_pass_match_quality=row.reference_pass_match_quality,
                reference_pass_selection_confidence=row.reference_pass_selection_confidence,
            )
        )
    return examples


def _fetch_trial_rows(con: duckdb.DuckDBPyConnection) -> list[TrialRow]:
    rows = con.execute(
        """
        SELECT a.trial_name,
               a.task_name,
               a.passed,
               a.wrong_path_status,
               a.first_wrong_path_write_step_ordinal,
               a.first_wrong_path_write_tool_call_ordinal,
               a.wrong_path_why,
               a.bug_write_status,
               a.first_bug_write_step_ordinal,
               a.first_bug_write_tool_call_ordinal,
               a.bug_write_why,
               a.reference_pass_status,
               a.reference_pass_trial_name,
               a.reference_pass_model,
               a.reference_pass_scope,
               a.reference_pass_selection_confidence,
               a.reference_pass_match_quality,
               a.verifier_stdout_text,
               r.verifier_stdout_text AS reference_pass_verifier_stdout_text
        FROM annotated_trials AS a
        LEFT JOIN trials AS r ON r.trial_name = a.reference_pass_trial_name
        WHERE lower(a.model) LIKE '%mimo%'
        """
    ).fetchall()
    return [TrialRow(*row) for row in rows]


def _fetch_write_like_steps_by_trial(
    con: duckdb.DuckDBPyConnection,
    trial_names: list[str],
) -> dict[str, list[int]]:
    if not trial_names:
        return {}
    rows = con.execute(
        """
        SELECT trial_name, step_ordinal, command
        FROM tool_calls
        WHERE trial_name IN (SELECT unnest(?))
        ORDER BY trial_name, step_ordinal
        """,
        [trial_names],
    ).fetchall()
    write_like: dict[str, list[int]] = defaultdict(list)
    for trial_name, step_ordinal, command in rows:
        if isinstance(command, str) and _is_write_like_command(command):
            write_like[str(trial_name)].append(int(step_ordinal))
    return write_like


def _build_slice_examples(
    con: duckdb.DuckDBPyConnection,
    *,
    include_escalation_slices: bool,
) -> list[SliceExample]:
    rows = _fetch_trial_rows(con)
    clean_rows = [
        row
        for row in rows
        if row.passed and row.wrong_path_status == "no_clear_wrong_write" and row.bug_write_status == "no_clear_bug_write"
    ]
    issue_rows = [
        row
        for row in rows
        if (
            not row.passed
            and row.reference_pass_status == REFERENCE_STATUS_FOUND
            and row.reference_pass_match_quality in HIGH_MATCH_QUALITIES
        )
    ]
    write_like_steps_by_trial = _fetch_write_like_steps_by_trial(con, [row.trial_name for row in clean_rows])
    return [
        *_choose_no_issue_examples(clean_rows, write_like_steps_by_trial),
        *_choose_issue_examples(issue_rows, include_escalation_slices=include_escalation_slices),
    ]


def _settings_from_env() -> Settings:
    return Settings(
        api_key_pepper="gepa-script-pepper",
        database_url="postgresql+asyncpg://example/test",
        sealing_keys=["a" * 43],
    )


class _CodexLBProvider(OpenAIProvider):
    def lookup(self, name: str) -> tuple[Any, ...]:
        if not name.strip():
            raise ChatCompletionUnsupportedRequestError("unsupported codex-lb model: empty model name")
        return ()


def _build_codex_lb_provider(*, api_key: str, base_url: str) -> Provider:
    return _CodexLBProvider(
        name=CODEX_LB_PROVIDER_SLUG,
        api_key=api_key,
        base_url=base_url,
        quirks=(),
        models={},
    )


def _script_provider_builders(*, codex_lb_base_url: str) -> dict[str, Any]:
    builders: dict[str, Any] = dict(PROVIDER_BUILDERS)
    builders[CODEX_LB_PROVIDER_SLUG] = lambda *, api_key: _build_codex_lb_provider(
        api_key=api_key,
        base_url=codex_lb_base_url,
    )
    return builders


def _build_script_providers(settings: Settings, *, codex_lb_base_url: str) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}
    for slug, build in _script_provider_builders(codex_lb_base_url=codex_lb_base_url).items():
        api_key = settings.llm_api_keys.get(slug)
        if not api_key:
            continue
        providers[f"{slug}/"] = build(api_key=api_key)
    return providers


def _build_chat_completion_client(runtime: RuntimeConfig) -> RoutingChatCompletionClient:
    providers = _build_script_providers(runtime.settings, codex_lb_base_url=runtime.codex_lb_base_url)
    routes = [ModelRoute(prefix=prefix, client=ChatCompletionClient(provider)) for prefix, provider in providers.items()]
    return RoutingChatCompletionClient(routes, stream_first_delta_timeout_seconds=DEFAULT_STREAM_FIRST_DELTA_TIMEOUT_SECONDS)


def _configured_route_prefixes(settings: Settings, *, codex_lb_base_url: str) -> tuple[str, ...]:
    _ = codex_lb_base_url
    return tuple(f"{slug}/" for slug in _script_provider_builders(codex_lb_base_url=codex_lb_base_url) if settings.llm_api_keys.get(slug))


def _ensure_model_routable(model: str, settings: Settings, *, codex_lb_base_url: str) -> None:
    prefixes = _configured_route_prefixes(settings, codex_lb_base_url=codex_lb_base_url)
    if any(model.startswith(prefix) for prefix in prefixes):
        return
    configured = ", ".join(sorted(prefixes)) or "<none>"
    raise SystemExit(f"Model {model!r} does not match any configured route prefix. Configured prefixes: {configured}")


def _new_chat_client(runtime: RuntimeConfig) -> RoutingChatCompletionClient:
    # Each evaluation/reflection call gets a fresh routed client so we never
    # reuse AsyncOpenAI-backed transports across separately created event loops.
    return _build_chat_completion_client(runtime)


async def _close_chat_client(client: RoutingChatCompletionClient) -> None:
    await client.aclose()


async def _complete_via_stream(
    client: RoutingChatCompletionClient,
    request: ChatCompletionRequest,
) -> ChatMessage:
    accumulator = Accumulator(tools=tuple(request.tools))
    final_message: ChatMessage | None = None
    async for delta in client.stream(request):
        snapshot = accumulator.apply(delta)
        if snapshot.results:
            final_message = snapshot.results[-1].message
    if final_message is None:
        raise CandidateExecutionError("stream completed without a final assistant message")
    return final_message


def _chat_messages_from_reflection_prompt(prompt: str | list[dict[str, Any]]) -> list[ChatMessage]:
    if isinstance(prompt, str):
        return [ChatMessage(role=ROLE_USER, content=prompt)]
    messages: list[ChatMessage] = []
    for item in prompt:
        role = item.get("role")
        if not isinstance(role, str):
            raise TypeError("reflection prompt message is missing a string role")
        content = item.get("content")
        rendered_content = content if isinstance(content, str) or content is None else _json_dumps(content)
        messages.append(ChatMessage(role=role, content=rendered_content))
    return messages


def _make_reflection_lm(runtime: RuntimeConfig):
    def reflection_lm(prompt: str | list[dict[str, Any]]) -> str:
        async def run() -> str:
            client = _new_chat_client(runtime)
            try:
                result = await _complete_via_stream(
                    client,
                    ChatCompletionRequest(
                        model=runtime.reflection_model,
                        messages=_chat_messages_from_reflection_prompt(prompt),
                        reasoning_effort=runtime.reflector_reasoning_effort,
                        max_completion_tokens=runtime.judge_max_completion_tokens,
                        temperature=0,
                    ),
                )
                return result.content or ""
            finally:
                await _close_chat_client(client)

        return asyncio.run(run())

    return reflection_lm


def _messages_json_block(label: str, messages: list[ChatMessage]) -> str:
    return f"{label}:\n```json\n{_json_dumps(_serialize_messages(messages))}\n```"


def _text_block(label: str, text: str | None) -> str:
    body = text or ""
    return f"{label}:\n```text\n{body}\n```"


def _judge_prompt(
    *,
    evaluation_mode: str,
    original_prefix_messages: list[ChatMessage],
    original_intercepted_turn: ChatMessage,
    synthetic_replacement_turn: ChatMessage | None,
    original_tail_messages: list[ChatMessage],
    verifier_stdout_text: str | None,
    issue_explanation: str | None,
    reference_messages: list[ChatMessage] | None,
    reference_verifier_stdout_text: str | None,
) -> str:
    sections = [_text_block("evaluation_mode", evaluation_mode)]
    if issue_explanation is not None:
        sections.append(_text_block("issue_explanation", issue_explanation))
    sections.extend(
        [
            _messages_json_block("original_prefix_messages", original_prefix_messages),
            _messages_json_block("original_intercepted_turn", [original_intercepted_turn]),
            _messages_json_block("original_tail_messages", original_tail_messages),
            _text_block("verifier_stdout_text", verifier_stdout_text),
        ]
    )
    if reference_messages is not None:
        sections.append(_messages_json_block("reference_pass_messages", reference_messages))
    if reference_verifier_stdout_text is not None:
        sections.append(_text_block("reference_pass_verifier_stdout_text", reference_verifier_stdout_text))
    if synthetic_replacement_turn is not None:
        sections.append(_messages_json_block("synthetic_replacement_turn", [synthetic_replacement_turn]))
    return "\n\n".join(sections)


def _candidate_overfit_prompt(candidate_text: str) -> str:
    return f"""Candidate:
```
{candidate_text}
```"""


def _parse_judge_result(raw_text: str) -> JudgeResult:
    parsed = recover(raw_text, partial=False)
    if parsed.outcome == Outcome.REJECTED or not isinstance(parsed.value, dict):
        raise ValueError(f"judge returned unparseable output: {raw_text[:400]}")
    value = parsed.value
    feedback = value.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError("judge output is missing non-empty feedback")
    overall_score = value.get("overall_score")
    if not isinstance(overall_score, int | float):
        raise TypeError("judge output is missing numeric overall_score")
    return JudgeResult(
        overall_score=max(0.0, min(1.0, float(overall_score))),
        feedback=feedback.strip(),
    )


def _parse_overfit_audit_result(raw_text: str) -> OverfitAuditResult:
    parsed = recover(raw_text, partial=False)
    if parsed.outcome == Outcome.REJECTED or not isinstance(parsed.value, dict):
        raise ValueError(f"overfit audit returned unparseable output: {raw_text[:400]}")
    return _parsed_overfit_audit_value(parsed.value)


async def _retry_on_invalid_judge_output(result, request) -> str | None:
    _ = request
    try:
        _parse_judge_result(result.message.content or "")
    except Exception:
        return (
            "Your previous answer could not be used as written.\n\n"
            "Problem:\n"
            "- The reply was not a valid JSON object with numeric `overall_score` and non-empty `feedback`.\n\n"
            "Reply again for the same judging task.\n"
            "- Output valid JSON only.\n"
            "- Include exactly the keys `overall_score` and `feedback`.\n"
            "- `overall_score` must be a number between 0 and 1.\n"
            "- `feedback` must be a non-empty string."
        )
    return None


async def _retry_on_invalid_overfit_audit_output(result, request) -> str | None:
    _ = request
    try:
        _parse_overfit_audit_result(result.message.content or "")
    except Exception:
        return (
            "Your previous answer could not be used as written.\n\n"
            "Problem:\n"
            "- The reply was not a valid JSON object with numeric `overfit_score` and non-empty `feedback`.\n\n"
            "Reply again for the same overfit-audit task.\n"
            "- Output valid JSON only.\n"
            "- Include exactly the keys `overfit_score` and `feedback`.\n"
            "- `overfit_score` must be a number between 0 and 1.\n"
            "- `feedback` must be a non-empty string."
        )
    return None


async def _run_stage_completion(
    client: RoutingChatCompletionClient,
    *,
    model: str,
    messages: list[ChatMessage],
    reasoning_effort: str | None,
    max_completion_tokens: int,
) -> ChatMessage:
    result = await _complete_via_stream(
        client,
        ChatCompletionRequest(
            model=model,
            messages=messages,
            # Stage execution should preserve the tool surface the model has
            # been operating with, but it must not spend the action yet.
            tools=[BASH_TOOL],
            tool_choice="none",
            parallel_tool_calls=False,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            temperature=0,
        ),
    )
    if result.tool_calls:
        raise CandidateExecutionError("stage completion emitted tool calls, but stages must emit plain assistant text only")
    return result


async def _run_replay_completion(
    client: RoutingChatCompletionClient,
    *,
    model: str,
    messages: list[ChatMessage],
    reasoning_effort: str | None,
    max_completion_tokens: int,
) -> ChatMessage:
    return await _complete_via_stream(
        client,
        ChatCompletionRequest(
            model=model,
            messages=messages,
            tools=[BASH_TOOL],
            tool_choice=ChatToolChoiceFunction(name=SINGLE_TOOL_NAME),
            parallel_tool_calls=False,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            temperature=0,
        ),
    )


async def _run_judge_completion(
    client: RoutingChatCompletionClient,
    *,
    model: str,
    prompt: str,
    reasoning_effort: str | None,
    max_completion_tokens: int,
) -> JudgeResult:
    base_request = ChatCompletionRequest(
        model=model,
        messages=[
            ChatMessage(role=ROLE_SYSTEM, content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role=ROLE_USER, content=prompt),
        ],
        response_format=ChatResponseFormat(type="json_object"),
        reasoning_effort=reasoning_effort,
        max_completion_tokens=max_completion_tokens,
        temperature=0,
    )
    snapshot = await retry_complete(
        client,
        next_request=lambda history: replace(base_request, messages=[*base_request.messages, *history.messages]),
        validators=(_retry_on_invalid_judge_output,),
        max_attempts=3,
    )
    if not snapshot.results:
        raise RetryLimitExceededError(last_retry_message=None)
    return _parse_judge_result(snapshot.results[-1].message.content or "")


async def _run_overfit_audit_completion(
    client: RoutingChatCompletionClient,
    *,
    request: ChatCompletionRequest,
) -> OverfitAuditResult:
    snapshot = await retry_complete(
        client,
        next_request=lambda history: replace(request, messages=[*request.messages, *history.messages]),
        validators=(_retry_on_invalid_overfit_audit_output,),
        max_attempts=3,
    )
    if not snapshot.results:
        raise RetryLimitExceededError(last_retry_message=None)
    return _parse_overfit_audit_result(snapshot.results[-1].message.content or "")


async def _compute_candidate_overfit_audit_async(
    runtime: RuntimeConfig,
    *,
    request: ChatCompletionRequest,
) -> OverfitAuditResult:
    client = _new_chat_client(runtime)
    try:
        return await _run_overfit_audit_completion(client, request=request)
    finally:
        await _close_chat_client(client)


def _candidate_overfit_request(runtime: RuntimeConfig, candidate_text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=runtime.judge_model,
        messages=[
            ChatMessage(role=ROLE_SYSTEM, content=OVERFIT_AUDIT_SYSTEM_PROMPT),
            ChatMessage(role=ROLE_USER, content=_candidate_overfit_prompt(candidate_text)),
        ],
        response_format=ChatResponseFormat(type="json_object"),
        reasoning_effort=runtime.judge_reasoning_effort,
        max_completion_tokens=min(runtime.judge_max_completion_tokens, DEFAULT_CANDIDATE_OVERFIT_MAX_COMPLETION_TOKENS),
        temperature=0,
    )


async def _get_candidate_overfit_audit_async(
    runtime: RuntimeConfig,
    *,
    candidate_text: str,
) -> OverfitAuditResult:
    request = _candidate_overfit_request(runtime, candidate_text)
    cache_key = _request_cache_key(request)
    cached = _load_cached_candidate_overfit(runtime, cache_key)
    if cached is not None:
        return cached

    with runtime.candidate_overfit_lock:
        cached = runtime.candidate_overfit_results.get(cache_key)
        if cached is not None:
            return cached
        future = runtime.candidate_overfit_futures.get(cache_key)
        if future is None:
            future = Future()
            runtime.candidate_overfit_futures[cache_key] = future
            owns_compute = True
        else:
            owns_compute = False

    if not owns_compute:
        return await asyncio.to_thread(future.result)

    try:
        cached = _load_cached_candidate_overfit(runtime, cache_key)
        if cached is not None:
            future.set_result(cached)
            return cached
        result = await _compute_candidate_overfit_audit_async(runtime, request=request)
        _store_cached_candidate_overfit(runtime, cache_key, result)
        future.set_result(result)
        return result
    except Exception as exc:
        result = OverfitAuditResult(
            overfit_score=0.0,
            feedback=f"candidate_overfit_audit_unavailable: {type(exc).__name__}: {exc}",
            available=False,
        )
        future.set_result(result)
    else:
        return result
    finally:
        with runtime.candidate_overfit_lock:
            runtime.candidate_overfit_futures.pop(cache_key, None)

    return result


async def _evaluate_example_async(
    runtime: RuntimeConfig,
    candidate_text: str,
    candidate: ParsedCandidate,
    example: SliceExample,
) -> EvalSummary:
    client = _new_chat_client(runtime)
    try:
        steps = _load_trial_steps(str(runtime.database_path), example.trial_name)
        intercepted_step = next(
            step for step in steps if step.get("source") == "agent" and _step_ordinal(step) == example.intercepted_step_ordinal
        )
        original_prefix_messages = _prefix_messages_before_intercept(steps, intercepted_step_ordinal=example.intercepted_step_ordinal)
        original_tail_messages = _original_tail_messages(steps, intercepted_step_ordinal=example.intercepted_step_ordinal)
        original_intercepted_turn = _assistant_message_from_step(intercepted_step, include_tool_calls=True)
        stage_context_messages = _intercepted_stage_context_messages(
            steps,
            intercepted_step_ordinal=example.intercepted_step_ordinal,
            tool_call_stub=candidate.tool_call_stub,
        )
        stage_messages = stage_context_messages[-2:]

        stage_outputs: list[ChatMessage] = []
        for index, stage_prompt in enumerate(candidate.stages):
            rendered_stage_prompt = _render_stage_prompt(
                stage_prompt,
                original_reasoning_content=intercepted_step.get("reasoning_content"),
                is_final_stage=index == len(candidate.stages) - 1,
            )
            stage_user = ChatMessage(role=ROLE_USER, content=rendered_stage_prompt)
            stage_context_messages.append(stage_user)
            stage_messages.append(stage_user)
            stage_output = await _run_stage_completion(
                client,
                model=runtime.task_model,
                messages=stage_context_messages,
                reasoning_effort=runtime.task_reasoning_effort,
                max_completion_tokens=runtime.stage_max_completion_tokens,
            )
            stage_context_messages.append(stage_output)
            stage_messages.append(stage_output)
            stage_outputs.append(stage_output)

        if not stage_outputs:
            raise CandidateExecutionError("candidate produced no stage outputs")
        reasoning_extension = ((stage_outputs[-1].content or "") or (stage_outputs[-1].reasoning_content or "")).strip()
        if not reasoning_extension:
            raise CandidateExecutionError("final stage produced an empty reasoning extension")

        replay_message_history = [
            *original_prefix_messages,
            _assistant_message_from_step(
                intercepted_step,
                include_tool_calls=False,
                reasoning_content=_append_reasoning_extension(intercepted_step.get("reasoning_content"), reasoning_extension),
            ),
        ]
        replay_result_message = await _run_replay_completion(
            client,
            model=runtime.task_model,
            messages=replay_message_history,
            reasoning_effort=runtime.task_reasoning_effort,
            max_completion_tokens=runtime.replay_max_completion_tokens,
        )
        if not replay_result_message.tool_calls:
            raise CandidateExecutionError("replay did not produce a replacement tool call")
        synthetic_replacement_turn = _synthetic_replacement_turn(
            intercepted_step,
            reasoning_extension=reasoning_extension,
            replay_result_message=replay_result_message,
        )

        reference_messages = None
        if example.reference_pass_trial_name is not None:
            reference_steps = _load_trial_steps(str(runtime.database_path), example.reference_pass_trial_name)
            reference_messages = _full_trajectory_messages(reference_steps)

        judge_prompt = _judge_prompt(
            evaluation_mode=example.evaluation_mode,
            original_prefix_messages=original_prefix_messages,
            original_intercepted_turn=original_intercepted_turn,
            synthetic_replacement_turn=synthetic_replacement_turn,
            original_tail_messages=original_tail_messages,
            verifier_stdout_text=example.verifier_stdout_text,
            issue_explanation=example.issue_explanation,
            reference_messages=reference_messages,
            reference_verifier_stdout_text=example.reference_pass_verifier_stdout_text,
        )
        judge_result = await _run_judge_completion(
            client,
            model=runtime.judge_model,
            prompt=judge_prompt,
            reasoning_effort=runtime.judge_reasoning_effort,
            max_completion_tokens=runtime.judge_max_completion_tokens,
        )
        overfit_audit = await _get_candidate_overfit_audit_async(runtime, candidate_text=candidate_text)
        _log_evaluation_trace_result(
            runtime,
            candidate_text=candidate_text,
            example=example,
            original_intercepted_turn=original_intercepted_turn,
            stage_messages=stage_messages,
            synthetic_replacement_turn=synthetic_replacement_turn,
            judge_result=judge_result,
            overfit_audit=overfit_audit,
        )
        candidate_specific_info: dict[str, Any] = {"genericity_feedback": overfit_audit.feedback}
        if overfit_audit.available:
            candidate_specific_info["scores"] = {"genericity": 1.0 - overfit_audit.overfit_score}

        side_info = {"evaluation_mode": example.evaluation_mode}
        if example.issue_explanation is not None:
            side_info["issue_explanation"] = example.issue_explanation
        side_info.update(
            {
                "original_prefix_messages": _serialize_messages(original_prefix_messages),
                "interrupted_stage_messages": _serialize_messages(stage_messages),
                "synthetic_replacement_turn": _serialize_message(synthetic_replacement_turn),
                "judge_score": judge_result.overall_score,
                "judge_feedback": judge_result.feedback,
                f"{CURRENT_CANDIDATE_PARAMETER_NAME}_specific_info": candidate_specific_info,
            }
        )
        return EvalSummary(score=judge_result.overall_score, side_info=side_info)
    finally:
        await _close_chat_client(client)


def _evaluate_example(candidate_text: str, *, example: SliceExample, runtime: RuntimeConfig) -> tuple[float, dict[str, Any]]:
    try:
        candidate = _parse_candidate(candidate_text, max_stages=runtime.max_stages)
    except Exception as exc:
        _log_evaluation_trace_error(
            runtime,
            candidate_text=candidate_text,
            example=example,
            error=f"candidate_parse_error: {exc}",
            candidate_prefix=candidate_text[:1200],
        )
        return 0.0, {
            "error": f"candidate_parse_error: {exc}",
            "candidate_prefix": candidate_text[:1200],
            "task_name": example.task_name,
            "trial_name": example.trial_name,
            "slice_kind": example.slice_kind,
        }
    try:
        summary = asyncio.run(_evaluate_example_async(runtime, candidate_text, candidate, example))
    except CandidateExecutionError as exc:
        _log_evaluation_trace_error(
            runtime,
            candidate_text=candidate_text,
            example=example,
            error=f"candidate_execution_error: {exc}",
        )
        return 0.0, {
            "error": f"candidate_execution_error: {exc}",
            "task_name": example.task_name,
            "trial_name": example.trial_name,
            "slice_kind": example.slice_kind,
        }
    return summary.score, summary.side_info


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _evaluate_partition(
    candidate_text: str,
    examples: list[SliceExample],
    *,
    runtime: RuntimeConfig,
    max_workers: int,
) -> list[tuple[SliceExample, float, dict[str, Any]]]:
    if not examples:
        return []

    def run_one(example: SliceExample) -> tuple[SliceExample, float, dict[str, Any]]:
        score, side_info = _evaluate_example(candidate_text, example=example, runtime=runtime)
        return example, score, side_info

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(run_one, examples))


def _print_partition_summary(name: str, rows: list[tuple[SliceExample, float, dict[str, Any]]]) -> None:
    print()
    print(f"{name} summary")
    if not rows:
        print("  no examples")
        return
    scores = [score for _, score, _ in rows]
    print(f"  examples: {len(rows)}")
    print(f"  mean_score: {_mean(scores):.3f}")
    by_kind: dict[str, list[float]] = defaultdict(list)
    for example, score, _ in rows:
        by_kind[example.slice_kind].append(score)
    for kind, kind_scores in sorted(by_kind.items()):
        print(f"  {kind}: count={len(kind_scores)} mean_score={_mean(kind_scores):.3f}")


def _print_dataset_stats(label: str, examples: list[SliceExample]) -> None:
    stats = _dataset_stats(examples)
    print(f"{label}: tasks={stats['tasks']} examples={stats['examples']} by_kind={stats['by_kind']}")


def _load_seed_candidate(path: Path | None) -> str:
    if path is None:
        return SEED_CANDIDATE
    return path.read_text()


def _resolved_codex_lb_base_url(value: str | None) -> str:
    if value is not None and value.strip():
        return value.strip()
    env_value = os.environ.get("PLAP_CODEX_LB_BASE_URL")
    if env_value is not None and env_value.strip():
        return env_value.strip()
    return DEFAULT_CODEX_LB_BASE_URL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GEPA over interrupted MIMO trajectories.")
    parser.add_argument("--database", default=DEFAULT_DATABASE, type=Path)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, type=Path)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path)
    parser.add_argument("--codex-lb-base-url")
    parser.add_argument("--task-model", default=DEFAULT_TASK_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--reflection-model", default=DEFAULT_REFLECTION_MODEL)
    parser.add_argument("--seed-candidate-file", type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--test-task-count", default=DEFAULT_TEST_TASK_COUNT, type=int)
    parser.add_argument("--max-stages", default=DEFAULT_MAX_STAGES, type=int)
    parser.add_argument("--max-metric-calls", default=DEFAULT_MAX_METRIC_CALLS, type=int)
    parser.add_argument("--reflection-minibatch-size", default=DEFAULT_REFLECTION_MINIBATCH_SIZE, type=int)
    parser.add_argument("--max-workers", default=DEFAULT_MAX_WORKERS, type=int)
    parser.add_argument("--task-reasoning-effort", default=DEFAULT_TASK_REASONING_EFFORT)
    parser.add_argument("--judge-reasoning-effort", default=DEFAULT_JUDGE_REASONING_EFFORT)
    parser.add_argument("--reflector-reasoning-effort", default=DEFAULT_REFLECTOR_REASONING_EFFORT)
    parser.add_argument("--stage-max-completion-tokens", default=DEFAULT_STAGE_MAX_COMPLETION_TOKENS, type=int)
    parser.add_argument("--replay-max-completion-tokens", default=DEFAULT_REPLAY_MAX_COMPLETION_TOKENS, type=int)
    parser.add_argument("--judge-max-completion-tokens", default=DEFAULT_JUDGE_MAX_COMPLETION_TOKENS, type=int)
    parser.add_argument("--include-escalation-slices", action="store_true")
    parser.add_argument("--stats-only", action="store_true")
    return parser.parse_args()


def _build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    _load_env_file(args.env_file.resolve())
    _populate_provider_aliases()
    stdlib_logging.getLogger("transformers").setLevel(stdlib_logging.ERROR)
    stdlib_logging.getLogger("huggingface_hub").setLevel(stdlib_logging.ERROR)
    accumulator_module._log_tool_call_repair = lambda *args, **kwargs: None
    settings = _settings_from_env()
    codex_lb_base_url = _resolved_codex_lb_base_url(args.codex_lb_base_url)
    _ensure_model_routable(args.task_model, settings, codex_lb_base_url=codex_lb_base_url)
    _ensure_model_routable(args.judge_model, settings, codex_lb_base_url=codex_lb_base_url)
    _ensure_model_routable(args.reflection_model, settings, codex_lb_base_url=codex_lb_base_url)
    run_dir = args.run_dir.resolve()
    return RuntimeConfig(
        settings=settings,
        database_path=args.database.resolve(),
        run_dir=run_dir,
        evaluation_trace_path=run_dir / DEFAULT_EVALUATION_TRACE_FILENAME,
        codex_lb_base_url=codex_lb_base_url,
        task_model=args.task_model,
        judge_model=args.judge_model,
        reflection_model=args.reflection_model,
        task_reasoning_effort=args.task_reasoning_effort,
        judge_reasoning_effort=args.judge_reasoning_effort,
        reflector_reasoning_effort=args.reflector_reasoning_effort,
        stage_max_completion_tokens=args.stage_max_completion_tokens,
        replay_max_completion_tokens=args.replay_max_completion_tokens,
        judge_max_completion_tokens=args.judge_max_completion_tokens,
        max_stages=args.max_stages,
    )


def _build_splits(
    args: argparse.Namespace,
    *,
    runtime: RuntimeConfig,
) -> tuple[list[SliceExample], list[SliceExample], list[SliceExample], list[JudgeOversizeExclusion], int]:
    con = duckdb.connect(str(args.database.resolve()), read_only=True)
    examples = _build_slice_examples(con, include_escalation_slices=args.include_escalation_slices)
    no_issue_examples = [example for example in examples if example.evaluation_mode == "preserve"]
    recover_examples = [example for example in examples if example.evaluation_mode == "recover"]
    filtered_no_issue_examples, no_issue_exclusions = _filter_judge_oversize_examples(no_issue_examples, runtime=runtime)
    filtered_recover_examples, recover_exclusions, oversized_but_replaced = _select_feasible_recover_examples(
        recover_examples,
        runtime=runtime,
    )
    filtered_examples = [*filtered_no_issue_examples, *filtered_recover_examples]
    exclusions = [*no_issue_exclusions, *recover_exclusions]
    task_to_examples = _task_to_examples(filtered_examples)
    train_tasks, val_tasks, test_tasks = _partition_task_names(
        task_to_examples,
        seed=args.seed,
        test_task_count=args.test_task_count,
    )
    train_examples = _filter_examples_for_tasks(filtered_examples, train_tasks)
    val_examples = _filter_examples_for_tasks(filtered_examples, val_tasks)
    test_examples = _filter_examples_for_tasks(filtered_examples, test_tasks)
    return train_examples, val_examples, test_examples, exclusions, oversized_but_replaced


def _print_split_stats(train_examples: list[SliceExample], val_examples: list[SliceExample], test_examples: list[SliceExample]) -> None:
    _print_dataset_stats("train", train_examples)
    _print_dataset_stats("val", val_examples)
    _print_dataset_stats("test", test_examples)


def _judge_stable_prompt_tokens(runtime: RuntimeConfig, example: SliceExample) -> int:
    steps = _load_trial_steps(str(runtime.database_path), example.trial_name)
    intercepted_step = next(
        step for step in steps if step.get("source") == "agent" and _step_ordinal(step) == example.intercepted_step_ordinal
    )
    original_prefix_messages = _prefix_messages_before_intercept(steps, intercepted_step_ordinal=example.intercepted_step_ordinal)
    original_tail_messages = _original_tail_messages(steps, intercepted_step_ordinal=example.intercepted_step_ordinal)
    original_intercepted_turn = _assistant_message_from_step(intercepted_step, include_tool_calls=True)
    reference_messages = None
    if example.reference_pass_trial_name is not None:
        reference_steps = _load_trial_steps(str(runtime.database_path), example.reference_pass_trial_name)
        reference_messages = _full_trajectory_messages(reference_steps)
    prompt = _judge_prompt(
        evaluation_mode=example.evaluation_mode,
        original_prefix_messages=original_prefix_messages,
        original_intercepted_turn=original_intercepted_turn,
        synthetic_replacement_turn=None,
        original_tail_messages=original_tail_messages,
        verifier_stdout_text=example.verifier_stdout_text,
        issue_explanation=example.issue_explanation,
        reference_messages=reference_messages,
        reference_verifier_stdout_text=example.reference_pass_verifier_stdout_text,
    )
    request = ChatCompletionRequest(
        model=runtime.judge_model,
        messages=[
            ChatMessage(role=ROLE_SYSTEM, content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role=ROLE_USER, content=prompt),
        ],
        response_format=ChatResponseFormat(type="json_object"),
        reasoning_effort=runtime.judge_reasoning_effort,
        max_completion_tokens=runtime.judge_max_completion_tokens,
        temperature=0,
    )
    cache_key = _judge_token_cache_key(request)
    cached = _load_cached_judge_token_count(runtime, cache_key)
    if cached is not None:
        return cached
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Token indices sequence length is longer than the specified maximum sequence length.*",
        )
        prompt_tokens = measure_request_tokens(request, tokenizer_config=_judge_tokenizer_config(runtime.judge_model))
    _store_cached_judge_token_count(runtime, cache_key, prompt_tokens)
    return prompt_tokens


def _filter_judge_oversize_examples(
    examples: list[SliceExample],
    *,
    runtime: RuntimeConfig,
) -> tuple[list[SliceExample], list[JudgeOversizeExclusion]]:
    kept: list[SliceExample] = []
    excluded: list[JudgeOversizeExclusion] = []
    for example in examples:
        prompt_tokens = _judge_stable_prompt_tokens(runtime, example)
        if prompt_tokens <= DEFAULT_JUDGE_STABLE_PROMPT_TOKEN_BUDGET:
            kept.append(example)
            continue
        excluded.append(
            JudgeOversizeExclusion(
                example_id=example.example_id,
                task_name=example.task_name,
                trial_name=example.trial_name,
                slice_kind=example.slice_kind,
                prompt_tokens=prompt_tokens,
            )
        )
    return kept, excluded


def _select_feasible_recover_examples(
    examples: list[SliceExample],
    *,
    runtime: RuntimeConfig,
) -> tuple[list[SliceExample], list[JudgeOversizeExclusion], int]:
    by_task: dict[str, list[SliceExample]] = defaultdict(list)
    for example in examples:
        by_task[example.task_name].append(example)

    kept: list[SliceExample] = []
    exclusions: list[JudgeOversizeExclusion] = []
    oversized_but_replaced = 0

    for task_name, task_examples in by_task.items():
        measured: list[tuple[SliceExample, int]] = [(example, _judge_stable_prompt_tokens(runtime, example)) for example in task_examples]
        feasible = [item for item in measured if item[1] <= DEFAULT_JUDGE_STABLE_PROMPT_TOKEN_BUDGET]
        if feasible:
            kept.append(min(feasible, key=lambda item: (_recover_candidate_rank(item[0]), item[1]))[0])
            oversized_but_replaced += sum(1 for _, prompt_tokens in measured if prompt_tokens > DEFAULT_JUDGE_STABLE_PROMPT_TOKEN_BUDGET)
            continue
        example, prompt_tokens = min(measured, key=lambda item: (_recover_candidate_rank(item[0]), item[1]))
        exclusions.append(
            JudgeOversizeExclusion(
                example_id=example.example_id,
                task_name=task_name,
                trial_name=example.trial_name,
                slice_kind=example.slice_kind,
                prompt_tokens=prompt_tokens,
            )
        )

    return kept, exclusions, oversized_but_replaced


def _print_judge_oversize_summary(exclusions: list[JudgeOversizeExclusion]) -> None:
    if not exclusions:
        print("judge_oversize_exclusions: 0")
        return
    by_kind: dict[str, int] = defaultdict(int)
    by_task: set[str] = set()
    for item in exclusions:
        by_kind[item.slice_kind] += 1
        by_task.add(item.task_name)
    print(f"judge_oversize_exclusions: {len(exclusions)} examples across {len(by_task)} tasks by_kind={dict(sorted(by_kind.items()))}")
    print("largest_judge_exclusions:")
    for item in sorted(exclusions, key=lambda row: row.prompt_tokens, reverse=True)[:10]:
        print(
            f"  {item.example_id} task={item.task_name} trial={item.trial_name} kind={item.slice_kind} prompt_tokens={item.prompt_tokens}"
        )


def _print_replaced_recover_summary(count: int) -> None:
    print(f"judge_oversize_recover_candidates_replaced: {count}")


def _run_gepa(
    args: argparse.Namespace,
    *,
    runtime: RuntimeConfig,
    train_examples: list[SliceExample],
    val_examples: list[SliceExample],
    seed_candidate: str,
):
    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(args.run_dir.resolve()),
            seed=args.seed,
            max_metric_calls=args.max_metric_calls,
            candidate_selection_strategy="top_k_pareto",
            parallel=True,
            max_workers=args.max_workers,
            cache_evaluation=True,
            use_cloudpickle=True,
            track_best_outputs=True,
        ),
        reflection=ReflectionConfig(
            reflection_lm=_make_reflection_lm(runtime),
            batch_sampler=_BalancedIssueNoIssueBatchSampler(
                args.reflection_minibatch_size,
                rng=random.Random(args.seed),
            ),
            reflection_minibatch_size=args.reflection_minibatch_size,
            reflection_prompt_template=GEPA_REFLECTION_PROMPT_TEMPLATE,
            skip_perfect_score=False,
        ),
        merge=None,
    )
    return oa.optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=lambda candidate, example, **_: _evaluate_example(candidate, example=example, runtime=runtime),
        dataset=train_examples,
        valset=val_examples,
        config=config,
    )


def _print_best_candidate(candidate: object) -> None:
    print()
    print("Best candidate")
    print("```")
    if isinstance(candidate, str):
        print(candidate)
    else:
        print(candidate)
    print("```")


def main() -> int:
    args = _parse_args()
    if args.test_task_count < 0:
        raise SystemExit("--test-task-count must be non-negative")
    if args.max_stages <= 0:
        raise SystemExit("--max-stages must be positive")
    if args.max_metric_calls <= 0:
        raise SystemExit("--max-metric-calls must be positive")
    if args.reflection_minibatch_size <= 0:
        raise SystemExit("--reflection-minibatch-size must be positive")
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")

    runtime = _build_runtime(args)
    train_examples, val_examples, test_examples, judge_exclusions, oversized_but_replaced = _build_splits(
        args,
        runtime=runtime,
    )
    _print_judge_oversize_summary(judge_exclusions)
    _print_replaced_recover_summary(oversized_but_replaced)
    _print_split_stats(train_examples, val_examples, test_examples)
    if args.stats_only:
        return 0

    seed_candidate = _load_seed_candidate(args.seed_candidate_file.resolve() if args.seed_candidate_file is not None else None)
    result = _run_gepa(
        args,
        runtime=runtime,
        train_examples=train_examples,
        val_examples=val_examples,
        seed_candidate=seed_candidate,
    )
    _print_best_candidate(result.best_candidate)

    best_candidate_text = result.best_candidate if isinstance(result.best_candidate, str) else str(result.best_candidate)
    val_rows = _evaluate_partition(best_candidate_text, val_examples, runtime=runtime, max_workers=args.max_workers)
    test_rows = _evaluate_partition(best_candidate_text, test_examples, runtime=runtime, max_workers=args.max_workers)
    _print_partition_summary("validation", val_rows)
    _print_partition_summary("test", test_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
