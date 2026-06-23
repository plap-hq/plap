#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from plap.llms.completions.chat import ChatCompletionRequest, ChatMessage, ChatStreamOptions  # noqa: E402
from plap.llms.completions.client import ChatCompletionClient  # noqa: E402
from plap.llms.completions.providers import build_openrouter_provider  # noqa: E402
from plap.llms.completions.tokens import estimate_text_tokens  # noqa: E402

DEFAULT_ENV_FILE = REPO_ROOT / "tests" / ".env"
DEFAULT_WARMUPS = 1
DEFAULT_ROUNDS = 5
DEFAULT_MAX_COMPLETION_TOKENS = 1024
SLEEP_BETWEEN_TRIALS_SECONDS = 1.0
TARGET = (
    "Distributed systems stay available by isolating failures, retrying carefully, and degrading gracefully when "
    "dependencies break. They use timeouts to avoid waiting forever, circuit breakers to stop cascading damage, and "
    "idempotent operations so retries do not duplicate work. Replication and leader election help services survive "
    "node loss, while health checks and load balancing shift traffic away from unhealthy instances. Good observability "
    "lets engineers detect partial failure quickly, understand blast radius, and restore stable behavior before small "
    "faults become outages."
)
PROMPT = f"Output the following paragraph exactly, with identical punctuation and spacing, and nothing else:\n\n{TARGET}"


@dataclass(frozen=True, slots=True)
class TrialResult:
    model: str
    round_index: int
    warmup: bool
    ttft_visible_seconds: float
    total_seconds: float
    visible_generation_seconds: float
    estimated_visible_tokens: int
    visible_tps: float
    exact_match: bool


@dataclass(frozen=True, slots=True)
class TrialFailure:
    model: str
    round_index: int
    warmup: bool
    error_type: str
    error_message: str


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


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


def _request(model: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=[ChatMessage(role="user", content=PROMPT)],
        reasoning_effort="low",
        max_completion_tokens=DEFAULT_MAX_COMPLETION_TOKENS,
        temperature=0,
        stream_options=ChatStreamOptions(include_usage=True),
    )


async def _run_trial(client: ChatCompletionClient, model: str, *, round_index: int, warmup: bool) -> TrialResult:
    content_parts: list[str] = []
    start = time.perf_counter()
    first_visible_content_at: float | None = None

    async for delta in client.stream(_request(model)):
        if delta.content_delta:
            if first_visible_content_at is None:
                first_visible_content_at = time.perf_counter()
            content_parts.append(delta.content_delta)

    end = time.perf_counter()
    if first_visible_content_at is None:
        raise RuntimeError("provider produced no visible content deltas")

    content = "".join(content_parts)
    estimated_visible_tokens = estimate_text_tokens(content)
    visible_generation_seconds = max(end - first_visible_content_at, 1e-9)
    return TrialResult(
        model=model,
        round_index=round_index,
        warmup=warmup,
        ttft_visible_seconds=first_visible_content_at - start,
        total_seconds=end - start,
        visible_generation_seconds=visible_generation_seconds,
        estimated_visible_tokens=estimated_visible_tokens,
        visible_tps=estimated_visible_tokens / visible_generation_seconds,
        exact_match=content == TARGET,
    )


async def _run_model(
    client: ChatCompletionClient,
    model: str,
    *,
    warmups: int,
    rounds: int,
) -> tuple[list[TrialResult], list[TrialFailure]]:
    results: list[TrialResult] = []
    failures: list[TrialFailure] = []
    total_rounds = warmups + rounds
    for round_index in range(total_rounds):
        warmup = round_index < warmups
        try:
            results.append(await _run_trial(client, model, round_index=round_index, warmup=warmup))
        except Exception as exc:
            failures.append(
                TrialFailure(
                    model=model,
                    round_index=round_index,
                    warmup=warmup,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
        await asyncio.sleep(SLEEP_BETWEEN_TRIALS_SECONDS)
    return results, failures


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def _print_table(rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row_index, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        print(line)
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def _print_failures(failures: list[TrialFailure]) -> None:
    if not failures:
        return
    print()
    print("Failures")
    for failure in failures:
        phase = "warmup" if failure.warmup else "measured"
        print(f"- {failure.model} round={failure.round_index} phase={phase} {failure.error_type}: {failure.error_message}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter-routed chat models for visible TTFT and TPS.")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE, type=Path)
    parser.add_argument("--warmups", default=DEFAULT_WARMUPS, type=int)
    parser.add_argument("--rounds", default=DEFAULT_ROUNDS, type=int)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")

    _load_env_file(args.env_file.resolve())
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set; put it in tests/.env or your shell before running this benchmark.")

    client = ChatCompletionClient(build_openrouter_provider(api_key=api_key))

    rows = [
        (
            "model",
            "ttft_median_s",
            "ttft_mean_s",
            "tps_median",
            "tps_mean",
            "total_median_s",
            "ok/rounds",
        )
    ]
    failures: list[TrialFailure] = []

    for model in args.model:
        results, model_failures = await _run_model(client, model, warmups=args.warmups, rounds=args.rounds)
        failures.extend(model_failures)
        measured = [item for item in results if not item.warmup]
        if not measured:
            rows.append((model, "ERR", "ERR", "ERR", "ERR", "ERR", f"0/{args.rounds}"))
            continue
        rows.append(
            (
                model,
                _format_number(statistics.median(item.ttft_visible_seconds for item in measured)),
                _format_number(statistics.mean(item.ttft_visible_seconds for item in measured)),
                _format_number(statistics.median(item.visible_tps for item in measured)),
                _format_number(statistics.mean(item.visible_tps for item in measured)),
                _format_number(statistics.median(item.total_seconds for item in measured)),
                f"{sum(1 for item in measured if item.exact_match)}/{len(measured)}",
            )
        )

    _print_table(rows)
    _print_failures(failures)
    return 0


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
