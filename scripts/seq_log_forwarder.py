#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True, slots=True)
class QueuedEvent:
    body: dict[str, Any]
    line_number: int


def _normalize_level(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized == "critical":
        normalized = "fatal"
    levels = {
        "trace": "Verbose",
        "debug": "Debug",
        "info": "Information",
        "information": "Information",
        "warn": "Warning",
        "warning": "Warning",
        "error": "Error",
        "fatal": "Fatal",
    }
    return levels.get(normalized)


def _timestamp(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())


def _message_template(event: dict[str, Any]) -> str:
    value = event.get("event")
    if isinstance(value, str) and value.strip():
        return value
    return "plap.log"


def _properties(event: dict[str, Any], *, log_file: Path, line_number: int) -> dict[str, Any]:
    properties = {key: value for key, value in event.items() if key not in {"level", "timestamp"}}
    properties.setdefault("PlapLogFile", str(log_file))
    properties.setdefault("PlapLogLine", line_number)
    return properties


def _queued_event(raw: dict[str, Any], *, log_file: Path, line_number: int) -> QueuedEvent:
    event = _properties(raw, log_file=log_file, line_number=line_number)
    event["@t"] = _timestamp(raw.get("timestamp"))
    event["@mt"] = _message_template(raw)
    level = _normalize_level(raw.get("level"))
    if level is not None:
        event["@l"] = level
    return QueuedEvent(body=event, line_number=line_number)


def _payload(events: list[QueuedEvent]) -> bytes:
    return b"".join(msg.encode("utf-8") + b"\n" for msg in (json.dumps(item.body, separators=(",", ":")) for item in events))


def _request(seq_url: str, events: list[QueuedEvent], *, timeout_seconds: float) -> None:
    request = urllib.request.Request(
        f"{seq_url.rstrip('/')}/ingest/clef",
        data=_payload(events),
        headers={"Content-Type": "application/vnd.serilog.clef"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"unexpected Seq ingestion status: {response.status}")


def _warn_bad_event(exc: BaseException, event: QueuedEvent) -> None:
    print(
        f"Skipping log line {event.line_number}: Seq rejected the event ({exc}).",
        file=sys.stderr,
        flush=True,
    )


def _send_events(seq_url: str, events: list[QueuedEvent], *, timeout_seconds: float) -> None:
    if not events:
        return
    try:
        _request(seq_url, events, timeout_seconds=timeout_seconds)
        return
    except urllib.error.HTTPError as exc:
        if exc.code not in {400, 413}:
            raise
        if len(events) == 1:
            _warn_bad_event(exc, events[0])
            return
    midpoint = len(events) // 2
    _send_events(seq_url, events[:midpoint], timeout_seconds=timeout_seconds)
    _send_events(seq_url, events[midpoint:], timeout_seconds=timeout_seconds)


def _flush_pending(seq_url: str, pending: list[QueuedEvent]) -> None:
    _send_events(seq_url, pending, timeout_seconds=10.0)
    pending.clear()


def _parse_line(raw_line: str, *, log_file: Path, line_number: int) -> QueuedEvent | None:
    line = raw_line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        print(f"Skipping invalid JSON log line {line_number}: {exc}", file=sys.stderr, flush=True)
        return None
    if not isinstance(payload, dict):
        print(f"Skipping non-object JSON log line {line_number}.", file=sys.stderr, flush=True)
        return None
    return _queued_event(payload, log_file=log_file, line_number=line_number)


def _open_log_file(log_file: Path) -> tuple[TextIO, int, int]:
    handle = log_file.open("r", encoding="utf-8")
    stat = log_file.stat()
    return handle, stat.st_ino, 0


def _reopen_if_rotated(log_file: Path, handle: TextIO, *, inode: int, position: int) -> tuple[TextIO, int, int, bool]:
    stat = log_file.stat()
    if stat.st_ino == inode and stat.st_size >= position:
        return handle, inode, position, False
    handle.close()
    reopened, new_inode, _ = _open_log_file(log_file)
    print("Log file was rotated or truncated; restarting import from the current file contents.", file=sys.stderr, flush=True)
    return reopened, new_inode, 0, True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate plap JSONL logs into CLEF and stream them into Seq.")
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--log-file", required=True, type=Path)
    parser.add_argument("--poll-interval-seconds", default=0.25, type=float)
    parser.add_argument("--seq-url", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.poll_interval_seconds <= 0:
        raise SystemExit("--poll-interval-seconds must be positive")

    log_file = args.log_file.resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch(exist_ok=True)

    handle, inode, line_number = _open_log_file(log_file)
    pending: list[QueuedEvent] = []

    try:
        while True:
            raw_line = handle.readline()
            if raw_line:
                line_number += 1
                parsed = _parse_line(raw_line, log_file=log_file, line_number=line_number)
                if parsed is not None:
                    pending.append(parsed)
                if len(pending) >= args.batch_size:
                    _flush_pending(args.seq_url, pending)
                continue

            if pending:
                _flush_pending(args.seq_url, pending)

            position = handle.tell()
            try:
                handle, inode, line_number, reopened = _reopen_if_rotated(
                    log_file,
                    handle,
                    inode=inode,
                    position=position,
                )
            except FileNotFoundError:
                time.sleep(args.poll_interval_seconds)
                continue

            if reopened:
                continue
            time.sleep(args.poll_interval_seconds)
    except KeyboardInterrupt:
        return 0
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
