from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SummaryDelta:
    text: str
    index: int


@dataclass(frozen=True, slots=True)
class SummaryDone:
    index: int
