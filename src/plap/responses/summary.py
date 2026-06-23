from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SummaryDelta:
    text: str


@dataclass(frozen=True, slots=True)
class SummaryDone:
    pass
