from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ErrorLevel(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PublicError:
    status_code: int
    type: str
    code: str
    message: str
    param: str | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class PrivateError:
    event: str
    reason: str
    message: str | None = None
    level: ErrorLevel = ErrorLevel.WARNING
    context: dict[str, Any] = field(default_factory=dict)
    cause: BaseException | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", ErrorLevel(self.level))
        if self.message is None:
            object.__setattr__(self, "message", self.reason)

    def log(self, logger: Any, **context: object) -> None:
        log = logger.error if self.level == ErrorLevel.ERROR else logger.warning
        log(
            self.event,
            exc_info=self.cause,
            private_message=self.message,
            private_reason=self.reason,
            **self.context,
            **context,
        )


class PlapError(Exception):
    def __init__(self, *, public: PublicError | None, private: PrivateError) -> None:
        super().__init__(private.message)
        self.public = public
        self.private = private

    def log(self, logger: Any, **context: object) -> None:
        self.private.log(logger, **context)
