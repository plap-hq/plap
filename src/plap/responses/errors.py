from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PrivateErrorLevel(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PublicResponseError:
    status_code: int
    type: str
    code: str
    message: str
    param: str | None = None
    headers: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class PrivateResponseError:
    event: str
    message: str
    level: PrivateErrorLevel = PrivateErrorLevel.WARNING
    context: dict[str, Any] = field(default_factory=dict)
    cause: BaseException | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", PrivateErrorLevel(self.level))

    def log(self, logger: Any, **context: object) -> None:
        log = logger.error if self.level == PrivateErrorLevel.ERROR else logger.warning
        log(
            self.event,
            exc_info=self.cause,
            private_message=self.message,
            **self.context,
            **context,
        )


class ResponseError(Exception):
    def __init__(self, public: PublicResponseError, private: PrivateResponseError) -> None:
        super().__init__(private.message)
        self.public = public
        self.private = private

    def log(self, logger: Any, **context: object) -> None:
        self.private.log(logger, **context)

    @classmethod
    def invalid_request(
        cls,
        *,
        public_message: str = "Invalid request.",
        private_message: str,
        public_code: str = "invalid_request",
        public_type: str = "invalid_request_error",
        param: str | None = None,
        headers: dict[str, str] | None = None,
        event: str = "response.invalid_request",
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> ResponseError:
        return cls(
            PublicResponseError(
                status_code=400,
                type=public_type,
                code=public_code,
                message=public_message,
                param=param,
                headers=headers,
            ),
            PrivateResponseError(
                event=event,
                message=private_message,
                level=PrivateErrorLevel.WARNING,
                context=dict(context or {}),
                cause=cause,
            ),
        )

    @classmethod
    def ingestion(
        cls,
        *,
        private_message: str,
        public_message: str = "Invalid request.",
        public_code: str = "invalid_request",
        event: str = "response.ingestion",
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> ResponseError:
        return cls.invalid_request(
            public_message=public_message,
            private_message=private_message,
            public_code=public_code,
            event=event,
            context=context,
            cause=cause,
        )

    @classmethod
    def tool_policy(
        cls,
        *,
        private_message: str,
        public_message: str = "Invalid request.",
        public_code: str = "invalid_tool",
        event: str = "response.tool_policy",
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> ResponseError:
        return cls.invalid_request(
            public_message=public_message,
            private_message=private_message,
            public_code=public_code,
            event=event,
            context=context,
            cause=cause,
        )

    @classmethod
    def unsupported_operation(
        cls,
        *,
        private_message: str,
        status_code: int = 501,
        public_message: str = "Operation is not supported.",
        event: str = "response.unsupported_operation",
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> ResponseError:
        return cls(
            PublicResponseError(
                status_code=status_code,
                type="invalid_request_error",
                code="unsupported_operation",
                message=public_message,
            ),
            PrivateResponseError(
                event=event,
                message=private_message,
                level=PrivateErrorLevel.WARNING,
                context=dict(context or {}),
                cause=cause,
            ),
        )

    @classmethod
    def unavailable(
        cls,
        *,
        private_message: str,
        public_message: str = "Response generation is temporarily unavailable.",
        public_code: str = "temporarily_unavailable",
        event: str = "response.unavailable",
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> ResponseError:
        return cls(
            PublicResponseError(
                status_code=503,
                type="server_error",
                code=public_code,
                message=public_message,
            ),
            PrivateResponseError(
                event=event,
                message=private_message,
                level=PrivateErrorLevel.WARNING,
                context=dict(context or {}),
                cause=cause,
            ),
        )

    @classmethod
    def internal(
        cls,
        *,
        private_message: str,
        public_message: str = "Response generation failed.",
        public_code: str = "server_error",
        event: str = "response.internal_error",
        context: Mapping[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> ResponseError:
        return cls(
            PublicResponseError(
                status_code=500,
                type="server_error",
                code=public_code,
                message=public_message,
            ),
            PrivateResponseError(
                event=event,
                message=private_message,
                level=PrivateErrorLevel.ERROR,
                context=dict(context or {}),
                cause=cause,
            ),
        )
