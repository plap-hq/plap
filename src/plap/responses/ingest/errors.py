from __future__ import annotations

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError


def reasoning_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_reasoning_replay",
            message="Reasoning replay data is invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def compaction_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_compaction_replay",
            message="Compaction replay data is invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def tool_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_tool_replay",
            message="Tool replay data is invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


__all__ = ["compaction_replay_error", "reasoning_replay_error", "tool_replay_error"]
