from __future__ import annotations

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError


def _replay_error(
    *,
    code: str,
    public_message: str,
    reason: str,
    private_message: str,
    cause: BaseException | None,
) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code=code,
            message=public_message,
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


def reasoning_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return _replay_error(
        code="invalid_reasoning_replay",
        public_message="Reasoning replay data is invalid.",
        reason=reason,
        private_message=private_message,
        cause=cause,
    )


def compaction_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return _replay_error(
        code="invalid_compaction_replay",
        public_message="Compaction replay data is invalid.",
        reason=reason,
        private_message=private_message,
        cause=cause,
    )


def tool_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return _replay_error(
        code="invalid_tool_replay",
        public_message="Tool replay data is invalid.",
        reason=reason,
        private_message=private_message,
        cause=cause,
    )


def input_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return _replay_error(
        code="invalid_input_replay",
        public_message="Input replay items are invalid.",
        reason=reason,
        private_message=private_message,
        cause=cause,
    )


__all__ = ["compaction_replay_error", "input_replay_error", "reasoning_replay_error", "tool_replay_error"]
