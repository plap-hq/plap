import re
import unicodedata
from collections.abc import Iterable
from typing import Any

_CONTEXT_LENGTH_EXCEEDED_CODES = frozenset(
    {
        "context_length_exceeded",
        "max_context_length_exceeded",
        "context_window_exceeded",
        "maximum_context_length_exceeded",
        "prompt_too_long",
        "input_too_long",
        "too_many_tokens",
        "token_limit_exceeded",
        "tokens_exceeded",
        "max_tokens_exceeded",
        "max_model_len_exceeded",
        "max_sequence_length_exceeded",
        "sequence_length_exceeded",
        "input_length_exceeded",
        "prompt_length_exceeded",
    }
)

_CONTEXT_LENGTH_EXCEEDED_CODE_FIELDS = frozenset({"code", "error_code", "type", "error_type"})
_CONTEXT_LENGTH_EXCEEDED_MESSAGE_FIELDS = frozenset(
    {"message", "detail", "details", "description", "error", "reason", "body", "response"}
)
_CONTEXT_LENGTH_EXCEEDED_ATTRIBUTES = (
    "code",
    "error_code",
    "type",
    "error_type",
    "message",
    "detail",
    "details",
    "description",
    "error",
    "reason",
    "body",
    "response",
)
_CONTEXT_LENGTH_EXCEEDED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcontext[_\s-]?length[_\s-]?(?:exceeded|exceeds|too\s+large|too\s+long)\b",
        r"\bcontext[_\s-]?window[_\s-]?(?:exceeded|exceeds|too\s+large|too\s+long)\b",
        r"\b(?:maximum|max)\s+context\s+(?:length|window)\s+(?:exceeded|reached)\b",
        r"\b(?:exceeded|exceeds)\s+(?:the\s+)?(?:maximum|max)\s+context\s+(?:length|window)\b",
        r"\bprompt\s+(?:is\s+)?too\s+long\b",
        r"\binput\s+(?:is\s+)?too\s+long\b",
        r"\bmessages?\s+(?:are|is)?\s*too\s+long\b",
        r"\btoo\s+many\s+tokens\b",
        r"\btoken\s+limit\s+(?:exceeded|reached)\b",
        r"\b(?:requested|provided|input|prompt|total)\s+tokens?\s+(?:exceeds?|exceeded)\b",
        r"\binput\s+token\s+count\s+(?:exceeds?|exceeded)\b",
        r"\btotal\s+token\s+count\s+(?:exceeds?|exceeded)\b",
        r"\bsequence\s+length\s+(?:exceeds?|exceeded|is\s+longer)\b",
        r"\blonger\s+than\s+(?:the\s+)?(?:model'?s?\s+)?maximum\s+(?:context\s+length|sequence\s+length|length)\b",
        r"\bmodel\s+(?:only\s+)?supports\s+up\s+to\s+\d+[\d,]*(?:\s+tokens?)?\b",
        r"\b(?:this|the)\s+model'?s?\s+maximum\s+(?:context\s+length|sequence\s+length|length)\s+is\s+\d+[\d,]*\b",
        r"\bmax[_\s-]?model[_\s-]?len\b.{0,80}\b(?:exceeded|exceeds|greater|larger|too\s+long)\b",
        r"\bmax[_\s-]?seq[_\s-]?len\b.{0,80}\b(?:exceeded|exceeds|greater|larger|too\s+long)\b",
        r"\bcontext\s+(?:is\s+)?full\b",
    )
)


def _normalize_context_length_value(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\x00", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def _safe_context_length_string(value: Any, *, max_len: int = 20_000) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        try:
            text = str(value)
        except Exception:
            return None
    text = text.strip()
    if not text:
        return None
    return text[:max_len]


def is_context_length_exceeded_code(value: str | None) -> bool:
    if value is None:
        return False
    normalized = _normalize_context_length_value(value).replace("-", "_").replace(" ", "_")
    return normalized in _CONTEXT_LENGTH_EXCEEDED_CODES


def is_context_length_exceeded_message(message: str | None) -> bool:
    if message is None:
        return False
    normalized = _normalize_context_length_value(message)
    if is_context_length_exceeded_code(normalized):
        return True
    return any(pattern.search(normalized) for pattern in _CONTEXT_LENGTH_EXCEEDED_PATTERNS)


def _iter_context_length_strings(error: Any, *, depth: int = 0, max_depth: int = 4) -> Iterable[str]:
    if error is None or depth > max_depth:
        return
    if isinstance(error, str | bytes):
        text = _safe_context_length_string(error)
        if text is not None:
            yield text
        return
    if isinstance(error, dict):
        for key, value in error.items():
            key_str = str(key)
            if key_str in _CONTEXT_LENGTH_EXCEEDED_CODE_FIELDS | _CONTEXT_LENGTH_EXCEEDED_MESSAGE_FIELDS:
                text = _safe_context_length_string(value)
                if text is not None:
                    yield text
            if isinstance(value, dict | list | tuple):
                yield from _iter_context_length_strings(value, depth=depth + 1, max_depth=max_depth)
        return
    if isinstance(error, list | tuple):
        for item in error[:20]:
            yield from _iter_context_length_strings(item, depth=depth + 1, max_depth=max_depth)
        return
    for attr in _CONTEXT_LENGTH_EXCEEDED_ATTRIBUTES:
        try:
            value = getattr(error, attr)
        except Exception:
            continue
        text = _safe_context_length_string(value)
        if text is not None:
            yield text
        if isinstance(value, dict | list | tuple):
            yield from _iter_context_length_strings(value, depth=depth + 1, max_depth=max_depth)
    text = _safe_context_length_string(error)
    if text is not None:
        yield text


def is_context_length_exceeded_error(error: Any) -> bool:
    seen: set[str] = set()
    for text in _iter_context_length_strings(error):
        normalized = _normalize_context_length_value(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if is_context_length_exceeded_code(normalized) or is_context_length_exceeded_message(normalized):
            return True
    return False


class ChatCompletionError(Exception):
    pass


class ChatCompletionProviderError(ChatCompletionError):
    pass


class ChatCompletionRateLimitError(ChatCompletionProviderError):
    pass


class ChatCompletionAuthenticationError(ChatCompletionProviderError):
    pass


class ChatCompletionInvalidRequestError(ChatCompletionProviderError):
    pass


class ChatCompletionContextLengthExceededError(ChatCompletionInvalidRequestError):
    pass


class ChatCompletionTimeoutError(ChatCompletionProviderError):
    pass


class ChatCompletionUnsupportedRequestError(ChatCompletionError):
    pass
