from __future__ import annotations

from pydantic import BaseModel, ConfigDict

type Metadata = dict[str, str]


def _validate_metadata(value: Metadata | None) -> Metadata | None:
    if value is None:
        return value
    if len(value) > 16:
        raise ValueError("metadata may contain at most 16 entries")
    for key, item in value.items():
        if len(key) > 64:
            raise ValueError("metadata keys may contain at most 64 characters")
        if len(item) > 512:
            raise ValueError("metadata values may contain at most 512 characters")
    return value


def _reject_unsupported_type_variants(
    value: object,
    *,
    allowed: set[str],
    label: str,
) -> object:
    if not isinstance(value, list):
        return value

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type is None:
            raise ValueError(f"Missing {label} type at index {index}")
        if item_type not in allowed:
            raise ValueError(
                f"Unsupported {label} variant '{item_type}' at index {index}"
            )

    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
