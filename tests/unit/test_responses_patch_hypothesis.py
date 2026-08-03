from __future__ import annotations

import jsonpatch
import msgspec
from hypothesis import given, settings
from hypothesis import strategies as st

from plap.responses.ingest.patch import JSONValue, diff

_JSON_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5),
    max_leaves=20,
)


@settings(max_examples=300)
@given(source=_JSON_VALUES, target=_JSON_VALUES)
def test_diff_patch_reaches_arbitrary_json_target(source: JSONValue, target: JSONValue) -> None:
    patch = diff(source, target)

    result = jsonpatch.apply_patch(source, patch, in_place=False)

    assert msgspec.json.encode(result, order="deterministic") == msgspec.json.encode(target, order="deterministic")


@given(
    source=st.lists(st.integers(min_value=-(2**63), max_value=2**63 - 1), unique=True, max_size=8),
    data=st.data(),
)
def test_diff_uses_minimum_moves_for_unique_array_permutation(source: list[int], data: st.DataObject) -> None:
    target = list(data.draw(st.permutations(source)))

    patch = diff(source, target)

    source_positions = {value: index for index, value in enumerate(source)}
    target_order = [source_positions[value] for value in target]
    lengths: list[int] = []
    for index, value in enumerate(target_order):
        lengths.append(1 + max((lengths[before] for before in range(index) if target_order[before] < value), default=0))
    minimum_moves = len(source) - max(lengths, default=0)

    assert len(patch) == minimum_moves
    assert all(operation["op"] == "move" for operation in patch)
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target
