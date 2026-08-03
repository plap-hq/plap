from __future__ import annotations

from bisect import bisect_left
from copy import deepcopy

import msgspec

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
type JSONPatchOperation = dict[str, JSONValue]
type JSONPatch = list[JSONPatchOperation]


class _Differ:
    def __init__(self) -> None:
        self._operations: JSONPatch = []
        self._tokens: dict[int, bytes] = {}
        self._weights: dict[int, int] = {}
        self._distances: dict[tuple[int, int], int] = {}

    def _token(self, value: JSONValue) -> bytes:
        key = id(value)
        token = self._tokens.get(key)
        if token is None:
            token = msgspec.json.encode(value, order="deterministic")
            self._tokens[key] = token
        return token

    def _equal(self, source: JSONValue, target: JSONValue) -> bool:
        return self._token(source) == self._token(target)

    def _weight(self, value: JSONValue) -> int:
        key = id(value)
        cached = self._weights.get(key)
        if cached is not None:
            return cached
        if isinstance(value, list):
            weight = 1 + sum(self._weight(item) for item in value)
        elif isinstance(value, dict):
            weight = 1 + sum(self._weight(item) for item in value.values())
        else:
            weight = 1
        self._weights[key] = weight
        return weight

    def _distance(self, source: JSONValue, target: JSONValue) -> int:
        key = (id(source), id(target))
        cached = self._distances.get(key)
        if cached is not None:
            return cached
        if self._equal(source, target):
            distance = 0
        elif isinstance(source, dict) and isinstance(target, dict):
            source_keys = set(source)
            target_keys = set(target)
            distance = sum(self._weight(source[name]) for name in source_keys - target_keys)
            distance += sum(self._weight(target[name]) for name in target_keys - source_keys)
            distance += sum(self._distance(source[name], target[name]) for name in source_keys & target_keys)
        elif isinstance(source, list) and isinstance(target, list):
            previous = [0]
            for target_item in target:
                previous.append(previous[-1] + self._weight(target_item))
            for source_item in source:
                current = [previous[0] + self._weight(source_item)]
                for target_index, target_item in enumerate(target, start=1):
                    current.append(
                        min(
                            previous[target_index] + self._weight(source_item),
                            current[target_index - 1] + self._weight(target_item),
                            previous[target_index - 1] + self._distance(source_item, target_item),
                        )
                    )
                previous = current
            distance = previous[-1]
        elif type(source) is type(target):
            distance = 1
        else:
            distance = self._weight(source) + self._weight(target)
        self._distances[key] = distance
        return distance

    def _can_align(self, source: JSONValue, target: JSONValue) -> bool:
        if type(source) is not type(target):
            return False
        if not isinstance(source, dict | list):
            return True
        return self._distance(source, target) < min(self._weight(source), self._weight(target))

    def _exact_array_matches(self, source: list[JSONValue], target: list[JSONValue]) -> dict[int, int]:
        source_tokens = [self._token(item) for item in source]
        target_tokens = [self._token(item) for item in target]
        lengths = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]
        for source_index in range(len(source) - 1, -1, -1):
            for target_index in range(len(target) - 1, -1, -1):
                if source_tokens[source_index] == target_tokens[target_index]:
                    lengths[source_index][target_index] = lengths[source_index + 1][target_index + 1] + 1
                else:
                    lengths[source_index][target_index] = max(
                        lengths[source_index + 1][target_index], lengths[source_index][target_index + 1]
                    )

        matches: dict[int, int] = {}
        source_index = 0
        target_index = 0
        while source_index < len(source) and target_index < len(target):
            if source_tokens[source_index] == target_tokens[target_index]:
                matches[target_index] = source_index
                source_index += 1
                target_index += 1
            elif lengths[source_index + 1][target_index] >= lengths[source_index][target_index + 1]:
                source_index += 1
            else:
                target_index += 1
        return matches

    def _array_correspondence(self, source: list[JSONValue], target: list[JSONValue]) -> dict[int, int]:
        # Exact elements can carry replay-time values safely, so establish them before aligning edited elements.
        matches = self._exact_array_matches(source, target)
        used_source = set(matches.values())

        candidates: dict[bytes, list[int]] = {}
        for source_index, source_item in enumerate(source):
            if source_index not in used_source:
                candidates.setdefault(self._token(source_item), []).append(source_index)
        for target_index, target_item in enumerate(target):
            if target_index in matches:
                continue
            available = candidates.get(self._token(target_item), [])
            if not available:
                continue
            source_index = min(available, key=lambda index: (abs(index - target_index), -index))
            available.remove(source_index)
            matches[target_index] = source_index
            used_source.add(source_index)

        source_indices = [index for index in range(len(source)) if index not in used_source]
        target_indices = [index for index in range(len(target)) if index not in matches]
        source_count = len(source_indices)
        target_count = len(target_indices)
        costs = [[0] * (target_count + 1) for _ in range(source_count + 1)]

        for source_position in range(source_count - 1, -1, -1):
            costs[source_position][target_count] = costs[source_position + 1][target_count] + self._weight(
                source[source_indices[source_position]]
            )
        for target_position in range(target_count - 1, -1, -1):
            costs[source_count][target_position] = costs[source_count][target_position + 1] + self._weight(
                target[target_indices[target_position]]
            )
        for source_position in range(source_count - 1, -1, -1):
            for target_position in range(target_count - 1, -1, -1):
                source_item = source[source_indices[source_position]]
                target_item = target[target_indices[target_position]]
                options = [
                    costs[source_position + 1][target_position] + self._weight(source_item),
                    costs[source_position][target_position + 1] + self._weight(target_item),
                ]
                if self._can_align(source_item, target_item):
                    options.append(costs[source_position + 1][target_position + 1] + self._distance(source_item, target_item))
                costs[source_position][target_position] = min(options)

        source_position = 0
        target_position = 0
        while source_position < source_count and target_position < target_count:
            source_index = source_indices[source_position]
            target_index = target_indices[target_position]
            source_item = source[source_index]
            target_item = target[target_index]
            best = costs[source_position][target_position]
            delete = costs[source_position + 1][target_position] + self._weight(source_item)
            add = costs[source_position][target_position + 1] + self._weight(target_item)
            align = None
            if self._can_align(source_item, target_item):
                align = costs[source_position + 1][target_position + 1] + self._distance(source_item, target_item)

            source_remaining = source_count - source_position
            target_remaining = target_count - target_position
            # On an equal-cost contraction, discard the surplus source before choosing the edited correspondence.
            if source_remaining > target_remaining and delete == best:
                source_position += 1
            elif target_remaining > source_remaining and add == best:
                target_position += 1
            elif align == best:
                matches[target_index] = source_index
                source_position += 1
                target_position += 1
            elif delete == best:
                source_position += 1
            else:
                target_position += 1

        return matches

    @staticmethod
    def _path(path: str, part: str | int) -> str:
        encoded = str(part).replace("~", "~0").replace("/", "~1")
        return f"{path}/{encoded}"

    @staticmethod
    def _longest_increasing_positions(values: list[int]) -> set[int]:
        if not values:
            return set()

        tails: list[int] = []
        tail_positions: list[int] = []
        previous = [-1] * len(values)
        for position, value in enumerate(values):
            insertion = bisect_left(tails, value)
            if insertion == len(tails):
                tails.append(value)
                tail_positions.append(position)
            else:
                tails[insertion] = value
                tail_positions[insertion] = position
            if insertion:
                previous[position] = tail_positions[insertion - 1]

        positions: set[int] = set()
        position = tail_positions[-1]
        while position >= 0:
            positions.add(position)
            position = previous[position]
        return positions

    def _reorder_array(self, path: str, current: list[int], target: list[int]) -> None:
        stationary_positions = self._longest_increasing_positions(target)
        stationary = {target[position] for position in stationary_positions}

        # Moving non-LIS elements from right to left preserves every stationary element's relative position.
        for target_position in range(len(target) - 1, -1, -1):
            marker = target[target_position]
            if marker in stationary:
                continue
            source_position = current.index(marker)
            if target_position == len(target) - 1:
                destination = len(current) - 1
            else:
                next_position = current.index(target[target_position + 1])
                destination = next_position - (source_position < next_position)
            if source_position == destination:
                continue
            self._operations.append(
                {
                    "op": "move",
                    "from": self._path(path, source_position),
                    "path": self._path(path, destination),
                }
            )
            current.insert(destination, current.pop(source_position))

        if current != target:  # pragma: no cover - guards the move planner's invariant
            raise RuntimeError("array move planning did not reach its target order")

    def _append(self, path: str, source: JSONValue, target: JSONValue) -> None:
        if self._equal(source, target):
            return
        if isinstance(source, dict) and isinstance(target, dict):
            source_keys = set(source)
            target_keys = set(target)
            for name in sorted(source_keys - target_keys):
                self._operations.append({"op": "remove", "path": self._path(path, name)})
            for name in sorted(target_keys - source_keys):
                self._operations.append({"op": "add", "path": self._path(path, name), "value": deepcopy(target[name])})
            for name in sorted(source_keys & target_keys):
                self._append(self._path(path, name), source[name], target[name])
            return
        if isinstance(source, list) and isinstance(target, list):
            matches = self._array_correspondence(source, target)
            matched_source = set(matches.values())
            current = list(range(len(source)))

            for index in range(len(current) - 1, -1, -1):
                if current[index] in matched_source:
                    continue
                self._operations.append({"op": "remove", "path": self._path(path, index)})
                current.pop(index)

            matched_target_indices = sorted(matches)
            target_order = [matches[index] for index in matched_target_indices]
            self._reorder_array(path, current, target_order)

            for target_index, target_item in enumerate(target):
                if target_index in matches:
                    continue
                self._operations.append({"op": "add", "path": self._path(path, target_index), "value": deepcopy(target_item)})

            for target_index, source_index in sorted(matches.items()):
                self._append(self._path(path, target_index), source[source_index], target[target_index])
            return
        self._operations.append({"op": "replace", "path": path, "value": deepcopy(target)})

    def __call__(self, source: JSONValue, target: JSONValue) -> JSONPatch:
        self._append("", source, target)
        return self._operations


def diff(source: JSONValue, target: JSONValue) -> JSONPatch:
    return _Differ()(source, target)
