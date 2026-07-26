"""Finite-value validation and canonical formal-result serialization."""

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
import math


class FormalSchemaError(ValueError):
    pass


def to_primitive(value, path="root"):
    if is_dataclass(value):
        return to_primitive(asdict(value), path)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key.value if isinstance(key, Enum) else key): to_primitive(
                item, f"{path}.{key}"
            )
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        raise FormalSchemaError(f"{path}: formal numeric value must be finite")
    return value


def to_canonical_json(value) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
