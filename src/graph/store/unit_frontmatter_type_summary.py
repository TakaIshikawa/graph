"""Summarize observed frontmatter value types by key."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import get, metadata, sort_key, unit_id

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-Z]+)?$")
_NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)$")


def summarize_unit_frontmatter_types(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    total_units = frontmatter_unit_count = 0
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for index, unit in enumerate(units):
        total_units += 1
        values = _frontmatter_values(unit)
        if not values:
            continue
        frontmatter_unit_count += 1
        uid = unit_id(unit) or str(index)
        for key, value in values.items():
            normalized_key = str(key).strip().casefold()
            value_type = _value_type(value)
            counts[normalized_key][value_type] += 1
            if len(samples[normalized_key][value_type]) < max(0, sample_limit):
                samples[normalized_key][value_type].append({"unit_id": uid, "value": value})

    type_counts_by_key = {
        key: [{"type": value_type, "count": counts[key][value_type]} for value_type in sorted(counts[key], key=sort_key)]
        for key in sorted(counts, key=sort_key)
    }
    mixed_type_keys = [key for key in sorted(counts, key=sort_key) if len(counts[key]) > 1]

    return {
        "unit_count": total_units,
        "frontmatter_unit_count": frontmatter_unit_count,
        "type_counts_by_key": type_counts_by_key,
        "mixed_type_keys": mixed_type_keys,
        "samples": {
            key: {
                value_type: sorted(samples[key][value_type], key=lambda row: sort_key(row["unit_id"]))[: max(0, sample_limit)]
                for value_type in sorted(samples[key], key=sort_key)
            }
            for key in sorted(samples, key=sort_key)
        },
    }


def _frontmatter_values(unit: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    meta = metadata(unit)
    frontmatter = meta.get("frontmatter")
    if isinstance(frontmatter, Mapping):
        values.update(frontmatter)
    values.update({key: value for key, value in meta.items() if key != "frontmatter"})
    values.update(_parse_yaml_frontmatter(_content(unit)))
    return values


def _parse_yaml_frontmatter(content: str) -> dict[str, Any]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    body: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            break
        body.append(line)
    else:
        return {}

    values: dict[str, Any] = {}
    index = 0
    while index < len(body):
        line = body[index]
        if not line.strip() or line.lstrip() != line:
            index += 1
            continue
        key, separator, raw = line.partition(":")
        if not separator:
            index += 1
            continue
        key = key.strip()
        raw = raw.strip()
        if raw == "":
            children = []
            next_index = index + 1
            while next_index < len(body) and body[next_index].startswith((" ", "\t")):
                child = body[next_index].strip()
                if child.startswith("- "):
                    children.append(_parse_scalar(child[2:].strip()))
                next_index += 1
            values[key] = children if children else None
            index = next_index
            continue
        values[key] = _parse_scalar(raw)
        index += 1
    return values


def _parse_scalar(value: str) -> Any:
    text = value.strip().strip("\"'")
    if text in {"", "null", "Null", "NULL", "~"}:
        return None
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    if _NUMBER_RE.match(text):
        return float(text) if "." in text else int(text)
    return text


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    if isinstance(value, str) and _DATE_RE.match(value.strip()):
        return "date-like string"
    return "scalar"


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content")
    return "" if value is None else str(value)
