"""CSV export for metadata value entropy by key."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["metadata_key", "unit_count", "non_empty_count", "distinct_value_count", "top_value", "top_value_count", "entropy_score", "concentration_level"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_metadata_entropy_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata value diversity by key."""
    unit_list = list(units)
    rows = _entropy_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _entropy_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    unit_counts: Counter[str] = Counter()
    non_empty_counts: Counter[str] = Counter()
    values: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        metadata = _metadata(unit)
        for key, value in metadata.items():
            key_text = _field_value(key)
            if not key_text:
                continue
            unit_counts[key_text] += 1
            items = [_field_value(item) for item in _value_items(value) if _field_value(item)]
            if items:
                non_empty_counts[key_text] += 1
            for text in items:
                values[key_text][text] += 1

    rows: list[dict[str, str | int]] = []
    for key in unit_counts:
        counts = values[key]
        top_value = ""
        top_count = 0
        if counts:
            top_value, top_count = sorted(counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))[0]
        entropy = _entropy(counts)
        rows.append(
            {
                "metadata_key": key,
                "unit_count": unit_counts[key],
                "non_empty_count": non_empty_counts[key],
                "distinct_value_count": len(counts),
                "top_value": top_value,
                "top_value_count": top_count,
                "entropy_score": f"{entropy:.2f}",
                "concentration_level": "empty" if not counts else "high" if len(counts) == 1 else "medium" if top_count / sum(counts.values()) >= 0.5 else "low",
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["metadata_key"]))


def _value_items(value: object) -> list[object]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, Mapping):
        return [_stable_mapping(value)]
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _value_items(entry)]
    return [value]


def _stable_mapping(value: Mapping[Any, Any]) -> str:
    return "{" + ", ".join(f"{_field_value(key)}={_field_value(item)}" for key, item in sorted(value.items(), key=lambda entry: _sort_key(entry[0]))) + "}"


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0 or len(counts) < 2:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
