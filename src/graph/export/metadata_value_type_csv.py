"""CSV export for metadata value type profiles."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "metadata_key",
    "unit_count",
    "null_count",
    "string_count",
    "number_count",
    "boolean_count",
    "list_count",
    "object_count",
    "other_count",
    "distinct_scalar_values",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_metadata_value_type_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata value type counts by metadata key."""
    unit_list = list(units)
    rows = _profile_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "metadata_key_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _profile_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    profiles: dict[str, dict[str, Any]] = defaultdict(_empty_profile)

    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        seen_keys: set[str] = set()
        for raw_key, value in metadata.items():
            key = _inline_text(raw_key)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            profile = profiles[key]
            profile["unit_count"] += 1
            category = _value_category(value)
            profile[f"{category}_count"] += 1
            scalar_value = _scalar_value(value)
            if scalar_value is not None:
                profile["scalar_values"].add(scalar_value)

    rows: list[dict[str, str | int]] = []
    for metadata_key in sorted(profiles, key=_sort_key):
        profile = profiles[metadata_key]
        rows.append(
            {
                "metadata_key": metadata_key,
                "unit_count": profile["unit_count"],
                "null_count": profile["null_count"],
                "string_count": profile["string_count"],
                "number_count": profile["number_count"],
                "boolean_count": profile["boolean_count"],
                "list_count": profile["list_count"],
                "object_count": profile["object_count"],
                "other_count": profile["other_count"],
                "distinct_scalar_values": len(profile["scalar_values"]),
            }
        )
    return rows


def _empty_profile() -> dict[str, Any]:
    return {
        "unit_count": 0,
        "null_count": 0,
        "string_count": 0,
        "number_count": 0,
        "boolean_count": 0,
        "list_count": 0,
        "object_count": 0,
        "other_count": 0,
        "scalar_values": set(),
    }


def _value_category(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list | tuple):
        return "list"
    if isinstance(value, Mapping):
        return "object"
    return "other"


def _scalar_value(value: object) -> tuple[str, str] | None:
    if value is None:
        return ("null", "")
    if isinstance(value, bool):
        return ("boolean", "true" if value else "false")
    if isinstance(value, int | float):
        return ("number", str(value))
    if isinstance(value, str):
        return ("string", _inline_text(value))
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
