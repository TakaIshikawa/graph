"""CSV export for Markdown abbreviation definitions and acronym tokens."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "abbreviation_definition_count",
    "acronym_token_count",
    "undefined_acronym_count",
    "defined_acronyms",
]
_ABBR_RE = re.compile(r"^\s*\*\[(?P<abbr>[^\]]+)\]:\s*(?P<expansion>.*?)\s*$")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_COMMON_WORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BUT",
    "BY",
    "FOR",
    "FROM",
    "HAS",
    "HAVE",
    "IN",
    "IS",
    "IT",
    "NOT",
    "OF",
    "ON",
    "OR",
    "THE",
    "THIS",
    "TO",
    "WAS",
    "WE",
    "WERE",
    "WITH",
}


def export_units_to_abbreviation_inventory_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {
        "path": output_path,
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": bytes_written,
    }


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    definitions: dict[str, str] = {}
    definition_count = 0
    acronyms: list[str] = []
    in_fence = False

    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        definition = _ABBR_RE.match(line)
        if definition:
            definition_count += 1
            label = _normalize_acronym(definition.group("abbr"))
            if label:
                definitions.setdefault(label.casefold(), label)
            continue

        for token in _ACRONYM_RE.findall(line):
            label = _normalize_acronym(token)
            if not label or label in _COMMON_WORDS:
                continue
            acronyms.append(label)

    return {
        "unit_id": unit_id(unit),
        "abbreviation_definition_count": definition_count,
        "acronym_token_count": len(acronyms),
        "undefined_acronym_count": sum(1 for label in acronyms if label.casefold() not in definitions),
        "defined_acronyms": "; ".join(sorted(definitions.values(), key=sort_key)),
    }


def _normalize_acronym(value: str) -> str:
    text = field_value(value)
    if len(text) < 2:
        return ""
    return text.upper()
