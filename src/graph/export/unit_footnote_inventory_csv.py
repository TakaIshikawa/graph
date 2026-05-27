"""CSV export for Markdown footnotes in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "footnote_definition_count", "footnote_reference_count", "unresolved_reference_count", "unused_definition_count", "duplicate_definition_count"]
_DEF_RE = re.compile(r"^\s*\[\^([^\]]+)\]:")
_REF_RE = re.compile(r"\[\^([^\]]+)\]")


def export_units_to_footnote_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    definitions: list[str] = []
    references: list[str] = []
    for line in ("" if get(unit, "content") is None else str(get(unit, "content"))).splitlines():
        definition = _DEF_RE.match(line)
        if definition:
            definitions.append(_label(definition.group(1)))
            tail = line[definition.end() :]
            references.extend(_label(match.group(1)) for match in _REF_RE.finditer(tail))
            continue
        references.extend(_label(match.group(1)) for match in _REF_RE.finditer(line))
    def_counts = Counter(definitions)
    ref_counts = Counter(references)
    return {
        "unit_id": unit_id(unit),
        "footnote_definition_count": len(definitions),
        "footnote_reference_count": len(references),
        "unresolved_reference_count": sum(count for label, count in ref_counts.items() if label not in def_counts),
        "unused_definition_count": sum(1 for label in def_counts if label not in ref_counts),
        "duplicate_definition_count": sum(count - 1 for count in def_counts.values() if count > 1),
    }


def _label(value: object) -> str:
    return field_value(value).casefold()
