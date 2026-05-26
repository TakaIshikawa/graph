"""CSV export for YAML frontmatter keys in unit Markdown content."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "has_frontmatter", "key_count", "keys", "duplicate_key_count", "malformed_frontmatter"]


def export_units_to_frontmatter_key_audit_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    content = "" if get(unit, "content") is None else str(get(unit, "content"))
    has_frontmatter, malformed, keys = _keys(content)
    unique = sorted(set(keys), key=sort_key)
    counts = Counter(keys)
    return {
        "unit_id": unit_id(unit),
        "has_frontmatter": str(has_frontmatter).lower(),
        "key_count": len(unique),
        "keys": "; ".join(unique),
        "duplicate_key_count": sum(count - 1 for count in counts.values() if count > 1),
        "malformed_frontmatter": str(malformed).lower(),
    }


def _keys(content: str) -> tuple[bool, bool, list[str]]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return False, False, []
    keys: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return True, False, keys
        if ":" in line and not line.startswith((" ", "\t", "-")):
            keys.append(field_value(line.split(":", 1)[0]))
    return True, True, keys
