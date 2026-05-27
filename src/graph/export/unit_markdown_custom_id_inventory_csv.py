"""CSV export for Pandoc-style Markdown attribute blocks by unit."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "custom_id_count", "duplicate_custom_id_count", "class_attribute_count", "key_value_attribute_count", "custom_ids"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ATTR_RE = re.compile(r"\{(?P<body>[^{}\n]+)\}")


def export_units_to_markdown_custom_id_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    ids: list[str] = []
    class_count = 0
    key_value_count = 0
    in_fence = False
    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ATTR_RE.finditer(line):
            for token in match.group("body").split():
                token = field_value(token)
                if token.startswith("#") and len(token) > 1:
                    ids.append(token[1:])
                elif token.startswith(".") and len(token) > 1:
                    class_count += 1
                elif "=" in token and not token.startswith("="):
                    key_value_count += 1
    counts = Counter(ids)
    return {
        "unit_id": unit_id(unit),
        "custom_id_count": len(ids),
        "duplicate_custom_id_count": sum(count - 1 for count in counts.values() if count > 1),
        "class_attribute_count": class_count,
        "key_value_attribute_count": key_value_count,
        "custom_ids": "; ".join(sorted(counts, key=sort_key)),
    }
