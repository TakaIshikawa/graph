"""CSV export for source metadata completeness."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, write_csv

_BASE_FIELDS = ["source_project", "unit_count", "units_with_metadata", "metadata_coverage_ratio", "unique_key_count"]
_UNKNOWN_SOURCE = "Unknown"


def export_source_metadata_completeness_csv(
    units: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
    required_keys: Iterable[str] | None = None,
) -> str | dict[str, Any]:
    unit_list = list(units)
    required = [field_value(key) for key in (required_keys or []) if field_value(key)]
    rows = _rows(unit_list, required)
    fields = _BASE_FIELDS + (["missing_required_keys", "required_key_coverage_ratio"] if required else [])
    text = render_csv(rows, fields)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(units: list[Mapping[str, Any] | object], required: list[str]) -> list[dict[str, str | int]]:
    groups: dict[str, list[Mapping[str, Any] | object]] = defaultdict(list)
    for unit in units:
        groups[_source(unit)].append(unit)
    rows = []
    for source in sorted(groups, key=sort_key):
        source_units = groups[source]
        key_counts: Counter[str] = Counter()
        units_with_metadata = 0
        present_required: set[str] = set()
        for unit in source_units:
            keys = {field_value(key) for key in metadata(unit) if field_value(key)}
            if keys:
                units_with_metadata += 1
            key_counts.update(keys)
            present_required.update(key for key in required if key in keys)
        unit_count = len(source_units)
        row: dict[str, str | int] = {
            "source_project": source,
            "unit_count": unit_count,
            "units_with_metadata": units_with_metadata,
            "metadata_coverage_ratio": f"{units_with_metadata / unit_count:.2f}",
            "unique_key_count": len(key_counts),
        }
        if required:
            missing = [key for key in required if key not in present_required]
            row["missing_required_keys"] = "; ".join(missing)
            row["required_key_coverage_ratio"] = f"{len(present_required) / len(required):.2f}"
        rows.append(row)
    return rows


def _source(unit: Mapping[str, Any] | object) -> str:
    return field_value(get(unit, "source_project")) or _UNKNOWN_SOURCE
