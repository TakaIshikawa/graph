"""CSV export for unit metadata key coverage."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["metadata_key", "unit_count", "coverage_percent", "sample_unit_ids", "value_type_mix"]
_SAMPLE_LIMIT = 5


def export_units_to_metadata_key_coverage_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    total_units = len(unit_list)
    key_counts: Counter[str] = Counter()
    samples: dict[str, set[str]] = defaultdict(set)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for index, unit in enumerate(unit_list):
        identifier = unit_id(unit) or str(index)
        for key, value in metadata(unit).items():
            key_text = str(key).strip()
            if not key_text:
                continue
            key_counts[key_text] += 1
            samples[key_text].add(identifier)
            type_counts[key_text][_value_type(value)] += 1

    rows = [
        {
            "metadata_key": key,
            "unit_count": count,
            "coverage_percent": f"{(count / total_units * 100) if total_units else 0:.2f}",
            "sample_unit_ids": "; ".join(sorted(samples[key], key=sort_key)[:_SAMPLE_LIMIT]),
            "value_type_mix": "; ".join(f"{kind}:{amount}" for kind, amount in sorted(type_counts[key].items(), key=lambda item: sort_key(item[0]))),
        }
        for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))
    ]
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": total_units, "rows_exported": len(rows), "bytes_written": bytes_written}


def _value_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, list | tuple | set):
        return "sequence"
    return "object"
