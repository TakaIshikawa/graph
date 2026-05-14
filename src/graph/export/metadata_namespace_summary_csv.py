"""CSV export for metadata namespace summaries."""

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
    "source_project",
    "source_entity_type",
    "namespace",
    "key_count",
    "unit_count",
    "total_value_count",
    "sample_keys",
]
_SEPARATOR_RE = re.compile(r"[._:-]")
_WHITESPACE_RE = re.compile(r"\s+")
_SAMPLE_LIMIT = 5


def export_metadata_namespace_summary_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write metadata namespace counts by source/type."""
    unit_list = list(units)
    rows = _summary_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _summary_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"keys": set(), "units": set(), "value_count": 0}
    )
    for unit in units:
        metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
        unit_seen: set[tuple[str, str, str]] = set()
        for key, value in metadata.items():
            key_text = _field_value(key)
            if not key_text:
                continue
            namespace = _namespace(key_text)
            group_key = (_unit_source(unit), _unit_source_type(unit), namespace)
            groups[group_key]["keys"].add(key_text)
            groups[group_key]["value_count"] += _value_count(value)
            unit_seen.add(group_key)
        for group_key in unit_seen:
            groups[group_key]["units"].add(_field_value(unit.id))

    rows: list[dict[str, str | int]] = []
    for source_project, source_entity_type, namespace in sorted(
        groups,
        key=lambda key: (_sort_key(key[0]), _sort_key(key[1]), _sort_key(key[2])),
    ):
        group = groups[(source_project, source_entity_type, namespace)]
        sample_keys = sorted(group["keys"], key=_sort_key)[:_SAMPLE_LIMIT]
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "namespace": namespace,
                "key_count": len(group["keys"]),
                "unit_count": len(group["units"]),
                "total_value_count": group["value_count"],
                "sample_keys": "; ".join(sample_keys),
            }
        )
    return rows


def _namespace(key: str) -> str:
    match = _SEPARATOR_RE.search(key)
    if match is None:
        return "root"
    return key[: match.start()] or "root"


def _value_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return sum(_value_count(item) for item in value.values()) or 1
    if isinstance(value, list | tuple | set):
        return sum(_value_count(item) for item in value) or 0
    return 1


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_source_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_entity_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
