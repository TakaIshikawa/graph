"""CSV export for source diversity by collection."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["collection", "unit_count", "source_count", "dominant_source", "dominant_source_unit_count", "dominant_source_share"]
_WHITESPACE_RE = re.compile(r"\s+")


def export_collection_source_diversity_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write collection-level source diversity summaries."""
    unit_list = list(units)
    rows = _diversity_rows(unit_list)
    text = _render_csv(rows)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _diversity_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for unit in units:
        groups[_collection(unit)][_source(unit)] += 1
    rows: list[dict[str, str | int]] = []
    for collection, counts in groups.items():
        unit_count = sum(counts.values())
        dominant_source, dominant_count = min(counts.items(), key=lambda item: (-item[1], _sort_key(item[0])))
        rows.append(
            {
                "collection": collection,
                "unit_count": unit_count,
                "source_count": len(counts),
                "dominant_source": dominant_source,
                "dominant_source_unit_count": dominant_count,
                "dominant_source_share": f"{(dominant_count / unit_count * 100):.2f}" if unit_count else "0.00",
            }
        )
    return sorted(rows, key=lambda row: _sort_key(row["collection"]))


def _collection(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    metadata = _metadata(unit)
    value = _get(unit, "collection") or _casefold_get(metadata, "collection")
    return _field_value(value) or "Unassigned"


def _source(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "source_project")) or _field_value(_get(unit, "source_id")) or "Unknown"


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


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
