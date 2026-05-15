"""CSV export for source language coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["language", "source_type", "source_label", "unit_count"]
_UNKNOWN = "unknown"
_LANGUAGE_KEYS = ("language", "lang", "locale", "content_language")
_LABEL_KEYS = ("source_label", "source_name", "source_title", "label", "name")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_language_coverage_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write unit counts grouped by language and source metadata."""
    unit_list = list(units)
    rows = _coverage_rows(unit_list)
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


def _coverage_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for unit in units:
        counts[(_language(unit), _source_type(unit), _source_label(unit))] += 1

    return [
        {
            "language": language,
            "source_type": source_type,
            "source_label": source_label,
            "unit_count": unit_count,
        }
        for (language, source_type, source_label), unit_count in sorted(
            counts.items(),
            key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]), _sort_key(item[0][2])),
        )
    ]


def _language(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    text = _metadata_text(metadata, _LANGUAGE_KEYS)
    if not text:
        return _UNKNOWN
    text = text.replace("_", "-").casefold()
    if "," in text:
        text = text.split(",", 1)[0]
    return text or _UNKNOWN


def _source_type(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    return _metadata_text(metadata, ("source_type", "type")) or _field_value(unit.source_entity_type) or _UNKNOWN


def _source_label(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    return _metadata_text(metadata, _LABEL_KEYS) or _field_value(unit.source_project) or _UNKNOWN


def _metadata_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _inline_text(metadata.get(key))
        if text:
            return text
    source = metadata.get("source")
    if isinstance(source, Mapping):
        for key in keys:
            text = _inline_text(source.get(key))
            if text:
                return text
    return ""


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
