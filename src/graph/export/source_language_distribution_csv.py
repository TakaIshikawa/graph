"""CSV export for source language distribution."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "language_code",
    "language_label",
    "unit_count",
    "percent_of_group",
]
_LANGUAGE_KEYS = ("language", "lang", "locale", "content_language", "source_language")
_LANGUAGE_LABELS = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "zh": "Chinese",
}
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_language_distribution_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write language bucket counts by source group."""
    unit_list = list(units)
    rows = _distribution_rows(unit_list)
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


def _distribution_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for unit in units:
        group = (_field_value(unit.source_project) or "Unknown", _field_value(unit.source_entity_type) or "Unknown")
        counts[group][_language_code(unit)] += 1

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type), language_counts in counts.items():
        total = sum(language_counts.values())
        for language_code, unit_count in sorted(language_counts.items(), key=lambda item: _sort_key(item[0])):
            rows.append(
                {
                    "source_project": source_project,
                    "source_entity_type": source_entity_type,
                    "language_code": language_code,
                    "language_label": _language_label(language_code),
                    "unit_count": unit_count,
                    "percent_of_group": f"{(unit_count / total * 100):.2f}" if total else "0.00",
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
            _sort_key(row["language_code"]),
        ),
    )


def _language_code(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for key in _LANGUAGE_KEYS:
        text = _inline_text(metadata.get(key))
        if text:
            return _normalize_language_code(text)
    return "unknown"


def _normalize_language_code(value: str) -> str:
    text = value.replace("_", "-").casefold()
    if "," in text:
        text = text.split(",", 1)[0]
    if "-" in text:
        text = text.split("-", 1)[0]
    return re.sub(r"[^a-z]", "", text) or "unknown"


def _language_label(language_code: str) -> str:
    if language_code == "unknown":
        return "Unknown"
    return _LANGUAGE_LABELS.get(language_code, language_code)


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
