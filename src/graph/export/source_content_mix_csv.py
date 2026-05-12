"""CSV export for per-source content type composition."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "content_type",
    "unit_count",
    "source_unit_count",
    "source_percent",
]
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_content_mix_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic per-source content type mix CSV."""
    unit_list = list(units)
    rows = _mix_rows(unit_list)
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


def _mix_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    source_totals: Counter[str] = Counter()

    for unit in units:
        source_project = _unit_source(unit)
        content_type = _unit_content_type(unit)
        groups[source_project][content_type] += 1
        source_totals[source_project] += 1

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        source_unit_count = source_totals[source_project]
        for content_type, unit_count in sorted(
            groups[source_project].items(),
            key=lambda item: (-item[1], _sort_key(item[0])),
        ):
            rows.append(
                {
                    "source_project": source_project,
                    "content_type": content_type,
                    "unit_count": unit_count,
                    "source_unit_count": source_unit_count,
                    "source_percent": _decimal(unit_count * 100 / source_unit_count),
                }
            )
    return rows


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_content_type(unit: KnowledgeUnit) -> str:
    return _field_value(unit.content_type) or "Unknown"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
