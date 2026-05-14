"""CSV export for unit review readiness."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_project",
    "source_entity_type",
    "missing_field_count",
    "missing_fields",
    "review_status",
    "needs_review",
]
_REVIEW_STATUS_KEYS = ("review_status", "triage_status", "status")
_NEEDS_REVIEW_KEYS = ("needs_review", "reviewed")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_review_readiness_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write one readiness row per unit."""
    unit_list = list(units)
    rows = _readiness_rows(unit_list)
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


def _readiness_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        missing = _missing_fields(unit)
        review_status = _review_status(unit)
        needs_review = _needs_review(unit, missing, review_status)
        rows.append(
            {
                "unit_id": _field_value(unit.id),
                "title": _inline_text(unit.title),
                "source_project": _field_value(unit.source_project),
                "source_entity_type": _field_value(unit.source_entity_type),
                "missing_field_count": len(missing),
                "missing_fields": ";".join(missing),
                "review_status": review_status,
                "needs_review": "true" if needs_review else "false",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            0 if row["needs_review"] == "true" else 1,
            _sort_key(row["unit_id"]),
            _sort_key(row["title"]),
        ),
    )


def _missing_fields(unit: KnowledgeUnit) -> list[str]:
    missing: list[str] = []
    checks = [
        ("title", unit.title),
        ("content", unit.content),
        ("tags", unit.tags),
        ("source_id", unit.source_id),
        ("source_project", unit.source_project),
        ("source_entity_type", unit.source_entity_type),
    ]
    for name, value in checks:
        if name == "tags":
            tags = [_inline_text(tag) for tag in (value or [])]
            if not any(tags):
                missing.append(name)
            continue
        if not _field_value(value):
            missing.append(name)
    return missing


def _review_status(unit: KnowledgeUnit) -> str:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for key in _REVIEW_STATUS_KEYS:
        text = _inline_text(metadata.get(key))
        if text:
            return text.casefold()
    reviewed = metadata.get("reviewed")
    if reviewed is not None:
        return "reviewed" if _truthy(reviewed) else "unreviewed"
    needs_review = metadata.get("needs_review")
    if needs_review is not None:
        return "needs_review" if _truthy(needs_review) else "ready"
    return ""


def _needs_review(unit: KnowledgeUnit, missing: list[str], review_status: str) -> bool:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    if "needs_review" in metadata:
        return _truthy(metadata.get("needs_review"))
    if "reviewed" in metadata:
        return not _truthy(metadata.get("reviewed"))
    if review_status in {"done", "complete", "completed", "approved", "ready", "reviewed"}:
        return False
    if review_status in {"todo", "needs_review", "unreviewed", "draft", "triage", "pending"}:
        return True
    return bool(missing)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _inline_text(value).casefold() in {"1", "true", "yes", "y", "reviewed", "done"}


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
