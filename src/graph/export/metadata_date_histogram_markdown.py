"""Markdown histogram export for date-like metadata paths."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_BUCKETS = {"day", "week", "month", "year"}
_WHITESPACE_RE = re.compile(r"\s+")


def export_metadata_date_histogram_markdown(
    units: Iterable[KnowledgeUnit],
    metadata_path: str,
    path: str | Path | None = None,
    *,
    bucket: str = "month",
    source_project: str | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown histogram for an ISO metadata date path."""
    normalized_path = _inline_text(metadata_path)
    if not normalized_path:
        raise ValueError("metadata_path must be a non-empty string")
    normalized_bucket = _inline_text(bucket).lower()
    if normalized_bucket not in _BUCKETS:
        valid = ", ".join(sorted(_BUCKETS))
        raise ValueError(f"Unsupported metadata date histogram bucket: {normalized_bucket}. Use one of: {valid}.")
    normalized_source = _inline_text(source_project) if source_project is not None else None
    if source_project is not None and not normalized_source:
        raise ValueError("source_project must be a non-empty string or None")

    unit_list = [
        unit
        for unit in sorted(list(units), key=_unit_sort_key)
        if normalized_source is None or _unit_source(unit) == normalized_source
    ]
    rows, missing_count, invalid_count = _histogram_rows(
        unit_list,
        metadata_path=normalized_path,
        bucket=normalized_bucket,
    )
    text = _render_report(
        rows,
        metadata_path=normalized_path,
        bucket=normalized_bucket,
        source_project=normalized_source,
        units_scanned=len(unit_list),
        missing_count=missing_count,
        invalid_count=invalid_count,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "metadata_path": normalized_path,
        "bucket": normalized_bucket,
        "source_project": normalized_source,
        "units_scanned": len(unit_list),
        "buckets_exported": len(rows),
        "missing_count": missing_count,
        "invalid_count": invalid_count,
        "bytes_written": output_path.stat().st_size,
    }


def _histogram_rows(
    units: list[KnowledgeUnit],
    *,
    metadata_path: str,
    bucket: str,
) -> tuple[list[dict[str, Any]], int, int]:
    bucket_units: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    missing_count = 0
    invalid_count = 0

    for unit in units:
        found, raw_value = _metadata_path_lookup(unit.metadata, metadata_path)
        if not found or raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
            missing_count += 1
            continue
        parsed = _parse_iso_date(raw_value)
        if parsed is None:
            invalid_count += 1
            continue
        label = _bucket_label(parsed, bucket)
        unit_key = _unit_id(unit)
        if unit_key in bucket_units[label]:
            continue
        bucket_units[label].add(unit_key)
        source_counts[label][_unit_source(unit)] += 1
        if len(examples[label]) < 3:
            examples[label].append(_unit_label(unit))

    rows = [
        {
            "bucket": label,
            "unit_count": len(bucket_units[label]),
            "source_project_counts": _counter_text(source_counts[label]),
            "examples": "; ".join(examples[label]),
        }
        for label in sorted(bucket_units)
    ]
    return rows, missing_count, invalid_count


def _metadata_path_lookup(metadata: Any, metadata_path: str) -> tuple[bool, Any]:
    current: Any = metadata
    for part in metadata_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _parse_iso_date(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed_date = date.fromisoformat(text)
        except ValueError:
            return None
        return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
    return parsed


def _bucket_label(value: datetime, bucket: str) -> str:
    if bucket == "day":
        return value.date().isoformat()
    if bucket == "week":
        week_start = value.date() - timedelta(days=value.weekday())
        return week_start.isoformat()
    if bucket == "month":
        return f"{value.year:04d}-{value.month:02d}"
    if bucket == "year":
        return f"{value.year:04d}"
    raise ValueError(f"Unsupported metadata date histogram bucket: {bucket}")


def _render_report(
    rows: list[dict[str, Any]],
    *,
    metadata_path: str,
    bucket: str,
    source_project: str | None,
    units_scanned: int,
    missing_count: int,
    invalid_count: int,
) -> str:
    valid_count = sum(row["unit_count"] for row in rows)
    lines = [
        "# Metadata Date Histogram",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Metadata path | {_markdown_cell(metadata_path)} |",
        f"| Bucket | {_markdown_cell(bucket)} |",
        f"| Units scanned | {units_scanned} |",
        f"| Valid units | {valid_count} |",
        f"| Missing units | {missing_count} |",
        f"| Invalid units | {invalid_count} |",
        f"| Skipped units | {missing_count + invalid_count} |",
    ]
    if source_project is not None:
        lines.append(f"| Source project | {_markdown_cell(source_project)} |")
    lines.extend(
        [
            "",
            "## Buckets",
            "",
            "| Bucket | Units | Sources | Examples |",
            "| --- | ---: | --- | --- |",
        ]
    )
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['bucket'])} | "
                f"{row['unit_count']} | "
                f"{_markdown_cell(row['source_project_counts'])} | "
                f"{_markdown_cell(row['examples'])} |"
            )
    else:
        lines.append("| _None_ | 0 | _None_ | _None_ |")
    return "\n".join(lines).rstrip() + "\n"


def _counter_text(counter: Counter[str]) -> str:
    if not counter:
        return "_None_"
    return "; ".join(
        f"{key} ({count})" for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def _unit_source(unit: KnowledgeUnit) -> str:
    return _inline_text(getattr(unit.source_project, "value", unit.source_project)) or "Unknown"


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id) or _inline_text(unit.title)


def _unit_label(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or _inline_text(unit.source_id) or _inline_text(unit.id) or "Untitled"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (_unit_source(unit), _inline_text(unit.source_id), _inline_text(unit.title), _inline_text(unit.id))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
