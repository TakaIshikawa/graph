"""CSV export helpers for source timeline reports."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any

_FIELDNAMES = [
    "bucket_start",
    "bucket_end",
    "bucket_label",
    "source_project",
    "source_entity_type",
    "unit_count",
    "top_titles",
]
_UNKNOWN = "unknown"
_UNTITLED = "Untitled"


def export_source_timeline_csv(
    timeline: dict,
    path: str | Path | None = None,
) -> str | dict:
    """Return or write source timeline buckets as deterministic CSV rows."""
    rows = _timeline_rows(timeline)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "rows_written": len(rows)}


def _timeline_rows(timeline: Mapping[str, Any]) -> list[dict[str, Any]]:
    bucket_kind = _text(_mapping(timeline.get("stats")).get("bucket")) or "month"
    events_by_bucket_source_entity = _events_by_bucket_source_entity(timeline, bucket_kind)
    rows: list[dict[str, Any]] = []

    for bucket in _sorted_buckets(timeline.get("buckets")):
        bucket_start = _text(bucket.get("start"))
        bucket_label = _text(bucket.get("bucket")) or bucket_start
        bucket_end = _bucket_end(bucket_start, bucket_kind)
        source_counts = _source_counts(bucket.get("sources"))

        for source_project, count in source_counts:
            matching_keys = sorted(
                key
                for key in events_by_bucket_source_entity
                if key[0] == bucket_start and key[1] == source_project
            )
            if not matching_keys:
                matching_keys = [(bucket_start, source_project, _UNKNOWN)]

            for _, _, source_entity_type in matching_keys:
                title_pairs = events_by_bucket_source_entity.get(
                    (bucket_start, source_project, source_entity_type),
                    [],
                )
                unit_count = len(title_pairs) if title_pairs else count
                rows.append(
                    {
                        "bucket_start": bucket_start,
                        "bucket_end": bucket_end,
                        "bucket_label": bucket_label,
                        "source_project": source_project,
                        "source_entity_type": source_entity_type,
                        "unit_count": unit_count,
                        "top_titles": "; ".join(_unique_titles(title_pairs)),
                    }
                )

    return sorted(
        rows,
        key=lambda row: (
            _sort_key(row["bucket_start"]),
            _sort_key(row["bucket_label"]),
            _sort_key(row["source_project"]),
            _sort_key(row["source_entity_type"]),
        ),
    )


def _render_csv(rows: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _events_by_bucket_source_entity(
    timeline: Mapping[str, Any],
    bucket_kind: str,
) -> dict[tuple[str, str, str], list[tuple[str, str]]]:
    indexed: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for event in timeline.get("events") or []:
        if not isinstance(event, Mapping):
            continue
        event_date = _parse_date(event.get("date"))
        if event_date is None:
            continue
        bucket_start = _event_bucket_start(event_date, bucket_kind)
        source_project = _source_label(event.get("source_project", event.get("source")))
        source_entity_type = _source_label(event.get("source_entity_type"))
        title = _title_label(event.get("title"))
        indexed[(bucket_start, source_project, source_entity_type)].append(
            (event_date.isoformat(), title)
        )

    return dict(sorted(indexed.items(), key=lambda item: item[0]))


def _sorted_buckets(value: Any) -> list[Mapping[str, Any]]:
    buckets = [bucket for bucket in value or [] if isinstance(bucket, Mapping)]
    return sorted(
        buckets,
        key=lambda bucket: (
            _sort_key(bucket.get("start")),
            _sort_key(bucket.get("bucket")),
        ),
    )


def _source_counts(value: Any) -> list[tuple[str, int]]:
    if not isinstance(value, Mapping):
        return []
    counts: dict[str, int] = defaultdict(int)
    for source, count in value.items():
        if isinstance(count, int) and not isinstance(count, bool):
            counts[_source_label(source)] += count
        else:
            counts[_source_label(source)] += 0
    return sorted(
        counts.items(),
        key=lambda item: _sort_key(item[0]),
    )


def _bucket_end(bucket_start: str, bucket_kind: str) -> str:
    start = _parse_date(bucket_start)
    if start is None:
        return ""
    if bucket_kind == "day":
        end = start
    elif bucket_kind == "week":
        end = start + timedelta(days=6)
    elif bucket_kind == "year":
        end = date(start.year, 12, 31)
    else:
        next_month = date(start.year + (start.month // 12), (start.month % 12) + 1, 1)
        end = next_month - timedelta(days=1)
    return end.isoformat()


def _event_bucket_start(value: date, bucket_kind: str) -> str:
    if bucket_kind == "day":
        return value.isoformat()
    if bucket_kind == "week":
        return (value - timedelta(days=value.weekday())).isoformat()
    if bucket_kind == "year":
        return date(value.year, 1, 1).isoformat()
    return date(value.year, value.month, 1).isoformat()


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None


def _unique_titles(pairs: list[tuple[str, str]]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for _, title in sorted(
        pairs,
        key=lambda item: (_sort_key(item[0]), _sort_key(item[1])),
    ):
        if title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _source_label(value: Any) -> str:
    label = _text(value)
    return label or _UNKNOWN


def _title_label(value: Any) -> str:
    label = _text(value)
    return label or _UNTITLED


def _text(value: Any) -> str:
    return " ".join(str(value).strip().split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
