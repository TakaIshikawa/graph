"""CSV export for tag overlap between source projects."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from itertools import combinations
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "left_source_project",
    "right_source_project",
    "left_tag_count",
    "right_tag_count",
    "shared_tag_count",
    "jaccard_similarity",
    "shared_tags",
]
_WHITESPACE_RE = re.compile(r"\s+")
_METADATA_TAG_KEYS = ("tags", "tag", "labels", "label", "keywords", "keyword", "topics", "topic")


def export_source_tag_overlap_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write pairwise tag overlap rows for source projects."""
    unit_list = list(units)
    rows = _overlap_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_pair_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _overlap_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    source_tags: dict[str, dict[str, str]] = defaultdict(dict)
    for unit in units:
        source_project = _field_value(getattr(unit, "source_project", None)) or "Unknown"
        source_tags[source_project]
        for normalized, display in _unit_tags(unit).items():
            source_tags[source_project].setdefault(normalized, display)

    rows: list[dict[str, str | int]] = []
    for left, right in combinations(sorted(source_tags, key=_sort_key), 2):
        left_tags = set(source_tags[left])
        right_tags = set(source_tags[right])
        shared = left_tags & right_tags
        union = left_tags | right_tags
        shared_tags = sorted((source_tags[left].get(tag) or source_tags[right][tag] for tag in shared), key=_sort_key)
        rows.append(
            {
                "left_source_project": left,
                "right_source_project": right,
                "left_tag_count": len(left_tags),
                "right_tag_count": len(right_tags),
                "shared_tag_count": len(shared),
                "jaccard_similarity": _format_ratio(len(shared), len(union)),
                "shared_tags": "; ".join(shared_tags),
            }
        )
    return rows


def _unit_tags(unit: KnowledgeUnit) -> dict[str, str]:
    tags: dict[str, str] = {}
    for value in getattr(unit, "tags", []) or []:
        _add_tag(tags, value)
    metadata = getattr(unit, "metadata", None)
    if isinstance(metadata, dict):
        for key in _METADATA_TAG_KEYS:
            if key in metadata:
                for value in _iter_tag_values(metadata.get(key)):
                    _add_tag(tags, value)
    return tags


def _iter_tag_values(value: object) -> Iterable[object]:
    if isinstance(value, dict):
        return value.values()
    if isinstance(value, list | tuple | set):
        return value
    return [value]


def _add_tag(tags: dict[str, str], value: object) -> None:
    display = _inline_text(value)
    normalized = display.casefold()
    if display and normalized:
        tags.setdefault(normalized, display)


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00"
    return f"{numerator / denominator:.2f}"


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
