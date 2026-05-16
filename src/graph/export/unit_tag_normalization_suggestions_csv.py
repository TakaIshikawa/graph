"""CSV export for unit tag normalization suggestions."""

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
    "normalized_tag",
    "variant_count",
    "unit_count",
    "variants",
    "suggested_tag",
    "source_projects",
    "unit_ids",
    "confidence",
]
_TAG_METADATA_KEYS = ("tags", "tag", "keywords", "labels")
_WHITESPACE_RE = re.compile(r"\s+")
_SEPARATOR_RE = re.compile(r"[\s_.-]+")
_PUNCT_RE = re.compile(r"[^A-Za-z0-9]+")


def export_unit_tag_normalization_suggestions_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write tag groups where raw variants normalize to one tag."""
    unit_list = list(units)
    rows = _suggestion_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "suggestion_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _suggestion_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, list[tuple[str, KnowledgeUnit | Mapping[str, Any]]]] = defaultdict(list)
    for unit in units:
        for tag in _unit_tags(unit):
            normalized_tag = _normalize_tag(tag)
            if not normalized_tag:
                continue
            groups[normalized_tag].append((tag, unit))

    rows: list[dict[str, str | int]] = []
    for normalized_tag, entries in groups.items():
        variants = sorted({tag for tag, _ in entries}, key=_sort_key)
        if len(variants) < 2:
            continue
        unit_ids = {_field_value(_get(unit, "id")) for _, unit in entries if _field_value(_get(unit, "id"))}
        source_projects = {
            _field_value(_get(unit, "source_project")) or "Unknown"
            for _, unit in entries
        }
        rows.append(
            {
                "normalized_tag": normalized_tag,
                "variant_count": len(variants),
                "unit_count": len(unit_ids),
                "variants": "; ".join(variants),
                "suggested_tag": _suggested_tag(tag for tag, _ in entries),
                "source_projects": "; ".join(sorted(source_projects, key=_sort_key)),
                "unit_ids": "; ".join(sorted(unit_ids, key=_sort_key)),
                "confidence": _confidence(len(variants), len(unit_ids)),
            }
        )

    return sorted(rows, key=lambda row: (-int(row["unit_count"]), _sort_key(row["normalized_tag"])))


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    tags.extend(_string_values(_get(unit, "tags")))
    metadata = _metadata(unit)
    for key in _TAG_METADATA_KEYS:
        tags.extend(_string_values(metadata.get(key)))
    return [_field_value(tag) for tag in tags if _field_value(tag)]


def _string_values(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, str):
        text = _field_value(value)
        return [text] if text else []
    if isinstance(value, Mapping):
        return []
    if isinstance(value, Iterable):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


def _normalize_tag(value: str) -> str:
    text = _field_value(value).lstrip("#")
    text = _SEPARATOR_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    normalized = _WHITESPACE_RE.sub(" ", text).strip().casefold()
    parts = normalized.split()
    if parts and all(len(part) == 1 for part in parts):
        return "".join(parts)
    return normalized


def _suggested_tag(tags: Iterable[str]) -> str:
    counts = Counter(tags)
    return sorted(counts, key=lambda tag: (-counts[tag], tag.casefold(), tag != tag.casefold(), tag))[0]


def _confidence(variant_count: int, unit_count: int) -> str:
    value = 0.6 + min(variant_count - 2, 3) * 0.08 + min(unit_count, 5) * 0.03
    return f"{min(value, 0.95):.2f}"


def _metadata(value: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(value, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


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
