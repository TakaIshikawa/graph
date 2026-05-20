"""CSV export for external link domains by unit source/type."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "source_entity_type",
    "domain",
    "unit_count",
    "linked_unit_count",
    "link_count",
    "sample_unit_ids",
]
_URL_RE = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_external_link_domain_matrix_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
    *,
    sample_limit: int = 3,
) -> str | dict[str, Any]:
    """Return or write external link domain counts by source project and entity type."""
    sample_limit = _positive_int(sample_limit, "sample_limit")
    unit_list = list(units)
    rows = _domain_rows(unit_list, sample_limit)
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
        "sample_limit": sample_limit,
        "bytes_written": output_path.stat().st_size,
    }


def _domain_rows(units: list[KnowledgeUnit | Mapping[str, Any]], sample_limit: int) -> list[dict[str, str | int]]:
    unit_counts: dict[tuple[str, str], set[str]] = defaultdict(set)
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for unit in units:
        source_project = _field_value(_get(unit, "source_project")) or "Unknown"
        source_entity_type = _field_value(_get(unit, "source_entity_type")) or "Unknown"
        unit_id = _unit_id(unit)
        unit_counts[(source_project, source_entity_type)].add(unit_id)
        counts = _unit_domain_counts(unit)
        for domain, link_count in counts.items():
            key = (source_project, source_entity_type, domain)
            group = groups.setdefault(key, {"linked_unit_ids": set(), "link_count": 0})
            group["linked_unit_ids"].add(unit_id)
            group["link_count"] += link_count

    rows: list[dict[str, str | int]] = []
    for (source_project, source_entity_type, domain), group in groups.items():
        linked_ids = sorted(group["linked_unit_ids"], key=_sort_key)
        rows.append(
            {
                "source_project": source_project,
                "source_entity_type": source_entity_type,
                "domain": domain,
                "unit_count": len(unit_counts[(source_project, source_entity_type)]),
                "linked_unit_count": len(linked_ids),
                "link_count": group["link_count"],
                "sample_unit_ids": "; ".join(linked_ids[:sample_limit]),
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source_project"]), _sort_key(row["source_entity_type"]), _sort_key(row["domain"])))


def _unit_domain_counts(unit: KnowledgeUnit | Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for text in [_field_value(_get(unit, "content")), *_metadata_strings(_metadata(unit))]:
        for raw_url in _URL_RE.findall(text):
            domain = _domain(raw_url)
            if domain:
                counts[domain] += 1
    return dict(counts)


def _metadata_strings(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    if isinstance(value, str):
        return [_inline_text(value)] if _inline_text(value) else []
    if isinstance(value, Mapping):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_metadata_strings(item))
        return strings
    if isinstance(value, Iterable):
        strings: list[str] = []
        for item in value:
            strings.extend(_metadata_strings(item))
        return strings
    return []


def _domain(raw_url: str) -> str:
    parsed = urlsplit(raw_url.strip().rstrip(".,;:!?)]}"))
    if parsed.scheme.casefold() not in {"http", "https"}:
        return ""
    return (parsed.hostname or "").casefold()


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
