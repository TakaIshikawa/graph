"""CSV export for source URL domain coverage."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["domain", "source_type", "source_label", "unit_count"]
_UNKNOWN = "unknown"
_URL_KEYS = {
    "url",
    "urls",
    "link",
    "links",
    "source_url",
    "external_url",
    "canonical_url",
    "web_url",
    "webpage_url",
}
_LABEL_KEYS = ("source_label", "source_name", "source_title", "label", "name")
_WHITESPACE_RE = re.compile(r"\s+")


def export_source_domain_coverage_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write unit counts grouped by URL domain and source metadata."""
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
        source_type = _source_type(unit)
        source_label = _source_label(unit)
        domains = _unit_domains(unit)
        for domain in domains or {_UNKNOWN}:
            counts[(domain, source_type, source_label)] += 1

    return [
        {
            "domain": domain,
            "source_type": source_type,
            "source_label": source_label,
            "unit_count": unit_count,
        }
        for (domain, source_type, source_label), unit_count in sorted(
            counts.items(),
            key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]), _sort_key(item[0][2])),
        )
    ]


def _unit_domains(unit: KnowledgeUnit) -> set[str]:
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    domains = {_domain(value) for value in _url_values(metadata)}
    return {domain for domain in domains if domain}


def _url_values(value: object, *, key: str = "") -> list[object]:
    if isinstance(value, Mapping):
        values: list[object] = []
        for nested_key, nested_value in value.items():
            normalized = _normalized_key(nested_key)
            if normalized in _URL_KEYS:
                values.extend(_flat_values(nested_value))
            elif normalized == "source":
                values.extend(_url_values(nested_value, key=normalized))
            elif key == "source":
                values.extend(_url_values(nested_value, key=normalized))
        return values
    return _flat_values(value)


def _flat_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flat_values(entry)]
    if isinstance(value, Mapping):
        return _url_values(value)
    return [value]


def _domain(value: object) -> str:
    text = _inline_text(value)
    if not text:
        return ""
    parsed = urlparse(text)
    if not parsed.scheme and "." in text and "/" not in text:
        parsed = urlparse(f"https://{text}")
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        return ""
    return parsed.hostname.casefold() if parsed.hostname else ""


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


def _normalized_key(value: object) -> str:
    return _inline_text(value).casefold().replace("-", "_").replace(" ", "_")


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
