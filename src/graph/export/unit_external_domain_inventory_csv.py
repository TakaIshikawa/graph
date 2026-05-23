"""CSV export for external domains referenced by units."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["unit_id", "title", "domain", "url_count", "schemes", "sample_urls"]
_URL_RE = re.compile(r"\b(?:https?|ftp)://[^\s<>'\"]+")
_WHITESPACE_RE = re.compile(r"\s+")
_SAMPLE_LIMIT = 3


def export_unit_external_domain_inventory_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write per-unit external domain counts from content and metadata."""
    unit_list = list(units)
    rows = _inventory_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _inventory_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for unit in units:
        groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "schemes": set(), "urls": []})
        for url in _unit_urls(unit):
            parsed = urlparse(url)
            domain = parsed.netloc.casefold()
            if not domain:
                continue
            groups[domain]["count"] += 1
            groups[domain]["schemes"].add(parsed.scheme.casefold())
            groups[domain]["urls"].append(url)
        for domain, group in groups.items():
            rows.append(
                {
                    "unit_id": _unit_id(unit),
                    "title": _field_value(_get(unit, "title")),
                    "domain": domain,
                    "url_count": group["count"],
                    "schemes": "; ".join(sorted(group["schemes"], key=_sort_key)),
                    "sample_urls": "; ".join(group["urls"][:_SAMPLE_LIMIT]),
                }
            )
    return sorted(rows, key=lambda row: (_sort_key(row["unit_id"]), _sort_key(row["domain"])))


def _unit_urls(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values = [_get(unit, "content")]
    metadata = _get(unit, "metadata")
    if isinstance(metadata, Mapping):
        values.extend(_metadata_values(metadata))
    urls: list[str] = []
    for value in values:
        urls.extend(_urls_from_text(value))
    return urls


def _metadata_values(value: object) -> list[object]:
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _metadata_values(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _metadata_values(child)]
    return [value]


def _urls_from_text(value: object) -> list[str]:
    if value is None or isinstance(value, bytes):
        return []
    urls: list[str] = []
    for candidate in _URL_RE.findall(str(value)):
        url = candidate.rstrip(".,);]")
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            urls.append(url)
    return urls


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
