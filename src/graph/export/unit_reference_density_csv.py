"""CSV export for reference density by source project."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "source_project",
    "unit_count",
    "total_urls",
    "total_markdown_wiki_references",
    "units_with_references",
    "average_references_per_unit",
    "unreferenced_unit_count",
]
_METADATA_REFERENCE_KEYS = {"url", "source_url", "links", "references"}
_URL_RE = re.compile(r"https?://[^\s<>()\]]+")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_reference_density_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write reference density grouped by source project."""
    unit_list = list(units)
    rows = _density_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "source_project_count": len(rows),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _density_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[_ReferenceCounts]] = defaultdict(list)

    for unit in sorted(units, key=_unit_sort_key):
        groups[_unit_source(unit)].append(_unit_references(unit))

    rows: list[dict[str, str | int]] = []
    for source_project in sorted(groups, key=_sort_key):
        counts = groups[source_project]
        unit_count = len(counts)
        total_references = sum(count.reference_count for count in counts)
        units_with_references = sum(1 for count in counts if count.reference_count > 0)
        rows.append(
            {
                "source_project": source_project,
                "unit_count": unit_count,
                "total_urls": sum(count.url_count for count in counts),
                "total_markdown_wiki_references": sum(count.markdown_wiki_count for count in counts),
                "units_with_references": units_with_references,
                "average_references_per_unit": _decimal(total_references / unit_count if unit_count else 0),
                "unreferenced_unit_count": unit_count - units_with_references,
            }
        )
    return rows


class _ReferenceCounts:
    def __init__(self, urls: set[str], markdown_wiki: set[str], all_references: set[str]) -> None:
        self.url_count = len(urls)
        self.markdown_wiki_count = len(markdown_wiki)
        self.reference_count = len(all_references)


def _unit_references(unit: KnowledgeUnit) -> _ReferenceCounts:
    urls: set[str] = set()
    markdown_wiki: set[str] = set()

    content = _inline_text(unit.content)
    urls.update(_normalize_reference(match.group(0)) for match in _URL_RE.finditer(content))
    markdown_wiki.update(
        _normalize_reference(match.group(1)) for match in _MARKDOWN_LINK_RE.finditer(content)
    )
    markdown_wiki.update(_normalize_reference(match.group(1)) for match in _WIKI_LINK_RE.finditer(content))

    metadata = unit.metadata if isinstance(unit.metadata, dict) else {}
    for key, value in metadata.items():
        if _inline_text(key).casefold() not in _METADATA_REFERENCE_KEYS:
            continue
        for text in _metadata_strings(value):
            urls.update(_normalize_reference(match.group(0)) for match in _URL_RE.finditer(text))

    urls.discard("")
    markdown_wiki.discard("")
    return _ReferenceCounts(urls, markdown_wiki, urls | markdown_wiki)


def _metadata_strings(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_metadata_strings(item))
        return strings
    if isinstance(value, list | tuple | set):
        strings = []
        for item in value:
            strings.extend(_metadata_strings(item))
        return strings
    return [_inline_text(value)]


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_source(unit)), _sort_key(unit.id or unit.source_id))


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize_reference(value: object) -> str:
    return _inline_text(value).rstrip(".,;:")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _decimal(value: float) -> str:
    return f"{value:.2f}"
