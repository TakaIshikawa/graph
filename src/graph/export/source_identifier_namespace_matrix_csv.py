"""CSV export for source identifier namespace coverage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, source_id, write_csv

_NAMESPACES = ("arxiv", "doi", "external_id", "id", "isbn", "orcid", "pmid", "uri", "url")


def export_source_identifier_namespace_matrix_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    source_list = list(sources)
    rows: list[dict[str, Any]] = []
    detected: set[str] = set()
    for source in source_list:
        counts = _counts(source)
        detected.update(key for key, count in counts.items() if count)
        rows.append({"source": source_id(source), **counts, "total_identifiers": sum(counts.values())})
    namespaces = sorted(detected)
    fieldnames = ["source", *namespaces, "total_identifiers"]
    normalized_rows = []
    for row in rows:
        normalized_rows.append({"source": row["source"], **{key: row.get(key, 0) for key in namespaces}, "total_identifiers": row["total_identifiers"]})
    normalized_rows.sort(key=lambda row: sort_key(row["source"]))
    text = render_csv(normalized_rows, fieldnames)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(normalized_rows), "bytes_written": bytes_written}


def _counts(source: Mapping[str, Any] | object) -> dict[str, int]:
    counts = {key: 0 for key in _NAMESPACES}
    for key in _NAMESPACES:
        counts[key] += _value_count(get(source, key))
        counts[key] += _value_count(metadata(source).get(key))
    return counts


def _value_count(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return sum(_value_count(child) for child in value.values())
    if isinstance(value, list | tuple | set):
        return sum(_value_count(child) for child in value)
    return 1 if field_value(value) else 0
