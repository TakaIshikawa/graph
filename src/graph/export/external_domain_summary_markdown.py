"""Markdown external domain summary report for knowledge units."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from graph.types.models import KnowledgeUnit

_CONTENT_URL_RE = re.compile(r"https?://[^\s<>\[\]{}\"']+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def export_external_domain_summary_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    metadata_keys: Iterable[str] | None = None,
    min_unit_count: int = 1,
    limit: int | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown summary of external URL domains."""
    normalized_keys = _validate_metadata_keys(metadata_keys)
    if (
        not isinstance(min_unit_count, int)
        or isinstance(min_unit_count, bool)
        or min_unit_count < 1
    ):
        raise ValueError("min_unit_count must be a positive integer")
    if limit is not None and (
        not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
    ):
        raise ValueError("limit must be a positive integer or None")

    unit_list = sorted(list(units), key=_unit_sort_key)
    rows = _domain_rows(unit_list, metadata_keys=normalized_keys, min_unit_count=min_unit_count)
    if limit is not None:
        rows = rows[:limit]
    text = _render_report(
        rows,
        units_scanned=len(unit_list),
        metadata_keys=normalized_keys,
        min_unit_count=min_unit_count,
        limit=limit,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "domains_exported": len(rows),
        "min_unit_count": min_unit_count,
        "limit": limit,
        "metadata_keys": list(normalized_keys) if normalized_keys is not None else None,
        "bytes_written": output_path.stat().st_size,
    }


def _validate_metadata_keys(metadata_keys: Iterable[str] | None) -> tuple[str, ...] | None:
    if metadata_keys is None:
        return None
    if isinstance(metadata_keys, str):
        raise ValueError("metadata_keys must be a sequence of non-empty strings or None")
    try:
        normalized = tuple(dict.fromkeys(_inline_text(key) for key in metadata_keys))
    except TypeError as exc:
        raise ValueError("metadata_keys must be a sequence of non-empty strings or None") from exc
    if any(not key for key in normalized):
        raise ValueError("metadata_keys must be a sequence of non-empty strings or None")
    return normalized


def _domain_rows(
    units: list[KnowledgeUnit],
    *,
    metadata_keys: tuple[str, ...] | None,
    min_unit_count: int,
) -> list[dict[str, Any]]:
    unit_ids: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    location_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)

    for unit in units:
        domains_for_unit: dict[str, set[str]] = defaultdict(set)
        for key, value in _metadata_values(unit.metadata, metadata_keys=metadata_keys):
            for scalar in _iter_scalar_values(value):
                domain = _external_url_domain(scalar)
                if domain is not None:
                    domains_for_unit[domain].add(key)
        for url in _content_urls(unit.content):
            domain = _external_url_domain(url)
            if domain is not None:
                domains_for_unit[domain].add("content")

        unit_key = _unit_id(unit)
        for domain, locations in sorted(domains_for_unit.items()):
            if unit_key in unit_ids[domain]:
                continue
            unit_ids[domain].add(unit_key)
            source_counts[domain][_unit_source(unit)] += 1
            for location in sorted(locations):
                location_counts[domain][location] += 1
            if len(examples[domain]) < 3:
                examples[domain].append(_unit_label(unit))

    rows = [
        {
            "domain": domain,
            "unit_count": len(unit_ids[domain]),
            "source_project_counts": _counter_text(source_counts[domain]),
            "location_counts": _counter_text(location_counts[domain]),
            "examples": "; ".join(examples[domain]),
        }
        for domain in unit_ids
        if len(unit_ids[domain]) >= min_unit_count
    ]
    return sorted(rows, key=lambda row: (-row["unit_count"], row["domain"]))


def _metadata_values(metadata: Any, *, metadata_keys: tuple[str, ...] | None) -> Iterable[tuple[str, Any]]:
    if not isinstance(metadata, Mapping):
        return []
    if metadata_keys is None:
        return list(_flatten_metadata(metadata))
    return [(key, value) for key in metadata_keys if (value := _metadata_path_value(metadata, key)) is not None]


def _flatten_metadata(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key)
            child_path = f"{prefix}.{key}" if prefix else key
            yield from _flatten_metadata(child, child_path)
        return
    yield prefix, value


def _metadata_path_value(metadata: Mapping[str, Any], path: str) -> Any:
    current: Any = metadata
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _iter_scalar_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_scalar_values(child)
        return
    if isinstance(value, list | tuple | set):
        for child in value:
            yield from _iter_scalar_values(child)
        return
    yield value


def _content_urls(content: str) -> set[str]:
    urls: set[str] = set()
    for match in _CONTENT_URL_RE.finditer(content or ""):
        text = match.group(0).rstrip(".,;:!?")
        while text.endswith(")") and text.count("(") < text.count(")"):
            text = text[:-1]
        urls.add(text)
    return urls


def _external_url_domain(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().rstrip(".,;:")
    if not text:
        return None
    parsed = urlsplit(text)
    if not parsed.scheme and not parsed.netloc:
        parsed = urlsplit(f"https://{text}")
    if parsed.scheme.casefold() not in {"http", "https"}:
        return None
    hostname = (parsed.hostname or "").casefold()
    if not hostname or "." not in hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    if port and not (
        (parsed.scheme.casefold() == "http" and port == 80)
        or (parsed.scheme.casefold() == "https" and port == 443)
    ):
        return f"{hostname}:{port}"
    return hostname


def _render_report(
    rows: list[dict[str, Any]],
    *,
    units_scanned: int,
    metadata_keys: tuple[str, ...] | None,
    min_unit_count: int,
    limit: int | None,
) -> str:
    lines = [
        "# External Domain Summary",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Domains reported | {len(rows)} |",
        f"| Min unit count | {min_unit_count} |",
    ]
    if limit is not None:
        lines.append(f"| Limit | {limit} |")
    if metadata_keys is not None:
        lines.append(f"| Metadata keys | {_markdown_cell(', '.join(metadata_keys))} |")
    lines.extend(
        [
            "",
            "## Domains",
            "",
            "| Domain | Units | Sources | Locations | Examples |",
            "| --- | ---: | --- | --- | --- |",
        ]
    )
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['domain'])} | "
                f"{row['unit_count']} | "
                f"{_markdown_cell(row['source_project_counts'])} | "
                f"{_markdown_cell(row['location_counts'])} | "
                f"{_markdown_cell(row['examples'])} |"
            )
    else:
        lines.append("| _None_ | 0 | _None_ | _None_ | _None_ |")
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
    return _inline_text(unit.id) or _inline_text(unit.source_id) or _unit_label(unit)


def _unit_label(unit: KnowledgeUnit) -> str:
    title = _inline_text(unit.title)
    if title:
        return title
    source_id = _inline_text(unit.source_id)
    return source_id or _unit_id(unit)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (_unit_source(unit), _inline_text(unit.source_id), _inline_text(unit.title), _inline_text(unit.id))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
