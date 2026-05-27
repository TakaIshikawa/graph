"""Summarize PDF references in unit content and metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_BARE_URL_RE = re.compile(r"\bhttps?://[^\s<>()\[\]\"']+", re.IGNORECASE)
_PDF_TARGET_RE = re.compile(r"\.pdf(?:$|[?#])", re.IGNORECASE)
_PAGE_RE = re.compile(r"(?:^|[&#])page=(\d+)(?:$|&)", re.IGNORECASE)
_METADATA_KEYS = ("url", "urls", "source_url", "path", "paths", "file", "filepath", "source_path")


def summarize_unit_pdf_references(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    """Return deterministic counts for PDF references across units."""
    limit = max(0, sample_limit)
    total_units = units_with_pdf = remote_pdf_url_count = local_pdf_path_count = page_fragment_count = 0
    samples: list[dict[str, str]] = []

    for index, unit in enumerate(units):
        total_units += 1
        refs = _references(unit)
        if not refs:
            continue
        units_with_pdf += 1
        uid = unit_id(unit) or str(index)
        for target in refs:
            if _is_remote(target):
                remote_pdf_url_count += 1
            else:
                local_pdf_path_count += 1
            if _PAGE_RE.search(target):
                page_fragment_count += 1
            if len(samples) < limit:
                samples.append({"unit_id": uid, "target": target})

    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["target"])))
    return {
        "total_units": total_units,
        "units_with_pdf_references": units_with_pdf,
        "pdf_reference_count": remote_pdf_url_count + local_pdf_path_count,
        "remote_pdf_url_count": remote_pdf_url_count,
        "local_pdf_path_count": local_pdf_path_count,
        "page_fragment_reference_count": page_fragment_count,
        "samples": samples[:limit],
    }


def _references(unit: Any) -> list[str]:
    refs: list[str] = []
    content = field_value(get(unit, "content") or metadata(unit).get("content"))
    for target in [*(match.group(1) for match in _MD_LINK_RE.finditer(content)), *(match.group(0) for match in _BARE_URL_RE.finditer(content))]:
        _append_pdf(refs, target)
    for value in _metadata_values(metadata(unit)):
        _append_pdf(refs, field_value(value))
    return sorted(dict.fromkeys(refs), key=sort_key)


def _metadata_values(meta: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in _METADATA_KEYS:
        raw = meta.get(key)
        if isinstance(raw, list | tuple | set):
            values.extend(raw)
        elif raw is not None:
            values.append(raw)
    return values


def _append_pdf(refs: list[str], target: str) -> None:
    clean = target.strip().rstrip(".,;")
    if clean and _PDF_TARGET_RE.search(clean):
        refs.append(clean)


def _is_remote(target: str) -> bool:
    return urlparse(target).scheme.casefold() in {"http", "https"}
