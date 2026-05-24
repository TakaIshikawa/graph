"""CSV export for source license and rights metadata coverage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, normalized_key, render_csv, sort_key, source_id, write_csv

_FIELDNAMES = ["license", "normalized_license", "count", "source_ids", "rights_keys", "status"]
_RIGHTS_KEYS = {"license", "rights", "copyright", "usage_terms", "terms_url"}


def export_source_license_coverage_csv(
    sources: Iterable[Mapping[str, Any] | object],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write license coverage grouped by normalized rights value."""
    source_list = list(sources)
    rows = _rows(source_list)
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "source_count": len(source_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(sources: list[Mapping[str, Any] | object]) -> list[dict[str, str | int]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(lambda: {"count": 0, "source_ids": set(), "rights_keys": set()})
    for source in sources:
        values = _rights_values(source)
        if not values:
            values = [("", "missing", "missing")]
        for license_text, key, status in values:
            normalized = _normalize_license(license_text) if license_text else "missing"
            bucket = groups[(license_text or "missing", normalized, status)]
            bucket["count"] += 1
            if source_id(source):
                bucket["source_ids"].add(source_id(source))
            if key:
                bucket["rights_keys"].add(key)
    rows: list[dict[str, str | int]] = []
    for license_text, normalized, status in sorted(groups, key=lambda key: (sort_key(key[2]), sort_key(key[1]), sort_key(key[0]))):
        bucket = groups[(license_text, normalized, status)]
        rows.append(
            {
                "license": license_text,
                "normalized_license": normalized,
                "count": bucket["count"],
                "source_ids": "; ".join(sorted(bucket["source_ids"], key=sort_key)),
                "rights_keys": "; ".join(sorted(bucket["rights_keys"], key=sort_key)),
                "status": status,
            }
        )
    return rows


def _rights_values(source: Mapping[str, Any] | object) -> list[tuple[str, str, str]]:
    values: list[tuple[str, str, str]] = []
    for key in _RIGHTS_KEYS:
        text = field_value(get(source, key))
        if text:
            values.append((text, key, _status(key, text)))
    for key, value in metadata(source).items():
        key_text = field_value(key)
        if normalized_key(key_text) in _RIGHTS_KEYS:
            for item in flatten_values(value):
                text = field_value(item)
                if text:
                    values.append((text, key_text, _status(key_text, text)))
    return values


def _normalize_license(value: str) -> str:
    text = value.casefold().strip()
    compact = text.replace("-", "").replace("_", "").replace(" ", "").replace(".", "")
    if "creativecommons.org/licenses/by/" in text or compact in {"ccby", "ccby40"}:
        return "cc-by"
    if "creativecommons.org/licenses/by-sa/" in text or compact.startswith("ccbysa"):
        return "cc-by-sa"
    if "creativecommons.org/publicdomain/zero" in text or compact in {"cc0", "cczero"}:
        return "cc0"
    if compact in {"mit", "mitlicense"}:
        return "mit"
    if compact.startswith("apache20") or "apache license" in text:
        return "apache-2.0"
    if "public domain" in text:
        return "public-domain"
    if "copyright" in text or "all rights reserved" in text:
        return "copyright"
    if text.startswith("http://") or text.startswith("https://"):
        return "terms-url"
    return compact or "unknown"


def _status(key: str, value: str) -> str:
    normalized = normalized_key(key)
    text = value.casefold()
    if normalized in {"copyright", "rights"} and ("copyright" in text or "all rights reserved" in text):
        return "restricted"
    if normalized == "terms_url" or "terms" in text or value.startswith(("http://", "https://")):
        return "ambiguous"
    return "present"
