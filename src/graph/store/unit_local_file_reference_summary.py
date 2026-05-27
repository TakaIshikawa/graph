"""Summarize local file references in Markdown units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FILE_URI_RE = re.compile(r"\bfile://[^\s<>()\[\]\"']+")


def summarize_unit_local_file_references(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_refs = total_refs = 0
    extensions: Counter[str] = Counter()
    schemes: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    top: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        total_units += 1
        refs = _refs(str(get(unit, "content") or ""))
        if refs:
            units_with_refs += 1
        total_refs += len(refs)
        extensions.update(Path(path).suffix.casefold().lstrip(".") or "(none)" for _scheme, path in refs)
        schemes.update(scheme or "(none)" for scheme, _path in refs)
        styles.update(_style(path) for _scheme, path in refs)
        top.append({"unit_id": unit_id(unit) or str(index), "title": _title(unit), "reference_count": len(refs)})
    return {
        "total_units": total_units,
        "units_with_local_references": units_with_refs,
        "local_reference_count": total_refs,
        "extensions": _counter_rows(extensions, "extension"),
        "schemes": _counter_rows(schemes, "scheme"),
        "path_styles": _counter_rows(styles, "path_style"),
        "top_units": sorted(top, key=lambda row: (-int(row["reference_count"]), sort_key(row["unit_id"])))[:limit],
    }


def _refs(content: str) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for line in content.splitlines():
        targets = list(dict.fromkeys([match.group(1) for match in _MD_LINK_RE.finditer(line)] + [match.group(0) for match in _FILE_URI_RE.finditer(line)]))
        for target in targets:
            parsed = _local_target(target)
            if parsed:
                refs.append(parsed)
    return refs


def _local_target(target: str) -> tuple[str, str] | None:
    clean = target.strip().split("#", 1)[0].split("?", 1)[0]
    parsed = urlparse(clean)
    scheme = parsed.scheme.casefold()
    if scheme in {"http", "https", "mailto", "tel"}:
        return None
    if scheme == "file":
        return ("file", unquote(parsed.path))
    if scheme:
        return None
    return ("", unquote(clean)) if clean else None


def _style(path: str) -> str:
    if Path(path).is_absolute():
        return "absolute"
    return "relative"


def _counter_rows(counter: Counter[str], key: str) -> list[dict[str, Any]]:
    return [{key: name, "count": count} for name, count in sorted(counter.items(), key=lambda item: (-item[1], sort_key(item[0])))]


def _title(unit: Any) -> str:
    return field_value(get(unit, "title") or metadata(unit).get("title"))
