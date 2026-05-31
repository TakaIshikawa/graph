"""Summarize orphan and missing unit attachment files."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, flatten_values, get, metadata, sort_key, unit_id

_MD_REF_RE = re.compile(r"!?\[[^\]\n]*\]\((?![a-z][a-z0-9+.-]*:)([^)\n]+)\)", re.IGNORECASE)
_WIKI_FILE_RE = re.compile(r"!\[\[([^\]\n]+)\]\]")


def summarize_unit_attachment_orphan_files(units: Iterable[Any], attachment_file_paths: Iterable[str | Path], sample_limit: int = 5) -> dict[str, Any]:
    available = {_norm(path) for path in attachment_file_paths if _norm(path)}
    refs: dict[str, set[str]] = {}
    for unit in units:
        uid = unit_id(unit)
        for ref in _references(unit):
            refs.setdefault(ref, set()).add(uid)
    referenced = set(refs)
    orphan = sorted(available - referenced, key=sort_key)
    missing = sorted(referenced - available, key=sort_key)
    return {"referenced_count": len(referenced), "available_count": len(available), "orphan_files": orphan, "missing_references": missing, "samples": [{"file": value, "unit_ids": sorted(refs.get(value, []), key=sort_key)} for value in missing[:sample_limit]] + [{"file": value, "unit_ids": []} for value in orphan[:sample_limit]]}


def _references(unit: Any) -> set[str]:
    refs: set[str] = set()
    content = str(get(unit, "content") or "")
    for match in _MD_REF_RE.finditer(content):
        refs.add(_norm(match.group(1).split()[0]))
    for match in _WIKI_FILE_RE.finditer(content):
        refs.add(_norm(match.group(1).partition("|")[0]))
    meta = metadata(unit)
    for key in ("attachments", "attachment", "files", "file"):
        for value in flatten_values(meta.get(key)):
            normalized = _norm(field_value(value))
            if normalized:
                refs.add(normalized)
    return {ref for ref in refs if ref}


def _norm(value: Any) -> str:
    text = field_value(value).strip("<>")
    return Path(text).name if text else ""
