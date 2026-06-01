"""Summarize target extensions from Markdown links and images."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(!?)\[[^\]\n]*]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


def summarize_unit_markdown_link_target_extensions(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize extensions for local Markdown link and image targets."""
    limit = max(0, sample_limit)
    total = target_count = external = extensionless = 0
    extension_counts: Counter[str] = Counter()
    examples: list[dict[str, str | int | bool]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, target, is_image in _targets(str(get(unit, "content") or "")):
            target_count += 1
            kind, extension = _classify(target)
            if kind == "external":
                external += 1
            elif kind == "extensionless":
                extensionless += 1
            else:
                extension_counts[extension] += 1
            examples.append({"unit_id": uid, "line": line_number, "target": target, "extension": extension, "is_image": is_image})
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["target"])))
    return {
        "total_units": total,
        "target_count": target_count,
        "extension_counts": {key: extension_counts[key] for key in sorted(extension_counts, key=sort_key)},
        "extensionless_count": extensionless,
        "external_url_count": external,
        "local_path_count": target_count - external,
        "examples": examples[:limit],
    }


def _targets(content: str) -> list[tuple[int, str, bool]]:
    rows: list[tuple[int, str, bool]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            rows.append((line_number, field_value(match.group(2)), bool(match.group(1))))
    return rows


def _classify(target: str) -> tuple[str, str]:
    parsed = urlsplit(target)
    if parsed.scheme in {"http", "https"} or (_SCHEME_RE.match(target) and parsed.scheme not in {"", "file"}):
        return ("external", "")
    path = parsed.path if parsed.scheme else target.split("#", 1)[0].split("?", 1)[0]
    suffix = PurePosixPath(path).suffix.casefold().lstrip(".")
    return ("extension", suffix) if suffix else ("extensionless", "")
