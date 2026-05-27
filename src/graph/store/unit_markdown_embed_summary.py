"""Summarize Markdown and Obsidian embeds in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any

from graph.export._report_csv import get

_OBSIDIAN_RE = re.compile(r"!\[\[([^\[\]\n]+)\]\]")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def summarize_unit_markdown_embeds(units: Iterable[Any]) -> dict[str, Any]:
    total = units_with = missing = 0
    syntax_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    for unit in units:
        found = False
        for line in str(get(unit, "content") or "").splitlines():
            for match in _OBSIDIAN_RE.finditer(line):
                total += 1; found = True; syntax_counts["obsidian"] += 1
                if not _caption(match.group(1)):
                    missing += 1
                ext = _extension(match.group(1))
                if ext:
                    extension_counts[ext] += 1
            for match in _MARKDOWN_IMAGE_RE.finditer(line):
                total += 1; found = True; syntax_counts["markdown_image"] += 1
                if not match.group(1).strip():
                    missing += 1
                ext = _extension(match.group(2))
                if ext:
                    extension_counts[ext] += 1
        if found:
            units_with += 1
    return {"total_embeds": total, "units_with_embeds": units_with, "syntax_counts": dict(sorted(syntax_counts.items())), "extension_counts": dict(sorted(extension_counts.items())), "missing_alt_or_caption_count": missing}


def _caption(value: str) -> str:
    return value.split("|", 1)[1].strip() if "|" in value else ""


def _extension(value: str) -> str:
    target = value.split("|", 1)[0].split("#", 1)[0].split("?", 1)[0]
    suffix = PurePosixPath(target).suffix.lower()
    return suffix[1:] if suffix else ""
