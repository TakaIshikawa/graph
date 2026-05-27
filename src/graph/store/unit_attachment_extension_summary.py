"""Summarize attachment file extensions referenced by units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

_ATTACHMENT_KEYS = {"attachment", "attachments", "file", "files", "path", "paths"}
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def summarize_unit_attachment_extensions(units: Iterable[Any]) -> dict[str, Any]:
    extension_units: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)
    total_units = 0
    units_with = 0
    for unit in units:
        total_units += 1
        unit_id = _unit_id(unit)
        extensions = _extensions(unit)
        if extensions:
            units_with += 1
        for extension, sample in extensions:
            extension_units[extension].add(unit_id)
            if sample not in examples[extension] and len(examples[extension]) < 3:
                examples[extension].append(sample)
    extension_counts = {extension: len(extension_units[extension]) for extension in sorted(extension_units, key=_sort_key)}
    return {"extension_counts": extension_counts, "units_with_attachments": units_with, "units_without_attachments": total_units - units_with, "examples_by_extension": dict(sorted(examples.items(), key=lambda item: _sort_key(item[0])))}


def _extensions(unit: Any) -> set[tuple[str, str]]:
    paths: list[str] = []
    for key, value in _metadata(unit).items():
        if _text(key).casefold() in _ATTACHMENT_KEYS:
            paths.extend(_text(item) for item in _flatten(value) if _text(item))
    content = str(_get(unit, "content") or "")
    for match in _MARKDOWN_LINK_RE.finditer(content):
        target = _text(match.group(1).split()[0])
        if target and not urlparse(target).scheme:
            paths.append(target)
    return {(_extension(path), path) for path in paths}


def _extension(path: str) -> str:
    clean = urlparse(path).path
    suffix = PurePosixPath(clean).suffix.casefold().lstrip(".")
    return suffix or "(none)"


def _metadata(unit: Any) -> Mapping[str, Any]:
    value = _get(unit, "metadata")
    return value if isinstance(value, Mapping) else {}


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [item for child in value.values() for item in _flatten(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _flatten(child)]
    return [value]


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _unit_id(unit: Any) -> str:
    return _text(_get(unit, "id") or _get(unit, "unit_id"))


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _sort_key(value: Any) -> tuple[str, str]:
    text = _text(value)
    return (text.casefold(), text)
