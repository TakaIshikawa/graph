"""Summarize attachments that are not referenced by unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from graph.export._report_csv import get, sort_key, unit_id

_ATTACHMENT_KEYS = {"attachment", "attachments", "file", "files", "path", "paths"}
_ATTACHMENT_VALUE_KEYS = ("path", "name", "url")
_MARKDOWN_TARGET_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_WIKI_EMBED_RE = re.compile(r"!\[\[([^\]#|]+)")


def summarize_unit_attachment_missing_references(units: Iterable[Mapping[str, Any] | object], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    unit_list = list(units)
    counts_by_extension: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    units_with_unreferenced = 0
    unreferenced_count = 0

    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        references = {_reference_key(target) for target in _content_references(str(get(unit, "content") or ""))}
        references.discard("")
        unreferenced = [attachment for attachment in _attachments(unit) if not _is_referenced(attachment, references)]
        if unreferenced:
            units_with_unreferenced += 1
        for attachment in unreferenced:
            unreferenced_count += 1
            extension = _extension(attachment)
            counts_by_extension[extension] += 1
            if len(samples) < limit:
                samples.append({"unit_id": uid, "attachment": attachment, "extension": extension})

    return {
        "unit_count": len(unit_list),
        "units_with_unreferenced_attachments_count": units_with_unreferenced,
        "unreferenced_attachment_count": unreferenced_count,
        "counts_by_extension": {extension: counts_by_extension[extension] for extension in sorted(counts_by_extension, key=lambda value: (-counts_by_extension[value], sort_key(value)))},
        "samples": samples,
    }


def _attachments(unit: Mapping[str, Any] | object) -> list[str]:
    attachments: list[str] = []
    attachments.extend(_attachment_values(get(unit, "attachments")))
    metadata = get(unit, "metadata")
    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if str(key).casefold() in _ATTACHMENT_KEYS:
                attachments.extend(_attachment_values(value))
    return list(dict.fromkeys(attachment for attachment in attachments if attachment))


def _attachment_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        direct = [_text(value.get(key)) for key in _ATTACHMENT_VALUE_KEYS if _text(value.get(key))]
        if direct:
            return direct
        return [item for child in value.values() for item in _attachment_values(child)]
    if isinstance(value, list | tuple | set):
        return [item for child in value for item in _attachment_values(child)]
    return [_text(value)]


def _content_references(content: str) -> list[str]:
    references = [match.group(1) for match in _MARKDOWN_TARGET_RE.finditer(content)]
    references.extend(match.group(1) for match in _WIKI_EMBED_RE.finditer(content))
    return references


def _is_referenced(attachment: str, references: set[str]) -> bool:
    attachment_key = _reference_key(attachment)
    attachment_name = _basename(attachment_key)
    return attachment_key in references or bool(attachment_name and attachment_name in references)


def _reference_key(value: str) -> str:
    parsed = urlparse(_text(value))
    path = parsed.path if parsed.scheme else _text(value)
    return unquote(path).lstrip("./").casefold()


def _basename(path: str) -> str:
    return PurePosixPath(path).name.casefold()


def _extension(path: str) -> str:
    suffix = PurePosixPath(urlparse(path).path).suffix.casefold().lstrip(".")
    return suffix or "(none)"


def _text(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""
