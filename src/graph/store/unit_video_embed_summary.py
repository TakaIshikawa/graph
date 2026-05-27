"""Summarize video references in unit content and metadata."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_URL_RE = re.compile(r"\bhttps?://[^\s<>()\[\]\"']+", re.IGNORECASE)
_EMBED_RE = re.compile(r"<(?:iframe|embed)\b[^>]*>", re.IGNORECASE)
_SRC_RE = re.compile(r"\bsrc=[\"']([^\"']+)[\"']", re.IGNORECASE)
_VIDEO_EXTS = {".mp4", ".mov", ".webm"}
_META_KEYS = ("url", "video_url", "embed_url", "source_url", "path", "file")


def summarize_unit_video_embeds(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total_units = units_with_video = embedded_count = linked_count = 0
    providers: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    for index, unit in enumerate(units):
        total_units += 1
        refs = _refs(unit)
        if refs:
            units_with_video += 1
        uid = unit_id(unit) or str(index)
        for target, mode, provider in refs:
            providers[provider] += 1
            embedded_count += mode == "embedded"
            linked_count += mode == "linked"
            if len(samples) < limit:
                samples.append({"unit_id": uid, "provider": provider, "mode": mode, "target": target})
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["target"])))
    return {
        "total_units": total_units,
        "units_with_video": units_with_video,
        "video_reference_count": embedded_count + linked_count,
        "embedded_count": embedded_count,
        "linked_count": linked_count,
        "provider_counts": [{"provider": key, "count": providers[key]} for key in sorted(providers, key=sort_key)],
        "samples": samples[:limit],
    }


def _refs(unit: Any) -> list[tuple[str, str, str]]:
    content = str(get(unit, "content") or metadata(unit).get("content") or "")
    refs: list[tuple[str, str, str]] = []
    for match in _EMBED_RE.finditer(content):
        src = _SRC_RE.search(match.group(0))
        target = src.group(1) if src else ""
        provider = _provider(target)
        if provider:
            refs.append((target, "embedded", provider))
    link_content = _EMBED_RE.sub("", content)
    for target in [*(m.group(1) for m in _MD_LINK_RE.finditer(link_content)), *(m.group(0) for m in _URL_RE.finditer(link_content))]:
        provider = _provider(target)
        if provider:
            refs.append((target, "linked", provider))
    for key in _META_KEYS:
        value = metadata(unit).get(key)
        values = value if isinstance(value, list | tuple | set) else [value]
        for item in values:
            target = field_value(item)
            provider = _provider(target)
            if provider:
                refs.append((target, "linked", provider))
    return sorted(dict.fromkeys(refs), key=lambda row: (sort_key(row[2]), sort_key(row[1]), sort_key(row[0])))


def _provider(target: str) -> str:
    parsed = urlparse(target)
    host = parsed.netloc.casefold()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "vimeo.com" in host:
        return "vimeo"
    if Path(parsed.path or target).suffix.casefold() in _VIDEO_EXTS:
        return "local"
    return ""
