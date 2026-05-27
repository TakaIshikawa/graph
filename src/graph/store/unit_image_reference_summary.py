"""Inventory markdown and HTML image references in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from urllib.parse import urlparse
from typing import Any

from graph.export._report_csv import get, metadata, sort_key

_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_IMG_TAG_RE = re.compile(r"<img\b([^>]*)>", re.I)
_ATTR_RE = re.compile(r"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(['\"])(.*?)\2")


def summarize_unit_image_references(units: Iterable[Any]) -> dict[str, Any]:
    total_units = total_images = markdown_image_count = html_image_count = 0
    remote_image_count = local_image_count = populated_alt_text_count = blank_alt_text_count = missing_alt_text_count = 0
    scheme_counts = Counter()
    extension_counts = Counter()

    for unit in units:
        total_units += 1
        for src, alt, kind in _references(_content(unit)):
            total_images += 1
            markdown_image_count += int(kind == "markdown")
            html_image_count += int(kind == "html")
            parsed = urlparse(src)
            scheme = parsed.scheme.casefold() if parsed.scheme else "local"
            scheme_counts[scheme] += 1
            remote_image_count += int(scheme in {"http", "https", "data"})
            local_image_count += int(scheme not in {"http", "https", "data"})
            extension = _extension(parsed.path or src)
            if extension:
                extension_counts[extension] += 1
            if alt is None:
                missing_alt_text_count += 1
            elif alt.strip():
                populated_alt_text_count += 1
            else:
                blank_alt_text_count += 1

    return {
        "total_units": total_units,
        "total_images": total_images,
        "markdown_image_count": markdown_image_count,
        "html_image_count": html_image_count,
        "remote_image_count": remote_image_count,
        "local_image_count": local_image_count,
        "populated_alt_text_count": populated_alt_text_count,
        "blank_alt_text_count": blank_alt_text_count,
        "missing_alt_text_count": missing_alt_text_count,
        "scheme_counts": [{"scheme": key, "count": scheme_counts[key]} for key in sorted(scheme_counts, key=sort_key)],
        "extension_counts": [
            {"extension": key, "count": extension_counts[key]} for key in sorted(extension_counts, key=sort_key)
        ],
    }


def _references(content: str) -> list[tuple[str, str | None, str]]:
    refs = [(match.group(2), match.group(1), "markdown") for match in _MD_IMAGE_RE.finditer(content)]
    for tag in _IMG_TAG_RE.finditer(content):
        attrs = {name.casefold(): value for name, _quote, value in _ATTR_RE.findall(tag.group(1))}
        if attrs.get("src"):
            refs.append((attrs["src"], attrs.get("alt"), "html"))
    return refs


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].casefold()


def _content(unit: Any) -> str:
    if isinstance(unit, str):
        return unit
    value = get(unit, "content") or metadata(unit).get("content")
    return "" if value is None else str(value)
