"""Summarize duplicate unit slugs."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id


def summarize_unit_duplicate_slugs(units: Iterable[Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for unit in units:
        slug = field_value(get(unit, "slug") or metadata(unit).get("slug")) or _slugify(get(unit, "title") or metadata(unit).get("title"))
        if not slug:
            continue
        normalized = _slugify(slug)
        groups[normalized].append({"unit_id": unit_id(unit), "title": field_value(get(unit, "title") or metadata(unit).get("title")), "slug": normalized})
    duplicates = {slug: sorted(items, key=lambda item: sort_key(item["unit_id"])) for slug, items in sorted(groups.items()) if len(items) > 1}
    return {"duplicate_slug_count": len(duplicates), "affected_unit_count": sum(len(items) for items in duplicates.values()), "groups": duplicates}


def _slugify(value: object) -> str:
    text = field_value(value).casefold()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")
