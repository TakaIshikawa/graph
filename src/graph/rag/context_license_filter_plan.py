"""Plan context use according to license and reuse constraints."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, metadata, result_id, string, value

_LICENSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cc-by-nc", re.compile(r"\b(?:cc[- ]?by[- ]?nc|creative commons attribution noncommercial)\b", re.I)),
    ("cc-by", re.compile(r"\b(?:cc[- ]?by|creative commons attribution)\b", re.I)),
    ("public-domain", re.compile(r"\b(?:public domain|cc0|unlicense)\b", re.I)),
    ("all-rights-reserved", re.compile(r"\b(?:all rights reserved|copyright)\b", re.I)),
)

_LICENSE_KEYS = ("license", "rights", "copyright", "usage", "reuse")


def plan_context_license_filter(context_items: Iterable[Any]) -> dict[str, Any]:
    """Classify context items by license and return a conservative use plan."""
    rows = [_classify(item, index) for index, item in enumerate(context_items or [])]
    return {
        "items": rows,
        "allowed_item_ids": [row["item_id"] for row in rows if "answer_generation" in row["allowed_uses"]],
        "excluded_item_ids": [row["item_id"] for row in rows if row["exclude_from_redistribution"]],
        "warnings": sorted({warning for row in rows for warning in row["cautions"]}),
    }


def _classify(item: Any, index: int) -> dict[str, Any]:
    text = _license_text(item)
    license_type = _license_type(text)
    allowed_uses, cautions, excluded = _policy(license_type)
    return {
        "item_id": result_id(item, index),
        "license": license_type,
        "allowed_uses": allowed_uses,
        "cautions": cautions,
        "exclude_from_redistribution": excluded,
    }


def _license_text(item: Any) -> str:
    parts = []
    for key in _LICENSE_KEYS:
        direct = string(value(item, key))
        if direct:
            parts.append(direct)
    parts.extend(iter_strings(metadata(item)))
    if not parts:
        parts.append(content_text(item))
    return " ".join(parts)


def _license_type(text: str) -> str:
    for name, pattern in _LICENSE_PATTERNS:
        if pattern.search(text):
            return name
    return "unknown"


def _policy(license_type: str) -> tuple[list[str], list[str], bool]:
    if license_type == "public-domain":
        return ["answer_generation", "quotation", "redistribution"], [], False
    if license_type == "cc-by":
        return ["answer_generation", "quotation", "redistribution"], ["attribution_required"], False
    if license_type == "cc-by-nc":
        return ["answer_generation", "quotation"], ["attribution_required", "noncommercial_only"], True
    if license_type == "all-rights-reserved":
        return ["answer_generation"], ["quote_sparingly", "no_redistribution"], True
    return ["answer_generation"], ["unknown_license", "quote_sparingly", "no_redistribution"], True
