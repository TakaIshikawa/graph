"""Summarize Markdown reference-style link text in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_CODE_SPAN_RE = re.compile(r"`+[^`]*`+")
_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[[^\]\n]+]:")
_FULL_OR_COLLAPSED_RE = re.compile(r"(?<!!)\[([^\]\n]+)]\[([^\]\n]*)]")
_SHORTCUT_RE = re.compile(r"(?<!!)(?<!])\[([^\]\n]+)](?![\[(])")


def summarize_unit_markdown_reference_link_texts(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    grouped: dict[str, dict[str, Any]] = {}
    for unit in units:
        total += 1
        uid = unit_id(unit)
        seen_in_unit: set[str] = set()
        for line_number, label, text, usage_type in _reference_links(str(get(unit, "content") or "")):
            normalized = _normalize(label)
            row = grouped.setdefault(
                normalized,
                {"label": normalized, "use_count": 0, "unit_ids": set(), "link_texts": set(), "examples": []},
            )
            row["use_count"] += 1
            row["link_texts"].add(field_value(text))
            seen_in_unit.add(normalized)
            if len(row["examples"]) < limit:
                row["examples"].append({"unit_id": uid, "line_number": line_number, "link_text": field_value(text), "usage_type": usage_type})
        for label in seen_in_unit:
            grouped[label]["unit_ids"].add(uid)

    reference_labels = []
    for row in grouped.values():
        examples = sorted(row["examples"], key=lambda sample: (sort_key(sample["unit_id"]), int(sample["line_number"]), sort_key(sample["link_text"])))
        reference_labels.append(
            {
                "label": row["label"],
                "use_count": row["use_count"],
                "unit_count": len(row["unit_ids"]),
                "link_texts": sorted(row["link_texts"], key=sort_key),
                "examples": examples[:limit],
            }
        )
    reference_labels.sort(key=lambda row: (-int(row["use_count"]), sort_key(row["label"])))
    return {"total_units": total, "reference_labels": reference_labels}


def _reference_links(content: str) -> list[tuple[int, str, str, str]]:
    rows: list[tuple[int, str, str, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _REF_DEF_RE.match(line):
            continue
        masked = _CODE_SPAN_RE.sub(lambda match: " " * (match.end() - match.start()), line)
        consumed = list(masked)
        for match in _FULL_OR_COLLAPSED_RE.finditer(masked):
            text = field_value(match.group(1))
            raw_label = field_value(match.group(2))
            label = raw_label or text
            usage_type = "collapsed" if not raw_label else "full"
            if label and text:
                rows.append((line_number, label, text, usage_type))
            consumed[match.start() : match.end()] = " " * (match.end() - match.start())
        for match in _SHORTCUT_RE.finditer("".join(consumed)):
            label = field_value(match.group(1))
            if label:
                rows.append((line_number, label, label, "shortcut"))
    return rows


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", field_value(value)).casefold()
