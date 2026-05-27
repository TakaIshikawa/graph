"""Summarize attributes in Markdown code fence info strings."""

from __future__ import annotations

import re
import shlex
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*(?P<info>.*)$")
_KEY_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z_][\w-]*)=(?P<value>.+)$")
_BRACE_RE = re.compile(r"\{(?P<body>[^}]*)\}")


def summarize_unit_code_fence_info_attributes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = 0
    attr_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    total_with_attrs = 0
    examples = []
    for unit in units:
        total_units += 1
        in_fence = False
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            match = _FENCE_RE.match(line)
            if not match:
                continue
            if in_fence:
                in_fence = False
                continue
            in_fence = True
            language, attrs = _parse_info(match.group("info"))
            if language:
                lang_counts[language] += 1
            if attrs:
                total_with_attrs += 1
                attr_counts.update(attrs)
                if len(examples) < sample_limit:
                    examples.append({"unit_id": unit_id(unit), "line": line_number, "language": language, "attributes": sorted(attrs, key=sort_key)})
    return {
        "total_units": total_units,
        "total_fences_with_attributes": total_with_attrs,
        "attribute_counts": _counter_rows(attr_counts, "attribute"),
        "language_counts": _counter_rows(lang_counts, "language"),
        "examples": examples,
    }


def _parse_info(info: str) -> tuple[str, list[str]]:
    brace_bodies = _BRACE_RE.findall(info)
    remaining = _BRACE_RE.sub(" ", info)
    tokens = shlex.split(remaining, posix=True) if remaining.strip() else []
    language = field_value(tokens[0]).casefold() if tokens and not tokens[0].startswith(("{", ".", "#")) and "=" not in tokens[0] else ""
    attrs: list[str] = []
    for body in brace_bodies:
        for part in shlex.split(body, posix=True):
            if part.startswith("#"):
                attrs.append("id")
            elif part.startswith("."):
                attrs.append("class")
            else:
                key_match = _KEY_VALUE_RE.match(part)
                if key_match:
                    attrs.append(_normalize_attr(key_match.group("key")))
    for token in tokens[1 if language else 0 :]:
        key_match = _KEY_VALUE_RE.match(token)
        if key_match:
            attrs.append(_normalize_attr(key_match.group("key")))
    return language, attrs


def _normalize_attr(key: str) -> str:
    normalized = field_value(key).casefold().replace("-", "_")
    return "filename" if normalized in {"file", "filepath", "path"} else normalized


def _counter_rows(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "count": count} for key, count in sorted(counter.items(), key=lambda item: (-item[1], sort_key(item[0])))]
