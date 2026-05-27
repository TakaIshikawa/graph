"""Summarize scalar fields in leading YAML frontmatter."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key

_FIELD_RE = re.compile(r"^(\s*)([A-Za-z0-9_-]+)\s*:\s*(.*)$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:$|[T\s])")
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def summarize_unit_frontmatter_scalar_fields(units: Iterable[Any], *, sample_limit: int = 5) -> dict[str, Any]:
    """Return deterministic counts for leading frontmatter scalar fields."""
    limit = max(0, sample_limit)
    total_units = units_with_frontmatter = 0
    data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"unit_ids": set(), "blank_value_count": 0, "type_hints": Counter(), "examples": []}
    )

    for index, unit in enumerate(units):
        total_units += 1
        fields = _scalar_fields(str(get(unit, "content") or ""))
        if fields is None:
            continue
        units_with_frontmatter += 1
        uid = str(get(unit, "id") or get(unit, "unit_id") or index)
        for key_path, raw_value in fields:
            row = data[key_path]
            row["unit_ids"].add(uid)
            if not raw_value.strip():
                row["blank_value_count"] += 1
            hint = _type_hint(raw_value)
            row["type_hints"][hint] += 1
            example = _example_value(raw_value)
            if example and example not in row["examples"] and len(row["examples"]) < limit:
                row["examples"].append(example)

    rows = []
    for key_path in sorted(data, key=sort_key):
        type_hints = data[key_path]["type_hints"]
        rows.append(
            {
                "key_path": key_path,
                "unit_count": len(data[key_path]["unit_ids"]),
                "blank_value_count": data[key_path]["blank_value_count"],
                "most_common_type_hint": _most_common_type(type_hints),
                "example_values": data[key_path]["examples"],
            }
        )
    return {"total_units": total_units, "units_with_frontmatter": units_with_frontmatter, "scalar_fields": rows}


def _scalar_fields(content: str) -> list[tuple[str, str]] | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return _parse_scalar_lines(block)
        block.append(line)
    return None


def _parse_scalar_lines(lines: list[str]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    matches = [(line, _FIELD_RE.match(line)) for line in lines]

    for index, (line, match) in enumerate(matches):
        if not match or line.lstrip().startswith("-"):
            continue
        indent_text, key, value = match.groups()
        indent = len(indent_text.replace("\t", "    "))
        while stack and stack[-1][0] >= indent:
            stack.pop()

        cleaned_value = _strip_comment(value).strip()
        if _is_collection_value(cleaned_value):
            stack.append((indent, key))
            continue
        if cleaned_value == "" and _has_indented_child(matches[index + 1 :], indent):
            stack.append((indent, key))
            continue

        rows.append((".".join([*(path for _, path in stack), key]), cleaned_value))
    return rows


def _has_indented_child(items: list[tuple[str, re.Match[str] | None]], indent: int) -> bool:
    for line, match in items:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not match:
            return len(line) - len(line.lstrip(" ")) > indent
        return len(match.group(1).replace("\t", "    ")) > indent
    return False


def _strip_comment(value: str) -> str:
    quote = ""
    for index, char in enumerate(value):
        if char in {"'", '"'} and (index == 0 or value[index - 1] != "\\"):
            quote = "" if quote == char else char if not quote else quote
        if char == "#" and not quote and (index == 0 or value[index - 1].isspace()):
            return value[:index]
    return value


def _is_collection_value(value: str) -> bool:
    return value.startswith(("[", "{")) or value in {"|", ">", "|-", ">-", "|+", ">+"}


def _type_hint(value: str) -> str:
    text = value.strip()
    if not text:
        return "blank"
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return "quoted"
    lowered = text.casefold()
    if lowered in {"true", "false", "yes", "no", "on", "off"}:
        return "boolean-like"
    if _DATE_RE.match(text):
        return "date-like"
    if _NUMERIC_RE.match(text):
        return "numeric-like"
    return "string"


def _example_value(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1]
    return text


def _most_common_type(type_hints: Counter[str]) -> str:
    if not type_hints:
        return ""
    return min(type_hints, key=lambda hint: (-type_hints[hint], sort_key(hint)))
