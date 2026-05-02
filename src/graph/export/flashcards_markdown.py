"""Markdown flashcard export helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_ANCHOR_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_units_to_flashcards_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    question_field: str = "title",
    answer_field: str = "content",
    include_tags: bool = True,
) -> dict:
    """Write units as one deterministic Markdown flashcard section per unit."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    exported_units = (
        all_units
        if isinstance(units, Sequence)
        else sorted(all_units, key=_unit_sort_key)
    )
    text = _render_flashcards_markdown(
        exported_units,
        question_field=question_field,
        answer_field=answer_field,
        include_tags=include_tags,
    )
    output_path.write_text(text, encoding="utf-8")

    return {
        "path": str(output_path),
        "units_exported": len(exported_units),
        "cards_exported": len(exported_units),
    }


def _render_flashcards_markdown(
    units: list[KnowledgeUnit],
    *,
    question_field: str,
    answer_field: str,
    include_tags: bool,
) -> str:
    cards = [
        _card_record(
            unit,
            question_field=question_field,
            answer_field=answer_field,
        )
        for unit in units
    ]
    _assign_unique_anchors(cards)

    lines = ["# Flashcards", "", "## Index", ""]
    if cards:
        for card in cards:
            lines.append(f"- [{_link_text(card['question'])}](#{card['anchor']})")
    else:
        lines.append("_No flashcards exported._")

    lines.extend(["", "## Cards", ""])
    if not cards:
        lines.extend(["_No flashcards exported._", ""])

    for card in cards:
        lines.extend(_card_section_lines(card, include_tags=include_tags))

    return "\n".join(lines).rstrip() + "\n"


def _card_record(
    unit: KnowledgeUnit,
    *,
    question_field: str,
    answer_field: str,
) -> dict[str, Any]:
    question = _field_markdown_text(
        _resolve_field(unit, question_field),
        fallback=_field_markdown_text(unit.title, fallback="Untitled"),
    )
    answer = _field_markdown_text(
        _resolve_field(unit, answer_field),
        fallback=_field_markdown_text(unit.content, fallback=""),
    )
    return {
        "unit": unit,
        "question": question or "Untitled",
        "answer": answer,
        "anchor_base": _unit_anchor_base(unit),
    }


def _assign_unique_anchors(cards: list[dict[str, Any]]) -> None:
    seen: dict[str, int] = {}
    for card in cards:
        base = card["anchor_base"]
        count = seen.get(base, 0) + 1
        seen[base] = count
        card["anchor"] = base if count == 1 else f"{base}-{count}"


def _card_section_lines(card: dict[str, Any], *, include_tags: bool) -> list[str]:
    unit = card["unit"]
    lines = [
        f'<a id="{card["anchor"]}"></a>',
        "",
        f"### {_heading_text(card['question'])}",
        "",
        "**Question**",
        "",
        _fenced_text(card["question"]),
        "",
        "**Answer**",
        "",
        _fenced_text(card["answer"]),
        "",
    ]
    if include_tags:
        lines.extend(["**Tags**", "", _tags_text(unit.tags), ""])
    lines.extend(
        [
            "**Source**",
            "",
            f"- Project: `{_code_text(_field_value(unit.source_project))}`",
            f"- ID: `{_code_text(unit.source_id)}`",
            "",
        ]
    )
    return lines


def _resolve_field(unit: KnowledgeUnit, field_path: str) -> Any:
    path = _inline_text(field_path)
    if not path:
        return None

    parts = path.split(".")
    if parts[0] == "metadata":
        return _traverse(unit.metadata, parts[1:])

    value = getattr(unit, parts[0], None)
    if value is not None:
        found = _traverse(value, parts[1:])
        if found is not None:
            return found

    if len(parts) > 1:
        return _traverse(unit.metadata, parts)

    return None


def _traverse(value: Any, parts: list[str]) -> Any:
    current = value
    for part in parts:
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(part)
            continue
        if isinstance(current, Sequence) and not isinstance(current, str):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
            continue
        current = getattr(current, part, None)
    return current


def _field_markdown_text(value: Any, *, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        text = value.strip()
        return text or fallback
    text = _markdown_value(value).strip()
    return text or fallback


def _markdown_value(value: Any) -> str:
    normalized = _json_value(value)
    if isinstance(normalized, str):
        return normalized
    if normalized is None:
        return ""
    if isinstance(normalized, int | float | bool):
        return str(normalized)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_value(item) for item in value]
    return str(value)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _field_value(unit.source_project),
        _inline_text(unit.source_id),
        _inline_text(unit.title),
    )


def _unit_anchor_base(unit: KnowledgeUnit) -> str:
    source = "-".join(
        part
        for part in [
            "card",
            _field_value(unit.source_project),
            _inline_text(unit.source_id),
        ]
        if part
    )
    text = _ANCHOR_RE.sub("-", source).strip("-").lower()
    return text or "card-unit"


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _heading_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("#", r"\#")


def _link_text(value: object) -> str:
    text = _inline_text(value)
    return (
        text.replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _code_text(value: object) -> str:
    return _inline_text(value).replace("`", r"\`")


def _tags_text(tags: list[str]) -> str:
    normalized = {_inline_text(tag) for tag in tags if _inline_text(tag)}
    if not normalized:
        return "_None._"
    return ", ".join(f"`{_code_text(tag)}`" for tag in sorted(normalized, key=_tag_key))


def _tag_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _fenced_text(value: str) -> str:
    longest_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", value)),
        default=0,
    )
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}\n{value}\n{fence}"
