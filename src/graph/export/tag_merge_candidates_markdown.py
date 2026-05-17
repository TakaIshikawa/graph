"""Markdown export for likely tag merge candidates."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^0-9A-Za-z]+")


def export_tag_merge_candidates_markdown(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write a Markdown report of likely tag merge candidates."""
    unit_list = list(units)
    sections = _candidate_sections(unit_list)
    text = _render_markdown(sections)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "candidate_group_count": len(sections),
        "bytes_written": output_path.stat().st_size,
    }


def _candidate_sections(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, object]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"variants": defaultdict(set), "counts": Counter()})

    for unit in units:
        unit_id = _unit_id(unit)
        for tag in _unit_tags(unit):
            normalized = _normalize_tag(tag)
            if not normalized:
                continue
            groups[normalized]["variants"][tag].add(unit_id)
            groups[normalized]["counts"][tag] += 1

    sections: list[dict[str, object]] = []
    for normalized, data in groups.items():
        variants: dict[str, set[str]] = data["variants"]
        if len(variants) < 2:
            continue
        sections.append(
            {
                "normalized": normalized,
                "canonical": _suggested_canonical(data["counts"]),
                "variants": [
                    {
                        "tag": tag,
                        "unit_count": len(variants[tag]),
                        "example_unit_ids": sorted(variants[tag], key=_sort_key)[:5],
                    }
                    for tag in sorted(variants, key=_sort_key)
                ],
            }
        )

    return sorted(sections, key=lambda section: _sort_key(section["normalized"]))


def _render_markdown(sections: list[dict[str, object]]) -> str:
    lines = ["# Tag Merge Candidates", ""]
    if not sections:
        lines.extend(["No tag merge candidates found.", ""])
        return "\n".join(lines)

    for section in sections:
        lines.append(f"## {section['normalized']}")
        lines.append("")
        lines.append(f"- Suggested canonical tag: `{section['canonical']}`")
        lines.append("- Raw variants:")
        for variant in section["variants"]:
            lines.append(
                "- "
                f"`{variant['tag']}` - {variant['unit_count']} unit(s); "
                f"examples: {_example_ids(variant['example_unit_ids'])}"
            )
        lines.append("")
    return "\n".join(lines)


def _suggested_canonical(counts: Counter[str]) -> str:
    return sorted(counts, key=lambda tag: (-counts[tag], len(tag), _sort_key(tag)))[0]


def _example_ids(unit_ids: list[str]) -> str:
    return ", ".join(f"`{unit_id}`" for unit_id in unit_ids if unit_id) or "(none)"


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    tags = _get(unit, "tags")
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Iterable):
        return []
    return sorted({_field_value(tag) for tag in tags if _field_value(tag)}, key=_sort_key)


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


def _normalize_tag(value: object) -> str:
    raw = _field_value(value)
    if raw.isalpha() and raw.isupper() and 1 < len(raw) <= 6:
        return " ".join(raw.casefold())
    text = _PUNCTUATION_RE.sub(" ", raw).strip().casefold()
    text = _WHITESPACE_RE.sub(" ", text)
    words = [_singularize(word) for word in text.split(" ") if word]
    return " ".join(words)


def _singularize(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _get(value: object, key: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
