"""CSV export for likely tag alias candidates."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["canonical_tag", "variant_tag", "unit_count", "source_count", "reason"]
_TAG_KEYS = ("tags", "tag")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\W_]+")


def export_tag_alias_candidates_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write likely duplicate tag groups."""
    unit_list = list(units)
    rows = _candidate_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {"path": str(output_path), "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": output_path.stat().st_size}


def _candidate_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str | int]]:
    groups: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: defaultdict(lambda: {"units": set(), "sources": set()}))
    for unit in units:
        unit_id = _unit_id(unit)
        source = _field_value(_get(unit, "source_project")) or "Unknown"
        for tag in _unit_tags(unit):
            groups[_alias_key(tag)][tag]["units"].add(unit_id)
            groups[_alias_key(tag)][tag]["sources"].add(source)

    rows: list[dict[str, str | int]] = []
    for variants in groups.values():
        if len(variants) < 2:
            continue
        canonical = sorted(variants, key=lambda tag: (-len(variants[tag]["units"]), _sort_key(tag)))[0]
        for variant in sorted(variants, key=_sort_key):
            if variant == canonical:
                continue
            rows.append(
                {
                    "canonical_tag": canonical,
                    "variant_tag": variant,
                    "unit_count": len(variants[variant]["units"]),
                    "source_count": len(variants[variant]["sources"]),
                    "reason": _reason(canonical, variant),
                }
            )
    return sorted(rows, key=lambda row: (_sort_key(row["canonical_tag"]), _sort_key(row["variant_tag"])))


def _unit_tags(unit: KnowledgeUnit | Mapping[str, Any]) -> list[str]:
    values = [_get(unit, "tags")]
    metadata = _metadata(unit)
    for key in _TAG_KEYS:
        values.append(_casefold_get(metadata, key))
    return sorted({_field_value(value) for value in _flatten(values) if _field_value(value)}, key=_sort_key)


def _flatten(values: Iterable[object]) -> list[object]:
    flattened: list[object] = []
    for value in values:
        if isinstance(value, str):
            flattened.extend(part for part in value.split(","))
        elif isinstance(value, list | tuple | set):
            flattened.extend(_flatten(value))
    return flattened


def _alias_key(tag: str) -> str:
    return _PUNCT_RE.sub("", tag.casefold())


def _reason(canonical: str, variant: str) -> str:
    if canonical.casefold() == variant.casefold():
        return "case"
    if re.sub(r"[-_\s]+", "", canonical.casefold()) == re.sub(r"[-_\s]+", "", variant.casefold()):
        return "separator"
    return "punctuation"


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _unit_id(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    return _field_value(_get(unit, "id")) or _field_value(_get(unit, "source_id"))


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

