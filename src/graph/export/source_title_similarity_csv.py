"""CSV export for likely duplicate source titles."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.types.models import KnowledgeUnit

_FIELDNAMES = [
    "normalized_title",
    "title_variants",
    "unit_count",
    "source_observation_count",
    "source_types",
    "unit_ids",
]
_TITLE_KEYS = ("title", "name", "label", "source_title")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^a-z0-9]+")
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass(frozen=True)
class _TitleObservation:
    normalized_title: str
    title: str
    unit_id: str
    source_type: str


def export_source_title_similarity_csv(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write normalized title groups with multiple observations."""
    unit_list = list(units)
    rows = _similarity_rows(unit_list)
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "unit_count": len(unit_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _similarity_rows(units: list[KnowledgeUnit]) -> list[dict[str, str | int]]:
    groups: dict[str, list[_TitleObservation]] = defaultdict(list)
    for unit in sorted(units, key=_unit_sort_key):
        for observation in _title_observations(unit):
            if observation.normalized_title:
                groups[observation.normalized_title].append(observation)

    rows: list[dict[str, str | int]] = []
    for normalized_title, observations in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        if len(observations) < 2:
            continue
        rows.append(
            {
                "normalized_title": normalized_title,
                "title_variants": _joined_unique(observation.title for observation in observations),
                "unit_count": len({_inline_text(observation.unit_id) for observation in observations}),
                "source_observation_count": len(observations),
                "source_types": _joined_unique(observation.source_type for observation in observations),
                "unit_ids": _joined_unique(observation.unit_id for observation in observations),
            }
        )
    return rows


def _title_observations(unit: KnowledgeUnit) -> list[_TitleObservation]:
    unit_id = _unit_id(unit)
    observations = [
        _TitleObservation(
            normalized_title=_normalized_title(unit.title),
            title=_inline_text(unit.title),
            unit_id=unit_id,
            source_type=_field_value(unit.source_entity_type) or "unknown",
        )
    ]
    metadata = unit.metadata if isinstance(unit.metadata, Mapping) else {}
    for source in _source_values(metadata):
        title = _source_title(source)
        if not title:
            continue
        observations.append(
            _TitleObservation(
                normalized_title=_normalized_title(title),
                title=title,
                unit_id=unit_id,
                source_type=_source_type(source),
            )
        )
    return observations


def _source_values(metadata: Mapping[str, Any]) -> list[object]:
    values: list[object] = []
    if "source" in metadata:
        values.extend(_flat_values(metadata.get("source")))
    if "sources" in metadata:
        values.extend(_flat_values(metadata.get("sources")))
    return values


def _flat_values(value: object) -> list[object]:
    if isinstance(value, list | tuple | set):
        return [item for entry in value for item in _flat_values(entry)]
    return [value]


def _source_title(value: object) -> str:
    if isinstance(value, Mapping):
        return _metadata_text(value, _TITLE_KEYS)
    return _inline_text(value)


def _source_type(value: object) -> str:
    if isinstance(value, Mapping):
        return _metadata_text(value, ("source_type", "type", "kind")) or "unknown"
    return "unknown"


def _metadata_text(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = _inline_text(metadata.get(key))
        if text:
            return text
    return ""


def _normalized_title(value: object) -> str:
    text = _inline_text(value).casefold()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        text = f"{parsed.netloc} {parsed.path}"
    text = _URL_RE.sub(" ", text)
    text = text.removeprefix("www.")
    text = _PUNCTUATION_RE.sub(" ", text)
    words = [word for word in text.split() if word not in {"http", "https", "www"}]
    return " ".join(words)


def _joined_unique(values: Iterable[object]) -> str:
    return "; ".join(sorted({_inline_text(value) for value in values if _inline_text(value)}, key=_sort_key))


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[tuple[str, str], tuple[str, str]]:
    return (_sort_key(_unit_id(unit)), _sort_key(unit.title))
