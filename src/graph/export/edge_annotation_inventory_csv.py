"""CSV export for edge annotation inventory rows."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeEdge

_FIELDNAMES = [
    "edge_id",
    "from_unit_id",
    "to_unit_id",
    "relation",
    "source",
    "source_project",
    "provenance",
    "annotation_keys",
    "note_comment_length",
    "evidence_count",
    "reference_count",
    "has_rationale_text",
]
_ANNOTATION_KEYS = {"note", "notes", "comment", "comments", "rationale", "evidence", "references", "citations"}
_NOTE_KEYS = {"note", "notes", "comment", "comments"}
_RATIONALE_KEYS = {"rationale"}
_EVIDENCE_KEYS = {"evidence"}
_REFERENCE_KEYS = {"references", "citations"}
_PROJECT_KEYS = ("source_project", "project")
_PROVENANCE_KEYS = ("provenance", "provenance_id", "source_provenance")
_WHITESPACE_RE = re.compile(r"\s+")


def export_edge_annotation_inventory_csv(
    edges: Iterable[KnowledgeEdge],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write deterministic edge annotation inventory rows."""
    edge_list = list(edges)
    rows = [_edge_row(edge) for edge in sorted(edge_list, key=_edge_sort_key)]
    text = _render_csv(rows)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="")
    return {
        "path": str(output_path),
        "edge_count": len(edge_list),
        "rows_exported": len(rows),
        "bytes_written": output_path.stat().st_size,
    }


def _edge_row(edge: KnowledgeEdge) -> dict[str, str | int]:
    metadata = edge.metadata if isinstance(edge.metadata, Mapping) else {}
    annotation_keys = _annotation_keys(metadata)
    return {
        "edge_id": _inline_text(edge.id),
        "from_unit_id": _inline_text(edge.from_unit_id),
        "to_unit_id": _inline_text(edge.to_unit_id),
        "relation": _field_value(edge.relation) or "Unknown",
        "source": _field_value(edge.source) or "Unknown",
        "source_project": _first_metadata_value(metadata, _PROJECT_KEYS),
        "provenance": _first_metadata_value(metadata, _PROVENANCE_KEYS),
        "annotation_keys": "; ".join(annotation_keys),
        "note_comment_length": _note_comment_length(metadata),
        "evidence_count": _count_values(metadata, _EVIDENCE_KEYS),
        "reference_count": _count_values(metadata, _REFERENCE_KEYS),
        "has_rationale_text": _bool_text(_has_text(metadata, _RATIONALE_KEYS)),
    }


def _annotation_keys(metadata: Mapping[object, object]) -> list[str]:
    keys = []
    for raw_key, value in metadata.items():
        key = _key(raw_key)
        if key in _ANNOTATION_KEYS and _has_any_value(value):
            keys.append(key)
    return sorted(set(keys), key=_sort_key)


def _note_comment_length(metadata: Mapping[object, object]) -> int:
    return sum(len(text) for key in _NOTE_KEYS for text in _metadata_texts(metadata, key))


def _count_values(metadata: Mapping[object, object], keys: set[str]) -> int:
    return sum(len(_metadata_texts(metadata, key)) for key in keys)


def _has_text(metadata: Mapping[object, object], keys: set[str]) -> bool:
    return any(_metadata_texts(metadata, key) for key in keys)


def _metadata_texts(metadata: Mapping[object, object], normalized_key: str) -> list[str]:
    texts: list[str] = []
    for raw_key, value in metadata.items():
        if _key(raw_key) == normalized_key:
            texts.extend(_texts(value))
    return texts


def _texts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        texts: list[str] = []
        for _key_value, item in sorted(value.items(), key=lambda item: _sort_key(item[0])):
            texts.extend(_texts(item))
        return texts
    if isinstance(value, list | tuple | set):
        texts = []
        for item in value:
            texts.extend(_texts(item))
        return texts
    text = _inline_text(value)
    return [text] if text else []


def _has_any_value(value: object) -> bool:
    return bool(_texts(value))


def _first_metadata_value(metadata: Mapping[object, object], keys: tuple[str, ...]) -> str:
    for desired_key in keys:
        for raw_key, value in metadata.items():
            if _key(raw_key) == desired_key:
                text = "; ".join(_texts(value))
                if text:
                    return text
    return ""


def _render_csv(rows: list[dict[str, str | int]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _key(value: object) -> str:
    return _field_value(value).casefold().replace("-", "_").replace(" ", "_")


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _edge_sort_key(
    edge: KnowledgeEdge,
) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str], tuple[str, str]]:
    return (
        _sort_key(edge.from_unit_id),
        _sort_key(edge.to_unit_id),
        _sort_key(_field_value(edge.relation)),
        _sort_key(edge.id),
    )


def _bool_text(value: bool) -> str:
    return "true" if value else "false"
