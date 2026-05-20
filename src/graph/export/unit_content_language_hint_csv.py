"""CSV export for lightweight unit content language hints."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_FIELDNAMES = ["source", "unit_id", "title", "language_hint", "confidence", "evidence"]
_STOPWORDS = {
    "English": {"the", "and", "to", "of", "in", "for", "with", "is", "this", "that"},
    "Spanish": {"el", "la", "los", "las", "de", "que", "y", "en", "para", "con"},
    "French": {"le", "la", "les", "des", "de", "que", "et", "en", "pour", "avec"},
}
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")
_WHITESPACE_RE = re.compile(r"\s+")
_MIN_TEXT_LENGTH = 12


def export_unit_content_language_hint_csv(
    units: Iterable[KnowledgeUnit | Mapping[str, Any]],
    path: str | Path | None = None,
) -> str | dict[str, Any]:
    """Return or write heuristic language hints per unit."""
    unit_list = list(units)
    rows = _language_rows(unit_list)
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


def _language_rows(units: list[KnowledgeUnit | Mapping[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for unit in units:
        language, confidence, evidence = _language_hint(_unit_text(unit))
        rows.append(
            {
                "source": _field_value(_get(unit, "source_project")) or "Unknown",
                "unit_id": _unit_id(unit),
                "title": _field_value(_get(unit, "title")),
                "language_hint": language,
                "confidence": f"{confidence:.2f}",
                "evidence": evidence,
            }
        )
    return sorted(rows, key=lambda row: (_sort_key(row["source"]), _sort_key(row["unit_id"])))


def _language_hint(text: str) -> tuple[str, float, str]:
    if len(text) < _MIN_TEXT_LENGTH:
        return "Unknown", 0.0, "too_short"

    japanese_count = len(_JAPANESE_RE.findall(text))
    if japanese_count >= 4:
        confidence = min(1.0, japanese_count / max(len(text), 1) * 4)
        return "Japanese", confidence, f"japanese_chars={japanese_count}"

    words = [word.casefold().strip("'") for word in _WORD_RE.findall(text)]
    if len(words) < 3:
        return "Unknown", 0.0, "too_few_words"

    scores = {language: sum(1 for word in words if word in stopwords) for language, stopwords in _STOPWORDS.items()}
    language, score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
    if score == 0:
        return "Unknown", 0.0, "no_stopword_match"
    return language, min(1.0, score / max(len(words), 1) * 4), f"stopwords={score}"


def _unit_text(unit: KnowledgeUnit | Mapping[str, Any]) -> str:
    parts = [_field_value(_get(unit, "title")), _field_value(_get(unit, "content"))]
    metadata = _metadata(unit)
    for key in ("language", "description", "summary", "text", "caption"):
        parts.append(_value_text(_casefold_get(metadata, key)))
    return _inline_text(" ".join(part for part in parts if part))


def _value_text(value: object) -> str:
    if value is None or isinstance(value, bytes):
        return ""
    if isinstance(value, Mapping):
        return ""
    if isinstance(value, list | tuple | set):
        return " ".join(_value_text(item) for item in value)
    return _field_value(value)


def _metadata(unit: KnowledgeUnit | Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _get(unit, "metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _casefold_get(mapping: Mapping[str, Any], key: str) -> object:
    for candidate_key, value in mapping.items():
        if _field_value(candidate_key).casefold() == key.casefold():
            return value
    return None


def _render_csv(rows: list[dict[str, str]]) -> str:
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
