"""Markdown duplicate candidate report helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def export_duplicate_candidates_markdown(
    candidates: Iterable[dict],
    *,
    min_score: float | None = None,
    limit: int | None = None,
) -> str:
    """Render duplicate candidate pairs as a deterministic Markdown report."""
    min_score_value = _validate_min_score(min_score)
    limit_value = _validate_limit(limit)

    rows = sorted(
        (_candidate_row(candidate) for candidate in candidates if isinstance(candidate, Mapping)),
        key=_candidate_sort_key,
    )
    if min_score_value is not None:
        rows = [row for row in rows if row["score"] >= min_score_value]
    if limit_value is not None:
        rows = rows[:limit_value]

    return _render_report(rows, min_score=min_score_value, limit=limit_value)


def _candidate_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    units = _candidate_units(candidate)
    unit_ids = _candidate_unit_ids(candidate, units)
    left = _unit_cell(units[0] if len(units) > 0 else {}, unit_ids[0])
    right = _unit_cell(units[1] if len(units) > 1 else {}, unit_ids[1])
    score = _score(candidate.get("score"))
    reasons = _reason_text(candidate)
    return {
        "score": score,
        "left_id": left["id"],
        "left_title": left["title"],
        "left_source": left["source"],
        "right_id": right["id"],
        "right_title": right["title"],
        "right_source": right["source"],
        "reasons": reasons,
    }


def _candidate_units(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    units = candidate.get("units")
    if isinstance(units, Sequence) and not isinstance(units, str | bytes):
        return [unit for unit in units if isinstance(unit, Mapping)]
    return []


def _candidate_unit_ids(
    candidate: Mapping[str, Any],
    units: list[Mapping[str, Any]],
) -> tuple[str, str]:
    raw_ids = candidate.get("unit_ids")
    ids: list[str] = []
    if isinstance(raw_ids, Sequence) and not isinstance(raw_ids, str | bytes):
        ids.extend(_inline_text(unit_id) for unit_id in raw_ids)
    if len(ids) < 2:
        ids.extend(_inline_text(unit.get("id")) for unit in units)
    ids = [unit_id for unit_id in ids if unit_id]
    return (
        ids[0] if len(ids) > 0 else "",
        ids[1] if len(ids) > 1 else "",
    )


def _unit_cell(unit: Mapping[str, Any], fallback_id: str) -> dict[str, str]:
    unit_id = _inline_text(unit.get("id")) or fallback_id
    title = _inline_text(unit.get("title")) or "_None_"
    source = _source_text(unit)
    return {
        "id": unit_id,
        "title": title,
        "source": source,
    }


def _source_text(unit: Mapping[str, Any]) -> str:
    source_project = _inline_text(unit.get("source_project"))
    source_id = _inline_text(unit.get("source_id"))
    source_entity_type = _inline_text(unit.get("source_entity_type"))
    parts = [part for part in (source_project, source_entity_type, source_id) if part]
    return " / ".join(parts) if parts else "_None_"


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _reason_text(candidate: Mapping[str, Any]) -> str:
    reasons = candidate.get("reasons")
    if isinstance(reasons, Sequence) and not isinstance(reasons, str | bytes):
        reason_parts = [_inline_text(reason) for reason in reasons if _inline_text(reason)]
    else:
        reason = _inline_text(candidate.get("reason"))
        reason_parts = [reason] if reason else []

    matching_fields = candidate.get("matching_fields")
    if isinstance(matching_fields, Mapping) and matching_fields:
        reason_parts.append(_json_text(matching_fields))

    if not reason_parts:
        return "_None_"
    return "; ".join(reason_parts)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _candidate_sort_key(row: Mapping[str, Any]) -> tuple[float, str, str, str, str]:
    return (
        -float(row["score"]),
        _sort_text(row["left_id"]),
        _sort_text(row["right_id"]),
        _sort_text(row["left_title"]),
        _sort_text(row["right_title"]),
    )


def _render_report(
    rows: list[Mapping[str, Any]],
    *,
    min_score: float | None,
    limit: int | None,
) -> str:
    lines = [
        "# Duplicate Candidates",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Candidates | {len(rows)} |",
        f"| Minimum score | {_filter_text(min_score)} |",
        f"| Limit | {_filter_text(limit)} |",
        "",
        "## Candidates",
        "",
        "| Score | Left ID | Left title | Left source | Right ID | Right title | Right source | Reasons |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]

    if not rows:
        lines.append("| 0 | _None_ | _None_ | _None_ | _None_ | _None_ | _None_ | _No matching duplicate candidates._ |")
        return "\n".join(lines).rstrip() + "\n"

    for row in rows:
        lines.append(
            "| "
            f"{_number_text(row['score'])} | "
            f"{_markdown_cell(row['left_id'])} | "
            f"{_markdown_cell(row['left_title'])} | "
            f"{_markdown_cell(row['left_source'])} | "
            f"{_markdown_cell(row['right_id'])} | "
            f"{_markdown_cell(row['right_title'])} | "
            f"{_markdown_cell(row['right_source'])} | "
            f"{_markdown_cell(row['reasons'])} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def _validate_min_score(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("min_score must be a number or None")
    return float(value)


def _validate_limit(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("limit must be a non-negative integer or None")
    return value


def _filter_text(value: Any) -> str:
    if value is None:
        return "_None_"
    if isinstance(value, float):
        return _number_text(value)
    return str(value)


def _number_text(value: Any) -> str:
    number = round(float(value), 6)
    return f"{number:g}"


def _markdown_cell(value: Any) -> str:
    return (
        _inline_text(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("*", r"\*")
        .replace("_", r"\_")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("`", r"\`")
    )


def _sort_text(value: Any) -> str:
    return _inline_text(value).casefold()


def _inline_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()
