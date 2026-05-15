"""Generate deterministic follow-up questions for RAG retrieval gaps."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit

_MISSING = object()


def _payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _candidate_values(result: Any, key: str) -> Iterable[Any]:
    payload = _payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING:
        yield value
    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get(key, _MISSING)
        if value is not _MISSING:
            yield value
    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        value = _field_value(unit, key)
        if value is not _MISSING:
            yield value
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            value = unit_metadata.get(key, _MISSING)
            if value is not _MISSING:
                yield value


def _string(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).split())
    return text or None


def _first_string(result: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for value in _candidate_values(result, key):
            text = _string(value)
            if text is not None:
                return text
    return None


def _strings(result: Any, keys: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for key in keys:
        for value in _candidate_values(result, key):
            if isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping):
                values.update(text.casefold() for item in value if (text := _string(item)) is not None)
            elif (text := _string(value)) is not None:
                values.add(text.casefold())
    return values


def _domain(result: Any) -> str | None:
    raw = _first_string(result, ("domain", "url", "source_url", "canonical_url"))
    if raw is None:
        return None
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    domain = parsed.hostname or parsed.netloc
    return domain.casefold().removeprefix("www.") if domain else None


def _parse_date(value: Any) -> date | None:
    if value is _MISSING or value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date() if value.tzinfo else value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(text)
            except ValueError:
                return None
    return None


def _latest_date(result: Any) -> date | None:
    dates = [
        parsed
        for key in ("published_at", "publication_date", "updated_at", "created_at", "date")
        for value in _candidate_values(result, key)
        if (parsed := _parse_date(value)) is not None
    ]
    return max(dates) if dates else None


def _add_question(questions: list[dict[str, Any]], question: str, reason: str, priority: int) -> None:
    if any(item["question"] == question for item in questions):
        return
    questions.append({"question": question, "reason": reason, "priority": priority})


def build_followup_questions(
    query: Any,
    results: Iterable[Any],
    *,
    required_facets: Mapping[str, Iterable[str]] | None = None,
    max_questions: int = 5,
) -> list[dict[str, Any]]:
    """Build ordered clarification or retrieval follow-up questions."""
    if not isinstance(max_questions, int) or isinstance(max_questions, bool) or max_questions < 0:
        raise ValueError("max_questions must be a non-negative integer")
    if max_questions == 0:
        return []

    query_text = _string(query) or "the topic"
    rows = list(results)
    questions: list[dict[str, Any]] = []
    if not rows:
        return [
            {
                "question": f"What broader sources or keywords should be searched for {query_text}?",
                "reason": "no retrieval results were provided",
                "priority": 100,
            }
        ][:max_questions]

    facet_keys = {
        "tags": ("tags", "tag"),
        "domains": ("domain", "url", "source_url", "canonical_url"),
        "source_projects": ("source_project",),
        "content_types": ("content_type", "source_entity_type", "type"),
    }
    observed = {
        "tags": set().union(*(_strings(row, facet_keys["tags"]) for row in rows)),
        "domains": {domain for row in rows if (domain := _domain(row)) is not None},
        "source_projects": set().union(*(_strings(row, facet_keys["source_projects"]) for row in rows)),
        "content_types": set().union(*(_strings(row, facet_keys["content_types"]) for row in rows)),
    }

    for facet, required in (required_facets or {}).items():
        required_values = {str(value).casefold() for value in required}
        missing = sorted(required_values - observed.get(facet, set()))
        if missing:
            label = ", ".join(missing)
            _add_question(
                questions,
                f"Should retrieval include {facet.replace('_', ' ')} for {label}?",
                f"required {facet} were not present in the retrieved results",
                90,
            )

    short_count = sum(1 for row in rows if len(_first_string(row, ("content", "text", "snippet")) or "") < 80)
    if short_count:
        _add_question(
            questions,
            "Should the search retrieve longer source passages before answering?",
            f"{short_count} result(s) have sparse content",
            70,
        )

    domains = [_domain(row) or "unknown" for row in rows]
    if len(set(domains) - {"unknown"}) <= 1 and len(rows) > 1:
        _add_question(
            questions,
            "What additional source domains should be checked for corroboration?",
            "retrieved results have low source diversity",
            60,
        )

    latest_dates = [parsed for row in rows if (parsed := _latest_date(row)) is not None]
    if latest_dates and max(latest_dates) < date(2025, 1, 1):
        _add_question(
            questions,
            "Should newer sources be retrieved before relying on this answer?",
            "all dated evidence is older than 2025",
            50,
        )

    projects = Counter(_first_string(row, ("source_project",)) or "unknown" for row in rows)
    if projects.get("unknown", 0):
        _add_question(
            questions,
            "Which source project or collection should unknown results be attributed to?",
            "some results are missing source project provenance",
            40,
        )

    return sorted(questions, key=lambda item: (-item["priority"], item["question"]))[:max_questions]
