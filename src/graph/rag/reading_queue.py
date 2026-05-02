"""Priority reading queue helpers for knowledge units."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit

QUEUE_RELATIONS = frozenset({EdgeRelation.REFERENCES.value, EdgeRelation.BUILDS_ON.value})
READ_STATUSES = frozenset({"read", "done", "completed", "finished"})
UNREAD_STATUSES = frozenset({"unread", "new", "not_started", "to_read", "queued"})
IN_PROGRESS_STATUSES = frozenset({"reading", "in_progress", "started"})
REVIEW_TAGS = frozenset(
    {"important", "priority", "review", "revisit", "todo", "follow-up", "followup"}
)


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    return limit


def _coerce_now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _datetime_value(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _metadata_datetime(unit: KnowledgeUnit, key: str) -> datetime | None:
    metadata = unit.metadata or {}
    if isinstance(metadata, dict):
        parsed = _datetime_value(metadata.get(key))
        if parsed is not None:
            return parsed
    return _datetime_value(getattr(unit, key, None))


def _metadata_value(unit: KnowledgeUnit, key: str) -> object:
    metadata = unit.metadata or {}
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return getattr(unit, key, None)


def _priority_score(value: object) -> tuple[float, str | None]:
    if isinstance(value, bool) or value is None:
        return 0.0, None
    if isinstance(value, (int, float)):
        if value <= 0:
            return 0.0, None
        score = min(float(value) * 10.0, 35.0)
        return score, "high priority" if score >= 25.0 else "priority"

    normalized = str(value).strip().lower().replace(" ", "_")
    if normalized in {"urgent", "critical", "p0", "p1", "highest", "high"}:
        return 35.0, "high priority"
    if normalized in {"medium", "normal", "p2"}:
        return 15.0, "priority"
    if normalized in {"low", "p3", "p4"}:
        return -5.0, "low priority"
    return 0.0, None


def _read_status_score(value: object) -> tuple[float, str]:
    if value is None:
        return 20.0, "unread"

    normalized = str(value).strip().lower().replace(" ", "_")
    if normalized in UNREAD_STATUSES:
        return 40.0, "unread"
    if normalized in IN_PROGRESS_STATUSES:
        return 20.0, "in progress"
    if normalized in READ_STATUSES:
        return 0.0, "read"
    return 15.0, "unread"


def _staleness_score(
    *,
    last_read_at: datetime | None,
    updated_at: datetime | None,
    created_at: datetime | None,
    now: datetime,
) -> tuple[float, str | None]:
    reference = updated_at or created_at
    if last_read_at is not None and reference is not None and reference > last_read_at:
        return 30.0, "updated since last read"

    if last_read_at is None:
        return 10.0, "never read"

    days_since_read = max((now - last_read_at).days, 0)
    if days_since_read >= 180:
        return 25.0, "stale"
    if days_since_read >= 60:
        return 15.0, "stale"
    if days_since_read >= 30:
        return 8.0, "stale"
    return 0.0, None


def _tag_score(unit: KnowledgeUnit) -> tuple[float, str | None]:
    tags = {str(tag).strip().lower() for tag in unit.tags}
    metadata = unit.metadata or {}
    if isinstance(metadata, dict):
        metadata_tags = metadata.get("tags")
        if isinstance(metadata_tags, list):
            tags.update(str(tag).strip().lower() for tag in metadata_tags)
    tags.discard("")

    if tags & REVIEW_TAGS:
        return 10.0, "review tag"
    return 0.0, None


def _relation_value(relation: EdgeRelation | str) -> str:
    return relation.value if isinstance(relation, EdgeRelation) else str(relation)


def _inbound_counts(
    units_by_id: dict[str, KnowledgeUnit], edges: Iterable[KnowledgeEdge] | None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    if edges is None:
        return counts

    for edge in edges:
        relation = _relation_value(edge.relation)
        if relation not in QUEUE_RELATIONS:
            continue
        from_unit_id = str(edge.from_unit_id)
        to_unit_id = str(edge.to_unit_id)
        if from_unit_id == to_unit_id:
            continue
        if from_unit_id in units_by_id and to_unit_id in units_by_id:
            counts[to_unit_id] += 1
    return counts


def _unit_payload(
    unit: KnowledgeUnit,
    *,
    score: float,
    explanation: list[str],
    inbound_reference_count: int,
) -> dict[str, Any]:
    return {
        "id": unit.id,
        "source_project": str(unit.source_project),
        "source_id": unit.source_id,
        "source_entity_type": unit.source_entity_type,
        "title": unit.title,
        "content_type": str(unit.content_type),
        "score": round(score, 2),
        "explanation": "; ".join(explanation[:4]) or "baseline",
        "inbound_reference_count": inbound_reference_count,
    }


def _score_unit(
    unit: KnowledgeUnit,
    *,
    inbound_reference_count: int,
    now: datetime,
) -> tuple[float, list[str]]:
    score = 0.0
    explanation: list[str] = []

    read_score, read_reason = _read_status_score(_metadata_value(unit, "read_status"))
    score += read_score
    if read_reason != "read":
        explanation.append(read_reason)

    priority_score, priority_reason = _priority_score(_metadata_value(unit, "priority"))
    score += priority_score
    if priority_reason is not None:
        explanation.append(priority_reason)

    staleness_score, staleness_reason = _staleness_score(
        last_read_at=_metadata_datetime(unit, "last_read_at"),
        updated_at=_metadata_datetime(unit, "updated_at"),
        created_at=_metadata_datetime(unit, "created_at"),
        now=now,
    )
    score += staleness_score
    if staleness_reason is not None:
        explanation.append(staleness_reason)

    tag_score, tag_reason = _tag_score(unit)
    score += tag_score
    if tag_reason is not None:
        explanation.append(tag_reason)

    if inbound_reference_count:
        graph_score = min(inbound_reference_count * 12.0, 36.0)
        score += graph_score
        explanation.append(f"referenced by {inbound_reference_count} unit(s)")

    return score, explanation


def build_reading_queue(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge] | None = None,
    *,
    limit: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic priority queue of units to read or revisit.

    Units rank higher when they are unread, high priority, stale or updated
    since last read, tagged for review, or referenced by other queued units via
    inbound ``REFERENCES`` or ``BUILDS_ON`` edges.
    """
    limit_value = _validate_limit(limit)
    now_value = _coerce_now(now)
    units_by_id = {str(unit.id): unit for unit in units}
    inbound_counts = _inbound_counts(units_by_id, edges)

    scored_units = []
    for unit_id, unit in units_by_id.items():
        score, explanation = _score_unit(
            unit,
            inbound_reference_count=inbound_counts[unit_id],
            now=now_value,
        )
        scored_units.append(
            (
                -score,
                unit.title.casefold(),
                unit_id,
                _unit_payload(
                    unit,
                    score=score,
                    explanation=explanation,
                    inbound_reference_count=inbound_counts[unit_id],
                ),
            )
        )

    queued_payloads = [payload for *_sort_key, payload in sorted(scored_units)]
    candidate_count = len(queued_payloads)
    if limit_value is not None:
        queued_payloads = queued_payloads[:limit_value]

    return {
        "units": queued_payloads,
        "stats": {
            "total_units": len(units_by_id),
            "candidate_units": candidate_count,
            "queued_units": len(queued_payloads),
            "omitted_units": candidate_count - len(queued_payloads),
            "edge_boosted_units": sum(
                1 for unit_id in units_by_id if inbound_counts[unit_id] > 0
            ),
            "limit": limit_value,
            "now": now_value.isoformat(),
        },
    }
