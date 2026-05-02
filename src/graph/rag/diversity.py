"""Source-diverse reranking helpers for RAG result lists."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


@dataclass(frozen=True)
class _Candidate:
    item: KnowledgeUnit | tuple[KnowledgeUnit, Any]
    unit: KnowledgeUnit
    score: Any
    source: str
    unit_id: str


def _validate_max_per_source(max_per_source: int) -> int:
    if (
        not isinstance(max_per_source, int)
        or isinstance(max_per_source, bool)
        or max_per_source < 1
    ):
        raise ValueError("max_per_source must be a positive integer")
    return max_per_source


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer or None")
    return limit


def _source_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, SourceProject):
        return value.value
    return str(value)


def _unit_source(unit: KnowledgeUnit, source_key: str) -> str:
    source = _source_value(getattr(unit, "source_project", None))
    if source is not None:
        return source

    metadata = getattr(unit, "metadata", {}) or {}
    if isinstance(metadata, dict):
        source = _source_value(metadata.get(source_key))
        if source is not None:
            return source

    return ""


def _candidate(
    item: KnowledgeUnit | tuple[KnowledgeUnit, Any],
    source_key: str,
) -> _Candidate:
    if isinstance(item, tuple) and len(item) == 2:
        unit, score = item
    else:
        unit = item
        score = None

    if not isinstance(unit, KnowledgeUnit):
        raise ValueError(
            "results must contain KnowledgeUnit objects or (KnowledgeUnit, score) tuples"
        )

    return _Candidate(
        item=item,
        unit=unit,
        score=score,
        source=_unit_source(unit, source_key),
        unit_id=str(unit.id),
    )


def _score_sort_value(score: Any) -> tuple[int, float | str]:
    if isinstance(score, bool) or score is None:
        return (1, "")
    if isinstance(score, (int, float)):
        return (0, -float(score))
    return (0, str(score))


def rerank_for_source_diversity(
    results: Iterable[KnowledgeUnit | tuple[KnowledgeUnit, Any]],
    *,
    source_key: str = "source_project",
    max_per_source: int = 2,
    limit: int | None = None,
) -> list[KnowledgeUnit | tuple[KnowledgeUnit, Any]]:
    """Rerank results so leading items are less dominated by one source.

    The returned items are the exact input objects or tuples, only reordered.
    Scores are used as the primary relevance order when present; ties are
    broken by unit id so equivalent input sets produce the same output.
    """
    max_count = _validate_max_per_source(max_per_source)
    limit_value = _validate_limit(limit)
    if not isinstance(source_key, str) or not source_key:
        raise ValueError("source_key must be a non-empty string")

    candidates = sorted(
        (_candidate(item, source_key) for item in results),
        key=lambda candidate: (
            _score_sort_value(candidate.score),
            candidate.unit_id,
            candidate.source,
        ),
    )
    if limit_value == 0:
        return []

    counts: dict[str, int] = defaultdict(int)
    selected: list[_Candidate] = []
    remaining = list(candidates)

    while remaining:
        next_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if counts[candidate.source] < max_count
            ),
            None,
        )
        if next_index is None:
            break

        candidate = remaining.pop(next_index)
        selected.append(candidate)
        counts[candidate.source] += 1

    selected.extend(remaining)
    items = [candidate.item for candidate in selected]
    if limit_value is not None:
        return items[:limit_value]
    return items
