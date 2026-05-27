"""Audit answer entities against retrieved RAG evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import content_text, iter_strings, result_id, string, value

_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w&'.-]+(?:\s+|$)){2,}")
_ENTITY_KEYS = (
    "entity",
    "entities",
    "person",
    "people",
    "author",
    "authors",
    "organization",
    "organizations",
    "company",
    "companies",
    "project",
    "projects",
    "publisher",
    "source",
    "source_project",
)
_TEXT_KEYS = ("title", "content", "text", "summary", "snippet")
_TRAILING_WORDS = {"and", "or", "the", "a", "an", "of", "for", "with", "in", "on", "by"}


def audit_answer_entity_consistency(answer: str, results: Iterable[Any]) -> dict[str, Any]:
    """Compare named entities in an answer with entities visible in evidence."""
    answer_text = string(answer) or ""
    result_list = list(results or [])
    answer_entities = _entities_from_text(answer_text)
    evidence_by_entity: dict[str, dict[str, Any]] = {}
    per_result = []

    for index, result in enumerate(result_list):
        rid = result_id(result, index)
        evidence_entities = _entities_from_result(result)
        supported = sorted((entity for entity in answer_entities if _norm(entity) in evidence_entities), key=_sort_key)
        for entity in supported:
            row = evidence_by_entity.setdefault(
                _norm(entity),
                {"entity": entity, "supporting_result_ids": [], "support_count": 0},
            )
            row["supporting_result_ids"].append(rid)
            row["support_count"] += 1
        per_result.append({"result_id": rid, "supported_entity_count": len(supported), "supported_entities": supported})

    supported_entities = sorted(evidence_by_entity.values(), key=lambda row: _sort_key(row["entity"]))
    evidence_entities = sorted(
        {entity for result in result_list for entity in _display_entities_from_result(result)},
        key=_sort_key,
    )
    unsupported = [
        {"entity": entity, "reason": "entity_absent_from_evidence"}
        for entity in sorted(answer_entities, key=_sort_key)
        if _norm(entity) not in evidence_by_entity
    ]
    warnings = []
    if not answer_text:
        warnings.append("empty_answer")
    if not result_list:
        warnings.append("no_evidence")

    return {
        "answer_entities": sorted(answer_entities, key=_sort_key),
        "evidence_entities": evidence_entities,
        "supported_entities": supported_entities,
        "unsupported_entities": unsupported,
        "support_ratio": round(len(supported_entities) / len(answer_entities), 3) if answer_entities else 1.0,
        "per_result_support": per_result,
        "warnings": warnings,
    }


def _entities_from_result(result: Any) -> set[str]:
    entities: set[str] = set()
    for key in _TEXT_KEYS:
        text = string(value(result, key))
        if text:
            entities.update(_norm(entity) for entity in _entities_from_text(text))
    for key in _ENTITY_KEYS:
        for text in iter_strings(value(result, key)):
            entities.add(_norm(text))
            entities.update(_norm(entity) for entity in _entities_from_text(text))
    return entities


def _display_entities_from_result(result: Any) -> set[str]:
    entities: set[str] = set()
    for key in _TEXT_KEYS:
        text = string(value(result, key))
        if text:
            entities.update(_entities_from_text(text))
    for key in _ENTITY_KEYS:
        for text in iter_strings(value(result, key)):
            entities.add(text)
            entities.update(_entities_from_text(text))
    return entities


def _entities_from_text(text: str) -> set[str]:
    entities = set()
    for match in _ENTITY_RE.finditer(text):
        entity = " ".join(match.group(0).strip(" ,.;:!?()[]{}").split())
        words = entity.split()
        while words and words[-1].casefold() in _TRAILING_WORDS:
            words.pop()
        if len(words) >= 2:
            entities.add(" ".join(words))
    return entities


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def _sort_key(text: str) -> tuple[str, str]:
    return (text.casefold(), text)
