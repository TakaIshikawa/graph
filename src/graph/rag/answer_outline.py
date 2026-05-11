"""Build deterministic RAG answer outlines from retrieved results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from graph.rag.keywords import COMMON_STOPWORDS, TOKEN_RE

_MISSING = object()


@dataclass(frozen=True)
class _OutlineResult:
    index: int
    result_id: str
    title: str | None
    source_project: str | None
    tags: tuple[str, ...]
    tag_keys: frozenset[str]
    covered_terms: frozenset[str]


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, value: int) -> int:
        parent = self.parents[value]
        if parent != value:
            self.parents[value] = self.find(parent)
        return self.parents[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parents[right_root] = left_root


def _validate_positive_int(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _string_value(value: Any) -> str | None:
    if value is _MISSING or value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = " ".join(str(value).strip().split())
    return text or None


def _field_value(item: Any, key: str) -> Any:
    if item is _MISSING or item is None:
        return _MISSING
    if isinstance(item, Mapping):
        return item.get(key, _MISSING)
    return getattr(item, key, _MISSING)


def _result_payload(result: Any) -> Any:
    if isinstance(result, tuple) and result:
        return result[0]
    return result


def _result_value(result: Any, key: str) -> Any:
    payload = _result_payload(result)
    value = _field_value(payload, key)
    if value is not _MISSING and value is not None:
        return value

    unit = _field_value(payload, "unit")
    if unit is not _MISSING and unit is not None:
        unit_value = _field_value(unit, key)
        if unit_value is not _MISSING and unit_value is not None:
            return unit_value

    metadata = _field_value(payload, "metadata")
    if isinstance(metadata, Mapping):
        metadata_value = metadata.get(key, _MISSING)
        if metadata_value is not _MISSING and metadata_value is not None:
            return metadata_value

    if unit is not _MISSING and unit is not None:
        unit_metadata = _field_value(unit, "metadata")
        if isinstance(unit_metadata, Mapping):
            return unit_metadata.get(key, _MISSING)
    return value


def _result_id(result: Any, index: int) -> str:
    for key in ("id", "unit_id", "source_id"):
        value = _string_value(_result_value(result, key))
        if value is not None:
            return value
    return f"result-{index + 1}"


def _iter_strings(value: Any) -> list[str]:
    if value is _MISSING or value is None:
        return []
    if isinstance(value, Mapping):
        return [item for nested in value.values() for item in _iter_strings(nested)]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return [item for nested in value for item in _iter_strings(nested)]
    text = _string_value(value)
    return [] if text is None else [text]


def _tag_key(label: str) -> str:
    return " ".join(label.casefold().split())


def _display_key(label: str) -> tuple[str, str]:
    return (label.casefold(), label)


def _query_terms(query: Any) -> list[str]:
    if query is None:
        return []
    terms: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(str(query).casefold()):
        if token in COMMON_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms


def _tokens_for_result(result: Any) -> set[str]:
    texts: list[str] = []
    for field in ("title", "content", "source_project", "source_entity_type"):
        value = _string_value(_result_value(result, field))
        if value is not None:
            texts.append(value)
    texts.extend(_iter_strings(_result_value(result, "tags")))
    texts.extend(_iter_strings(_result_value(result, "metadata")))
    return {
        token
        for text in texts
        for token in TOKEN_RE.findall(text.casefold())
        if token not in COMMON_STOPWORDS
    }


def _result_item(result: Any, index: int, query_terms: list[str]) -> _OutlineResult:
    tag_labels: dict[str, str] = {}
    for tag in _iter_strings(_result_value(result, "tags")):
        key = _tag_key(tag)
        if key and key not in tag_labels:
            tag_labels[key] = tag

    tokens = _tokens_for_result(result)
    return _OutlineResult(
        index=index,
        result_id=_result_id(result, index),
        title=_string_value(_result_value(result, "title")),
        source_project=_string_value(_result_value(result, "source_project")),
        tags=tuple(sorted(tag_labels.values(), key=_display_key)),
        tag_keys=frozenset(tag_labels),
        covered_terms=frozenset(term for term in query_terms if term in tokens),
    )


def _should_link(left: _OutlineResult, right: _OutlineResult) -> bool:
    if left.tag_keys & right.tag_keys:
        return True
    if left.covered_terms & right.covered_terms:
        return True
    return (
        left.source_project is not None
        and left.source_project == right.source_project
        and not left.tag_keys
        and not right.tag_keys
    )


def _component_label(component: list[_OutlineResult]) -> str:
    tag_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    term_counts: Counter[str] = Counter()
    for item in component:
        tag_counts.update(item.tags)
        if item.source_project is not None:
            source_counts[item.source_project] += 1
        term_counts.update(item.covered_terms)

    if tag_counts:
        tags = sorted(tag_counts, key=lambda tag: (-tag_counts[tag], _display_key(tag)))[:2]
        return " + ".join(tags)
    if term_counts:
        terms = sorted(term_counts, key=lambda term: (-term_counts[term], term))[:2]
        return " + ".join(terms)
    if source_counts:
        return sorted(source_counts, key=lambda source: (-source_counts[source], source))[0]
    return "Retrieved evidence"


def _component_sort_key(component: list[_OutlineResult], query_terms: list[str]) -> tuple:
    covered = {term for item in component for term in item.covered_terms}
    first_id = min(item.result_id for item in component)
    coverage_positions = tuple(query_terms.index(term) for term in query_terms if term in covered)
    return (
        -len(covered),
        -len(component),
        coverage_positions,
        first_id,
        _component_label(component),
    )


def _components(items: list[_OutlineResult], query_terms: list[str]) -> list[list[_OutlineResult]]:
    disjoint_set = _DisjointSet(len(items))
    for left_index, left in enumerate(items):
        for right_index in range(left_index + 1, len(items)):
            if _should_link(left, items[right_index]):
                disjoint_set.union(left_index, right_index)

    grouped: dict[int, list[_OutlineResult]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped[disjoint_set.find(index)].append(item)

    components = list(grouped.values())
    for component in components:
        component.sort(key=lambda item: (-len(item.covered_terms), item.result_id, item.index))
    components.sort(key=lambda component: _component_sort_key(component, query_terms))
    return components


def _rationale(component: list[_OutlineResult]) -> str:
    shared_tags = sorted(
        set.intersection(*(set(item.tags) for item in component)) if component else set(),
        key=_display_key,
    )
    sources = sorted(
        {item.source_project for item in component if item.source_project is not None}
    )
    reasons: list[str] = []
    if shared_tags:
        reasons.append("shared tags: " + ", ".join(shared_tags))
    if sources:
        reasons.append("sources: " + ", ".join(sources))
    covered = sorted({term for item in component for term in item.covered_terms})
    if covered:
        reasons.append("query coverage: " + ", ".join(covered))
    return "; ".join(reasons) if reasons else "Grouped as standalone retrieved evidence."


def build_answer_outline(
    results: Iterable[Any],
    query: Any,
    *,
    max_sections: int = 5,
    max_evidence_per_section: int = 3,
) -> dict[str, Any]:
    """Return a deterministic outline payload for downstream RAG synthesis."""
    max_sections_value = _validate_positive_int(max_sections, "max_sections")
    max_evidence_value = _validate_positive_int(
        max_evidence_per_section,
        "max_evidence_per_section",
    )
    result_list = list(results)
    query_terms = _query_terms(query)
    items = [
        _result_item(result, index, query_terms)
        for index, result in enumerate(result_list)
    ]

    sections: list[dict[str, Any]] = []
    for component in _components(items, query_terms)[:max_sections_value]:
        covered_terms = [
            term
            for term in query_terms
            if any(term in item.covered_terms for item in component)
        ]
        evidence = component[:max_evidence_value]
        title = _component_label(component)
        sections.append(
            {
                "title": title,
                "rationale": _rationale(component),
                "evidence_result_ids": [item.result_id for item in evidence],
                "coverage_terms": covered_terms,
                "missing_terms": [term for term in query_terms if term not in covered_terms],
            }
        )

    globally_covered = {
        term
        for section in sections
        for term in section["coverage_terms"]
    }
    return {
        "query": "" if query is None else str(query),
        "query_terms": query_terms,
        "sections": sections,
        "missing_terms": [term for term in query_terms if term not in globally_covered],
    }
