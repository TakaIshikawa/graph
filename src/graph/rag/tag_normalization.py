"""Local tag normalization suggestions for graph units."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from graph.types.models import KnowledgeUnit

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _validate_min_count(min_count: int) -> int:
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count <= 0:
        raise ValueError("min_count must be a positive integer")
    return min_count


def _validate_similarity(min_similarity: float) -> float:
    if not isinstance(min_similarity, int | float) or isinstance(min_similarity, bool):
        raise ValueError("min_similarity must be a number between 0 and 1")
    value = float(min_similarity)
    if value < 0 or value > 1:
        raise ValueError("min_similarity must be a number between 0 and 1")
    return value


def _validate_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    return limit


def _singularize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith(("sses", "ss")):
        return token
    if token.endswith("es") and (
        token.endswith(("ches", "shes", "xes", "zes")) or token[-3] == "s"
    ):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def _tokens(tag: str) -> list[str]:
    return TOKEN_RE.findall(tag.casefold())


def _normalized(tag: str) -> str:
    return " ".join(_singularize(token) for token in _tokens(tag))


def _display_key(tag: str) -> tuple[str, bool, int, str]:
    stripped = tag.strip()
    return (stripped.casefold(), not stripped.islower(), len(stripped), stripped)


def _choose_canonical(tags: Iterable[str], counts: dict[str, int]) -> str:
    return sorted(tags, key=lambda tag: (-counts[tag], _display_key(tag)))[0]


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _is_similar(left: str, right: str, threshold: float) -> bool:
    score = _similarity(left, right)
    if score < threshold:
        return False
    if left == right:
        return True

    left_tokens = left.split()
    right_tokens = right.split()
    if len(left_tokens) != len(right_tokens):
        return True

    for left_token, right_token in zip(left_tokens, right_tokens, strict=True):
        if left_token == right_token:
            continue
        if left_token[:1] != right_token[:1]:
            return False
    return True


def _component_similarity(tags: list[str], normalized: dict[str, str]) -> float:
    if len(tags) < 2:
        return 1.0
    scores = [
        _similarity(normalized[left], normalized[right])
        for index, left in enumerate(tags)
        for right in tags[index + 1 :]
    ]
    return round(min(scores), 6)


def _sorted_tags(tags: Iterable[str], counts: dict[str, int]) -> list[str]:
    return sorted(tags, key=lambda tag: (-counts[tag], _display_key(tag)))


def suggest_tag_normalizations(
    units: Iterable[KnowledgeUnit],
    *,
    min_count: int = 1,
    min_similarity: float = 0.82,
    limit: int | None = 50,
) -> list[dict[str, Any]]:
    """Suggest deterministic local merges for likely duplicate tags.

    The helper only reads ``unit.tags`` and returns suggestions; it does not
    update units or rewrite tag lists.
    """
    min_count_value = _validate_min_count(min_count)
    min_similarity_value = _validate_similarity(min_similarity)
    limit_value = _validate_limit(limit)

    counts: dict[str, int] = defaultdict(int)
    unit_ids_by_tag: dict[str, set[str]] = defaultdict(set)

    for unit in units:
        unit_id = str(unit.id)
        for raw_tag in unit.tags:
            tag = raw_tag.strip()
            if not tag:
                continue
            counts[tag] += 1
            unit_ids_by_tag[tag].add(unit_id)

    tags = sorted(counts, key=lambda tag: _display_key(tag))
    normalized = {tag: _normalized(tag) for tag in tags}
    parent = {tag: tag for tag in tags}

    def find(tag: str) -> str:
        while parent[tag] != tag:
            parent[tag] = parent[parent[tag]]
            tag = parent[tag]
        return tag

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        parent[max(left_root, right_root)] = min(left_root, right_root)

    for left_index, left in enumerate(tags):
        for right in tags[left_index + 1 :]:
            if _is_similar(normalized[left], normalized[right], min_similarity_value):
                union(left, right)

    components: dict[str, list[str]] = defaultdict(list)
    for tag in tags:
        components[find(tag)].append(tag)

    suggestions: list[dict[str, Any]] = []
    for component_tags in components.values():
        if len(component_tags) < 2:
            continue
        total_count = sum(counts[tag] for tag in component_tags)
        if total_count < min_count_value:
            continue

        canonical = _choose_canonical(component_tags, counts)
        variants = [tag for tag in _sorted_tags(component_tags, counts) if tag != canonical]
        affected_unit_ids = sorted(
            {unit_id for tag in component_tags for unit_id in unit_ids_by_tag[tag]}
        )

        suggestions.append(
            {
                "canonical_tag": canonical,
                "variants": variants,
                "counts": {tag: counts[tag] for tag in _sorted_tags(component_tags, counts)},
                "similarity": _component_similarity(component_tags, normalized),
                "affected_unit_ids": affected_unit_ids,
            }
        )

    suggestions.sort(
        key=lambda item: (
            -sum(item["counts"].values()),
            -item["similarity"],
            item["canonical_tag"].casefold(),
            item["variants"],
        )
    )
    if limit_value is not None:
        return suggestions[:limit_value]
    return suggestions
