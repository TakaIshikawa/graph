from __future__ import annotations

import pytest

from graph.rag import rerank_for_source_diversity
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str = SourceProject.MAX,
    *,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def unit_ids(results) -> list[str]:
    return [item[0].id if isinstance(item, tuple) else item.id for item in results]


def test_rerank_for_source_diversity_limits_repeated_sources_in_leading_results():
    results = [
        (unit("max-1", SourceProject.MAX), 0.99),
        (unit("max-2", SourceProject.MAX), 0.98),
        (unit("max-3", SourceProject.MAX), 0.97),
        (unit("presence-1", SourceProject.PRESENCE), 0.80),
        (unit("forty-two-1", SourceProject.FORTY_TWO), 0.70),
        (unit("max-4", SourceProject.MAX), 0.60),
    ]

    reranked = rerank_for_source_diversity(results, max_per_source=2)

    assert unit_ids(reranked) == [
        "max-1",
        "max-2",
        "presence-1",
        "forty-two-1",
        "max-3",
        "max-4",
    ]


def test_rerank_for_source_diversity_preserves_tuple_payload_shape_and_objects():
    first = unit("max-1", SourceProject.MAX)
    second = unit("max-2", SourceProject.MAX)
    third = unit("presence-1", SourceProject.PRESENCE)
    results = [(first, 0.9), (second, 0.8), (third, 0.7)]

    reranked = rerank_for_source_diversity(results, max_per_source=1)

    assert reranked == [(first, 0.9), (third, 0.7), (second, 0.8)]
    assert reranked[0] is results[0]
    assert reranked[1] is results[2]
    assert reranked[2] is results[1]


def test_rerank_for_source_diversity_preserves_unit_payload_shape_and_uses_metadata_fallback():
    first = KnowledgeUnit.model_construct(
        id="external-1",
        source_id="source-external-1",
        source_entity_type="insight",
        title="External 1",
        content="Content for external 1",
        content_type=ContentType.INSIGHT,
        metadata={"origin": "external"},
    )
    second = KnowledgeUnit.model_construct(
        id="external-2",
        source_id="source-external-2",
        source_entity_type="insight",
        title="External 2",
        content="Content for external 2",
        content_type=ContentType.INSIGHT,
        metadata={"origin": "external"},
    )
    third = KnowledgeUnit.model_construct(
        id="local-1",
        source_id="source-local-1",
        source_entity_type="insight",
        title="Local 1",
        content="Content for local 1",
        content_type=ContentType.INSIGHT,
        metadata={},
    )

    reranked = rerank_for_source_diversity(
        [first, second, third],
        source_key="origin",
        max_per_source=1,
    )

    assert reranked == [first, third, second]
    assert all(not isinstance(item, tuple) for item in reranked)
    assert reranked[0] is first
    assert reranked[1] is third
    assert reranked[2] is second


def test_rerank_for_source_diversity_is_deterministic_across_reversed_equivalent_inputs():
    results = [
        (unit("unit-c", SourceProject.MAX), 0.5),
        (unit("unit-a", SourceProject.PRESENCE), 0.5),
        (unit("unit-b", SourceProject.MAX), 0.5),
        (unit("unit-d", SourceProject.FORTY_TWO), 0.5),
    ]

    first = rerank_for_source_diversity(results, max_per_source=1)
    second = rerank_for_source_diversity(reversed(results), max_per_source=1)

    assert first == second
    assert unit_ids(first) == ["unit-a", "unit-b", "unit-d", "unit-c"]


def test_rerank_for_source_diversity_accepts_zero_limit():
    results = [(unit("max-1", SourceProject.MAX), 0.9)]

    assert rerank_for_source_diversity(results, limit=0) == []


@pytest.mark.parametrize("max_per_source", [0, -1, 1.5, "2", True])
def test_rerank_for_source_diversity_rejects_invalid_max_per_source(max_per_source):
    with pytest.raises(ValueError, match="max_per_source must be a positive integer"):
        rerank_for_source_diversity([], max_per_source=max_per_source)


@pytest.mark.parametrize("limit", [-1, 1.5, "2", True])
def test_rerank_for_source_diversity_rejects_invalid_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        rerank_for_source_diversity([], limit=limit)


@pytest.mark.parametrize("source_key", ["", 1, None])
def test_rerank_for_source_diversity_rejects_invalid_source_key(source_key):
    with pytest.raises(ValueError, match="source_key must be a non-empty string"):
        rerank_for_source_diversity([], source_key=source_key)
