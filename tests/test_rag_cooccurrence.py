from __future__ import annotations

import pytest

from graph.rag import build_keyphrase_cooccurrence
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, title: str, content: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
    )


def sample_units() -> list[KnowledgeUnit]:
    return [
        unit("unit-a", "Solar Storage Plan", "solar grid storage"),
        unit("unit-b", "Solar Finance", "storage finance"),
        unit("unit-c", "Grid Storage", "grid storage"),
        unit("unit-d", "Wind Note", "offshore market"),
    ]


def test_build_keyphrase_cooccurrence_returns_ranked_phrase_records_with_counts():
    result = build_keyphrase_cooccurrence(sample_units(), max_phrases=3)

    assert result["phrases"] == [
        {
            "phrase": "storage",
            "score": 9,
            "count": 5,
            "unit_count": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
        },
        {
            "phrase": "solar",
            "score": 7,
            "count": 3,
            "unit_count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
        {
            "phrase": "grid",
            "score": 5,
            "count": 3,
            "unit_count": 2,
            "unit_ids": ["unit-a", "unit-c"],
        },
    ]
    assert result["stats"] == {
        "unit_count": 4,
        "phrase_count": 3,
        "edge_count": 3,
        "min_count": 2,
        "max_phrases": 3,
    }


def test_build_keyphrase_cooccurrence_weights_pairs_that_share_units():
    result = build_keyphrase_cooccurrence(sample_units())

    assert result["edges"] == [
        {
            "source": "grid",
            "target": "storage",
            "weight": 2,
            "unit_ids": ["unit-a", "unit-c"],
        },
        {
            "source": "solar",
            "target": "storage",
            "weight": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
        {
            "source": "finance",
            "target": "solar",
            "weight": 1,
            "unit_ids": ["unit-b"],
        },
        {
            "source": "finance",
            "target": "storage",
            "weight": 1,
            "unit_ids": ["unit-b"],
        },
        {
            "source": "grid",
            "target": "solar",
            "weight": 1,
            "unit_ids": ["unit-a"],
        },
    ]
    assert result["stats"]["edge_count"] == 5


def test_build_keyphrase_cooccurrence_is_deterministic_and_applies_phrase_limit():
    first = build_keyphrase_cooccurrence(sample_units(), max_phrases=2)
    second = build_keyphrase_cooccurrence(reversed(sample_units()), max_phrases=2)

    assert first == second
    assert [phrase["phrase"] for phrase in first["phrases"]] == ["storage", "solar"]
    assert first["edges"] == [
        {
            "source": "solar",
            "target": "storage",
            "weight": 2,
            "unit_ids": ["unit-a", "unit-b"],
        }
    ]


def test_build_keyphrase_cooccurrence_accepts_zero_max_phrases():
    result = build_keyphrase_cooccurrence(sample_units(), max_phrases=0)

    assert result == {
        "phrases": [],
        "edges": [],
        "stats": {
            "unit_count": 4,
            "phrase_count": 0,
            "edge_count": 0,
            "min_count": 2,
            "max_phrases": 0,
        },
    }


@pytest.mark.parametrize("min_count", [-1, "bad", True])
def test_build_keyphrase_cooccurrence_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a non-negative integer"):
        build_keyphrase_cooccurrence([], min_count=min_count)


@pytest.mark.parametrize("max_phrases", [-1, "bad", True])
def test_build_keyphrase_cooccurrence_validates_max_phrases(max_phrases):
    with pytest.raises(ValueError, match="max_phrases must be a non-negative integer"):
        build_keyphrase_cooccurrence([], max_phrases=max_phrases)
