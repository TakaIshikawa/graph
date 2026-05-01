from __future__ import annotations

import pytest

from graph.rag import suggest_tag_normalizations
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_suggest_tag_normalizations_groups_case_variants_without_mutating_units():
    units = [
        unit("unit-b", ["AI Ethics", "research"]),
        unit("unit-a", ["ai ethics"]),
        unit("unit-c", ["Ai Ethics"]),
    ]
    original_tags = [list(item.tags) for item in units]

    assert suggest_tag_normalizations(units) == [
        {
            "canonical_tag": "ai ethics",
            "variants": ["AI Ethics", "Ai Ethics"],
            "counts": {"ai ethics": 1, "AI Ethics": 1, "Ai Ethics": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-a", "unit-b", "unit-c"],
        }
    ]
    assert [item.tags for item in units] == original_tags


def test_suggest_tag_normalizations_groups_punctuation_variants():
    units = [
        unit("unit-a", ["solar-energy"]),
        unit("unit-b", ["solar energy"]),
        unit("unit-c", ["solar_energy"]),
        unit("unit-d", ["solar/storage"]),
    ]

    assert suggest_tag_normalizations(units) == [
        {
            "canonical_tag": "solar energy",
            "variants": ["solar-energy", "solar_energy"],
            "counts": {"solar energy": 1, "solar-energy": 1, "solar_energy": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-a", "unit-b", "unit-c"],
        }
    ]


def test_suggest_tag_normalizations_groups_plural_variants():
    units = [
        unit("unit-a", ["batteries"]),
        unit("unit-b", ["battery"]),
        unit("unit-c", ["graphs"]),
        unit("unit-d", ["graph"]),
    ]

    assert suggest_tag_normalizations(units) == [
        {
            "canonical_tag": "batteries",
            "variants": ["battery"],
            "counts": {"batteries": 1, "battery": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-a", "unit-b"],
        },
        {
            "canonical_tag": "graph",
            "variants": ["graphs"],
            "counts": {"graph": 1, "graphs": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-c", "unit-d"],
        },
    ]


def test_suggest_tag_normalizations_respects_count_and_similarity_thresholds():
    units = [
        unit("unit-a", ["cache invalidation"]),
        unit("unit-b", ["cache-invalidation"]),
        unit("unit-c", ["cache validation"]),
        unit("unit-d", ["cache validations"]),
    ]

    assert suggest_tag_normalizations(units, min_count=3) == []
    assert suggest_tag_normalizations(units, min_similarity=0.96) == [
        {
            "canonical_tag": "cache invalidation",
            "variants": ["cache-invalidation"],
            "counts": {"cache invalidation": 1, "cache-invalidation": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-a", "unit-b"],
        },
        {
            "canonical_tag": "cache validation",
            "variants": ["cache validations"],
            "counts": {"cache validation": 1, "cache validations": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-c", "unit-d"],
        },
    ]


def test_suggest_tag_normalizations_respects_limit():
    units = [
        unit("unit-a", ["alpha tag", "beta-tag", "gamma tag"]),
        unit("unit-b", ["alpha-tag", "beta tag", "gamma-tag"]),
    ]

    assert suggest_tag_normalizations(units, limit=2) == [
        {
            "canonical_tag": "alpha tag",
            "variants": ["alpha-tag"],
            "counts": {"alpha tag": 1, "alpha-tag": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-a", "unit-b"],
        },
        {
            "canonical_tag": "beta tag",
            "variants": ["beta-tag"],
            "counts": {"beta tag": 1, "beta-tag": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-a", "unit-b"],
        },
    ]


def test_suggest_tag_normalizations_is_deterministic_across_input_ordering():
    units = [
        unit("unit-c", ["JavaScript"]),
        unit("unit-a", ["java script"]),
        unit("unit-b", ["JS tooling"]),
        unit("unit-d", ["js-tooling"]),
    ]

    first = suggest_tag_normalizations(units)
    second = suggest_tag_normalizations(reversed(units))

    assert first == second
    assert first == [
        {
            "canonical_tag": "JS tooling",
            "variants": ["js-tooling"],
            "counts": {"JS tooling": 1, "js-tooling": 1},
            "similarity": 1.0,
            "affected_unit_ids": ["unit-b", "unit-d"],
        },
        {
            "canonical_tag": "java script",
            "variants": ["JavaScript"],
            "counts": {"java script": 1, "JavaScript": 1},
            "similarity": 0.952381,
            "affected_unit_ids": ["unit-a", "unit-c"],
        },
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_count": 0}, "min_count"),
        ({"min_count": "2"}, "min_count"),
        ({"min_similarity": -0.1}, "min_similarity"),
        ({"min_similarity": 1.1}, "min_similarity"),
        ({"min_similarity": "0.8"}, "min_similarity"),
        ({"limit": 0}, "limit"),
        ({"limit": -1}, "limit"),
        ({"limit": "2"}, "limit"),
    ],
)
def test_suggest_tag_normalizations_validates_arguments(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        suggest_tag_normalizations([], **kwargs)
