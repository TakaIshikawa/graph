from __future__ import annotations

import pytest

from graph.rag import rank_duplicate_candidates
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str,
    *,
    source_id: str | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def test_exact_source_id_match_ranks_as_high_confidence_candidate():
    units = [
        unit("unit-b", "Copied note", "One version of the note.", source_id="external-1"),
        unit("unit-a", "Imported note", "Another version of the note.", source_id="external-1"),
        unit("unit-c", "Different note", "Nothing similar here."),
    ]

    results = rank_duplicate_candidates(units)

    assert results == [
        {
            "unit_ids": ["unit-a", "unit-b"],
            "score": 0.99,
            "reasons": ["source_id"],
            "matching_fields": {
                "source_id": "external-1",
                "title_similarity": 0.666667,
                "content_token_overlap": 0.5,
            },
        }
    ]


def test_normalized_url_match_ranks_as_high_confidence_candidate():
    units = [
        unit(
            "unit-a",
            "Original bookmark",
            "Short bookmark note.",
            metadata={"url": "https://www.example.com/articles/solar/?utm_source=newsletter"},
        ),
        unit(
            "unit-b",
            "Imported bookmark",
            "Different summary.",
            metadata={"canonical_url": "https://example.com/articles/solar"},
        ),
    ]

    results = rank_duplicate_candidates(units)

    assert results == [
        {
            "unit_ids": ["unit-a", "unit-b"],
            "score": 0.98,
            "reasons": ["url"],
            "matching_fields": {
                "urls": ["https://example.com/articles/solar"],
                "title_similarity": 0.647059,
            },
        }
    ]


def test_near_identical_title_and_content_produce_candidate_above_threshold():
    units = [
        unit(
            "unit-a",
            "Solar storage roadmap",
            "Battery storage economics improve when grid demand forecasting gets better.",
        ),
        unit(
            "unit-b",
            "Solar storage roadmap updated",
            "Battery storage economics improve when grid demand forecasts get better.",
        ),
        unit("unit-c", "Python async patterns", "Structured concurrency improves services."),
    ]

    results = rank_duplicate_candidates(units)

    assert len(results) == 1
    assert results[0]["unit_ids"] == ["unit-a", "unit-b"]
    assert results[0]["score"] > 0.82
    assert results[0]["reasons"] == ["title"]
    assert results[0]["matching_fields"]["title_similarity"] == 1.0
    assert results[0]["matching_fields"]["content_token_overlap"] == 0.666667


def test_unrelated_units_are_not_returned():
    units = [
        unit("unit-a", "Solar storage roadmap", "Battery economics and grid operations."),
        unit("unit-b", "Python async patterns", "Structured concurrency improves services."),
        unit("unit-c", "Meal planning", "Pantry inventory and weekly recipes."),
    ]

    assert rank_duplicate_candidates(units) == []


def test_results_are_deterministic_sorted_and_limited():
    units = [
        unit("unit-c", "Shared URL C", "One note.", metadata={"url": "https://example.com/c"}),
        unit("unit-a", "Shared URL A", "One note.", metadata={"url": "https://example.com/a"}),
        unit("unit-d", "Shared URL C2", "Two note.", metadata={"url": "https://example.com/c/"}),
        unit("unit-b", "Shared URL A2", "Two note.", metadata={"url": "https://www.example.com/a/"}),
    ]

    first = rank_duplicate_candidates(units, limit=1)
    second = rank_duplicate_candidates(reversed(units), limit=1)

    assert first == second
    assert first == [
        {
            "unit_ids": ["unit-a", "unit-b"],
            "score": 0.98,
            "reasons": ["url", "title"],
            "matching_fields": {
                "urls": ["https://example.com/a"],
                "title_similarity": 0.96,
                "content_token_overlap": 0.333333,
            },
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": -0.1}, "threshold"),
        ({"threshold": 1.1}, "threshold"),
        ({"threshold": "0.8"}, "threshold"),
        ({"limit": 0}, "limit"),
        ({"limit": -1}, "limit"),
        ({"limit": "2"}, "limit"),
    ],
)
def test_rank_duplicate_candidates_validates_arguments(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        rank_duplicate_candidates([], **kwargs)
