from __future__ import annotations

import pytest

from graph.rag import score_source_agreement
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    source_project: SourceProject | str,
    title: str,
    content: str,
    tags: list[str],
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def by_key(rows: list[dict], evidence_type: str, evidence_key: str) -> dict:
    return next(
        row
        for row in rows
        if row["evidence_type"] == evidence_type
        and row["evidence_key"] == evidence_key
    )


def sample_units() -> list[KnowledgeUnit]:
    return [
        unit(
            "unit-a",
            SourceProject.MAX,
            "Solar storage plan",
            "Grid storage improves solar reliability.",
            ["Solar", "Storage"],
        ),
        unit(
            "unit-b",
            SourceProject.PRESENCE,
            "Solar finance update",
            "Storage finance improves adoption.",
            ["solar", "finance"],
        ),
        unit(
            "unit-c",
            SourceProject.FORTY_TWO,
            "Grid storage note",
            "Storage supports grid planning.",
            ["Grid", "Storage"],
        ),
    ]


def test_score_source_agreement_scores_tags_keywords_and_terms_across_sources():
    rows = score_source_agreement(sample_units(), min_source_count=2)

    tag = by_key(rows, "tag", "solar")
    assert tag == {
        "evidence_type": "tag",
        "evidence_key": "solar",
        "supporting_source_projects": ["max", "presence"],
        "supporting_unit_ids": ["unit-a", "unit-b"],
        "source_count": 2,
        "unit_count": 2,
        "agreement_score": pytest.approx(2 / 3),
    }

    keyword = by_key(rows, "keyword", "storage")
    term = by_key(rows, "term", "storage")
    assert keyword["supporting_source_projects"] == ["forty_two", "max", "presence"]
    assert keyword["supporting_unit_ids"] == ["unit-a", "unit-b", "unit-c"]
    assert keyword["agreement_score"] == 1.0
    assert term["supporting_source_projects"] == ["forty_two", "max", "presence"]
    assert term["agreement_score"] == 1.0


def test_agreement_score_increases_with_more_distinct_supporting_sources():
    rows = score_source_agreement(sample_units())

    solar = by_key(rows, "tag", "solar")
    storage = by_key(rows, "tag", "storage")

    assert storage["source_count"] == 2
    assert solar["source_count"] == 2
    assert storage["agreement_score"] == solar["agreement_score"]

    all_source_storage = by_key(rows, "term", "storage")
    assert all_source_storage["source_count"] == 3
    assert all_source_storage["agreement_score"] > solar["agreement_score"]


def test_min_source_count_filters_single_source_evidence():
    rows = score_source_agreement(sample_units(), min_source_count=2)

    assert not any(
        row["evidence_key"] == "finance" and row["evidence_type"] == "tag"
        for row in rows
    )
    assert by_key(rows, "tag", "solar")["source_count"] == 2


def test_explicit_keywords_are_used_when_present_on_mapping_results():
    rows = score_source_agreement(
        [
            {
                "id": "a",
                "source_project": "alpha",
                "title": "Alpha",
                "content": "brief",
                "metadata": {
                    "keywords": [
                        {"keyword": "Load Forecast", "score": 4},
                        {"keyword": "Grid", "score": 2},
                    ]
                },
            },
            {
                "id": "b",
                "source_project": "beta",
                "title": "Beta",
                "content": "brief",
                "keywords": ["load forecast"],
            },
        ],
        min_source_count=2,
    )

    assert by_key(rows, "keyword", "load forecast") == {
        "evidence_type": "keyword",
        "evidence_key": "load forecast",
        "supporting_source_projects": ["alpha", "beta"],
        "supporting_unit_ids": ["a", "b"],
        "source_count": 2,
        "unit_count": 2,
        "agreement_score": 1.0,
    }


def test_score_source_agreement_is_deterministic_and_can_limit_rows():
    first = score_source_agreement(sample_units(), min_source_count=2, limit=3)
    second = score_source_agreement(reversed(sample_units()), min_source_count=2, limit=3)

    assert first == second
    assert len(first) == 3


def test_score_source_agreement_is_importable_from_graph_rag():
    assert callable(score_source_agreement)


@pytest.mark.parametrize("min_source_count", [0, -1, "2", True])
def test_score_source_agreement_validates_min_source_count(min_source_count):
    with pytest.raises(ValueError, match="min_source_count must be a positive integer"):
        score_source_agreement([], min_source_count=min_source_count)


@pytest.mark.parametrize("limit", [-1, "2", True])
def test_score_source_agreement_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        score_source_agreement([], limit=limit)


@pytest.mark.parametrize("min_term_length", [0, -1, "3", True])
def test_score_source_agreement_validates_min_term_length(min_term_length):
    with pytest.raises(ValueError, match="min_term_length must be a positive integer"):
        score_source_agreement([], min_term_length=min_term_length)
