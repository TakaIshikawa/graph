from __future__ import annotations

import pytest

from graph.rag import build_answer_outline
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
    )


def test_build_answer_outline_groups_by_tags_source_and_query_terms():
    results = [
        {
            "id": "solar-storage",
            "title": "Solar storage field notes",
            "content": "Storage deployments improve grid resilience.",
            "source_project": "max",
            "tags": ["Solar", "storage"],
        },
        {
            "id": "battery-costs",
            "title": "Battery cost update",
            "content": "Cost curves for storage procurement.",
            "source_project": "presence",
            "tags": ["storage", "battery"],
        },
        {
            "id": "policy",
            "title": "Policy memo",
            "content": "Interconnection rules affect utility planning.",
            "source_project": "presence",
            "tags": ["regulation"],
        },
    ]

    outline = build_answer_outline(results, "solar storage policy")

    assert outline["query_terms"] == ["solar", "storage", "policy"]
    assert outline["sections"] == [
        {
            "title": "storage + battery",
            "rationale": (
                "shared tags: storage; sources: max, presence; "
                "query coverage: solar, storage"
            ),
            "evidence_result_ids": ["solar-storage", "battery-costs"],
            "coverage_terms": ["solar", "storage"],
            "missing_terms": ["policy"],
        },
        {
            "title": "regulation",
            "rationale": "shared tags: regulation; sources: presence; query coverage: policy",
            "evidence_result_ids": ["policy"],
            "coverage_terms": ["policy"],
            "missing_terms": ["solar", "storage"],
        },
    ]
    assert outline["missing_terms"] == []


def test_build_answer_outline_supports_nested_units_and_missing_terms():
    outline = build_answer_outline(
        [
            {"score": 0.9, "unit": unit("unit-a", "Solar finance", "Capital costs")},
            {"id": "flat", "title": "Thermal", "content": "District heating"},
        ],
        "solar storage finance",
    )

    assert outline["sections"][0]["evidence_result_ids"] == ["unit-a"]
    assert outline["sections"][0]["coverage_terms"] == ["solar", "finance"]
    assert outline["missing_terms"] == ["storage"]


def test_build_answer_outline_applies_section_and_evidence_limits():
    results = [
        {"id": "a1", "title": "Alpha one", "content": "alpha", "tags": ["alpha"]},
        {"id": "a2", "title": "Alpha two", "content": "alpha", "tags": ["alpha"]},
        {"id": "b1", "title": "Beta one", "content": "beta", "tags": ["beta"]},
        {"id": "c1", "title": "Gamma one", "content": "gamma", "tags": ["gamma"]},
    ]

    outline = build_answer_outline(
        results,
        "alpha beta gamma",
        max_sections=2,
        max_evidence_per_section=1,
    )

    assert [section["title"] for section in outline["sections"]] == ["alpha", "beta"]
    assert outline["sections"][0]["evidence_result_ids"] == ["a1"]
    assert outline["missing_terms"] == ["gamma"]


def test_build_answer_outline_is_stable_for_reordered_low_score_inputs():
    results = [
        {"id": "b", "title": "Battery", "content": "storage", "tags": ["storage"]},
        {"id": "a", "title": "Solar", "content": "solar storage", "tags": ["storage"]},
        {"id": "c", "title": "Policy", "content": "policy", "tags": ["policy"]},
    ]

    first = build_answer_outline(results, "solar storage policy")
    second = build_answer_outline(list(reversed(results)), "solar storage policy")

    assert first == second


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sections": 0},
        {"max_sections": True},
        {"max_evidence_per_section": 0},
        {"max_evidence_per_section": "3"},
    ],
)
def test_build_answer_outline_validates_limits(kwargs: dict):
    with pytest.raises(ValueError):
        build_answer_outline([], "query", **kwargs)
