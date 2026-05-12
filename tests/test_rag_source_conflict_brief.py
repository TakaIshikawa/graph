from __future__ import annotations

import pytest

from graph.rag.source_conflict_brief import build_source_conflict_brief
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def _unit(
    unit_id: str,
    source_project: SourceProject | str,
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="claim",
        title=title,
        content=content,
        tags=tags or [],
    )


def _by_term(rows: list[dict], term: str) -> dict:
    return next(row for row in rows if row["term"] == term)


def test_source_conflict_brief_groups_units_by_normalized_terms_and_cues():
    rows = build_source_conflict_brief(
        [
            _unit(
                "b",
                SourceProject.PRESENCE,
                "Battery economics",
                "Battery storage is cost effective for peak demand.",
                ["Grid Storage"],
            ),
            _unit(
                "a",
                SourceProject.MAX,
                "Battery economics",
                "Battery storage is not cost effective for peak demand.",
                ["grid storage"],
            ),
            _unit(
                "c",
                SourceProject.MAX,
                "Unrelated",
                "Solar exports increased.",
                ["solar"],
            ),
        ]
    )

    row = _by_term(rows, "grid storage")
    assert row["source_projects"] == ["max", "presence"]
    assert row["supporting_unit_ids"] == ["a", "b"]
    assert row["source_project_count"] == 2
    assert row["unit_count"] == 2
    assert row["has_disagreement_cue"] is True
    assert row["confidence"] == "high"
    assert row["disagreement_cues"] == ["not"]
    assert row["claim_snippets"] == [
        {
            "unit_id": "a",
            "snippet": "Battery storage is not cost effective for peak demand.",
        },
        {
            "unit_id": "b",
            "snippet": "Battery storage is cost effective for peak demand.",
        },
    ]


def test_source_conflict_brief_accepts_search_result_dictionaries_and_nested_units():
    rows = build_source_conflict_brief(
        [
            {
                "unit_id": "search-a",
                "source_project": "paper",
                "claim": "The trial contradicts earlier retention claims.",
                "metadata": {"keywords": [{"keyword": "Retention"}]},
            },
            {
                "id": "wrapper",
                "source_project": "notes",
                "unit": _unit(
                    "nested",
                    "ignored",
                    "Retention",
                    "Retention improved in the follow-up cohort.",
                    ["retention"],
                ),
            },
        ]
    )

    assert rows == [
        {
            "term": "retention",
            "source_projects": ["notes", "paper"],
            "source_project_count": 2,
            "supporting_unit_ids": ["search-a", "wrapper"],
            "unit_count": 2,
            "claim_snippets": [
                {
                    "unit_id": "search-a",
                    "snippet": "The trial contradicts earlier retention claims.",
                },
                {
                    "unit_id": "wrapper",
                    "snippet": "Retention improved in the follow-up cohort.",
                },
            ],
            "disagreement_cues": ["contradicts"],
            "has_disagreement_cue": True,
            "confidence": "high",
        }
    ]


def test_source_conflict_brief_is_deterministic_and_can_limit_rows():
    results = [
        _unit("z", "beta", "Zeta", "Alpha topic is disputed.", ["alpha"]),
        _unit("a", "alpha", "Alpha", "Alpha topic is stable.", ["alpha"]),
        _unit("b", "alpha", "Beta", "Beta topic is stable.", ["beta"]),
        _unit("y", "beta", "Beta", "Beta topic is stable.", ["beta"]),
    ]

    first = build_source_conflict_brief(results, limit=1)
    second = build_source_conflict_brief(reversed(results), limit=1)

    assert first == second
    assert [row["term"] for row in first] == ["alpha"]


def test_source_conflict_brief_empty_or_single_source_inputs_return_no_conflicts():
    assert build_source_conflict_brief([]) == []
    assert build_source_conflict_brief(
        [
            _unit("a", "notes", "A", "The claim is disputed.", ["claim"]),
            _unit("b", "notes", "B", "The claim is supported.", ["claim"]),
        ]
    ) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_source_count": 0}, "min_source_count must be a positive integer"),
        ({"min_source_count": True}, "min_source_count must be a positive integer"),
        ({"limit": -1}, "limit must be a non-negative integer or None"),
        ({"limit": True}, "limit must be a non-negative integer or None"),
        ({"min_term_length": 0}, "min_term_length must be a positive integer"),
    ],
)
def test_source_conflict_brief_validates_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_source_conflict_brief([], **kwargs)
