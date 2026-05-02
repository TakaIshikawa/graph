from __future__ import annotations

import pytest

from graph.rag import build_result_coverage_checklist
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(
    unit_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        metadata=metadata or {},
    )


def by_term(rows: list[dict], term: str) -> dict:
    return next(row for row in rows if row["term"] == term)


def test_build_result_coverage_checklist_matches_title_content_tags_and_metadata():
    report = build_result_coverage_checklist(
        "Does solar storage need finance and resilience?",
        [
            unit(
                "unit-a",
                "Solar storage plan",
                "Grid batteries improve reliability for cloudy weeks.",
                ["deployment"],
            ),
            {
                "id": "unit-b",
                "title": "Adoption memo",
                "content": "Revenue risk remains high.",
                "tags": ["Finance"],
                "metadata": {"themes": ["community resilience"]},
            },
        ],
    )

    assert report["query_terms"] == [
        "solar",
        "storage",
        "need",
        "finance",
        "resilience",
    ]
    assert report["uncovered"] == ["need"]
    assert by_term(report["covered"], "solar") == {
        "term": "solar",
        "supporting_unit_ids": ["unit-a"],
        "snippets": [
            {
                "unit_id": "unit-a",
                "field": "title",
                "snippet": "Solar storage plan",
            }
        ],
    }
    assert by_term(report["covered"], "finance")["snippets"] == [
        {"unit_id": "unit-b", "field": "tag", "snippet": "Finance"}
    ]
    assert by_term(report["covered"], "resilience")["snippets"] == [
        {
            "unit_id": "unit-b",
            "field": "metadata",
            "snippet": "community resilience",
        }
    ]


def test_covered_items_include_sorted_unit_ids_and_short_snippets():
    report = build_result_coverage_checklist(
        "storage",
        [
            unit(
                "unit-b",
                "Battery memo",
                "Battery storage can shift renewable generation across peak demand.",
            ),
            unit("unit-a", "Storage strategy", "A short note."),
        ],
        snippet_chars=32,
    )

    storage = by_term(report["covered"], "storage")
    assert storage["supporting_unit_ids"] == ["unit-a", "unit-b"]
    assert storage["snippets"] == [
        {"unit_id": "unit-a", "field": "title", "snippet": "Storage strategy"},
        {
            "unit_id": "unit-b",
            "field": "content",
            "snippet": "Battery storage can shift renewa",
        },
    ]


def test_custom_stopwords_and_min_token_length_control_query_items():
    report = build_result_coverage_checklist(
        "AI grid and solar risk",
        [unit("unit-a", "Grid risk", "Solar risk model.")],
        stopwords={"risk"},
        min_token_length=2,
    )

    assert report["query_terms"] == ["ai", "grid", "solar"]
    assert report["uncovered"] == ["ai"]
    assert [row["term"] for row in report["covered"]] == ["grid", "solar"]


def test_nested_unit_payloads_and_tuple_results_are_supported():
    report = build_result_coverage_checklist(
        "forecast weather",
        [
            (
                {
                    "unit": {
                        "id": "unit-a",
                        "title": "Load forecast",
                        "content": "Brief.",
                        "metadata": {
                            "signals": {"forecast_driver": "weather anomaly"}
                        },
                    }
                },
                0.8,
            )
        ],
    )

    assert by_term(report["covered"], "forecast")["supporting_unit_ids"] == ["unit-a"]
    assert by_term(report["covered"], "weather")["snippets"] == [
        {
            "unit_id": "unit-a",
            "field": "metadata",
            "snippet": "weather anomaly",
        }
    ]


def test_matching_uses_exact_normalized_tokens_not_substrings():
    report = build_result_coverage_checklist(
        "grid",
        [{"id": "unit-a", "title": "Microgrid rollout", "content": "brief"}],
    )

    assert report["covered"] == []
    assert report["uncovered"] == ["grid"]


def test_build_result_coverage_checklist_is_deterministic_for_result_order():
    first = build_result_coverage_checklist(
        "solar storage",
        [
            unit("unit-b", "Storage note", "Solar detail."),
            unit("unit-a", "Solar plan", "Storage detail."),
        ],
    )
    second = build_result_coverage_checklist(
        "solar storage",
        [
            unit("unit-a", "Solar plan", "Storage detail."),
            unit("unit-b", "Storage note", "Solar detail."),
        ],
    )

    assert first == second


def test_build_result_coverage_checklist_is_importable_from_graph_rag():
    assert callable(build_result_coverage_checklist)


@pytest.mark.parametrize("min_token_length", [0, -1, "3", True])
def test_build_result_coverage_checklist_validates_min_token_length(min_token_length):
    with pytest.raises(ValueError, match="min_token_length must be a positive integer"):
        build_result_coverage_checklist("solar", [], min_token_length=min_token_length)


@pytest.mark.parametrize("snippet_chars", [0, -1, "3", True])
def test_build_result_coverage_checklist_validates_snippet_chars(snippet_chars):
    with pytest.raises(ValueError, match="snippet_chars must be a positive integer"):
        build_result_coverage_checklist("solar", [], snippet_chars=snippet_chars)
