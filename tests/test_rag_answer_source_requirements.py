from __future__ import annotations

from dataclasses import dataclass

from graph.rag.answer_source_requirements import plan_answer_source_requirements


@dataclass
class ResultObject:
    source: str


def test_answer_source_requirements_neutral_query_has_minimal_requirements():
    payload = plan_answer_source_requirements("Summarize the onboarding notes")

    assert payload == {
        "normalized_query": "Summarize the onboarding notes",
        "minimum_sources": 1,
        "require_citations": False,
        "require_source_diversity": False,
        "source_gap": False,
        "available_source_count": None,
        "reasons": [],
    }


def test_answer_source_requirements_increases_for_comparison_and_latest():
    payload = plan_answer_source_requirements("Compare the latest options and cite sources")

    assert payload["minimum_sources"] == 3
    assert payload["require_citations"] is True
    assert payload["require_source_diversity"] is True
    assert payload["reasons"] == [
        "comparison_query",
        "latest_or_current_query",
        "citation_requested",
    ]


def test_answer_source_requirements_increases_for_high_stakes_and_quantitative_queries():
    payload = plan_answer_source_requirements("What is the average medical claim denial rate in 2025?")

    assert payload["minimum_sources"] == 3
    assert payload["require_citations"] is True
    assert payload["require_source_diversity"] is True
    assert payload["reasons"] == [
        "medical_legal_or_financial_query",
        "quantitative_query",
    ]


def test_answer_source_requirements_reports_source_gap_from_result_sources():
    payload = plan_answer_source_requirements(
        "Compare financial options",
        results=[
            {"id": "a", "metadata": {"source_id": "source-a"}},
            ResultObject("source-a"),
            ({"id": "c", "url": "https://example.com/report"}, 0.4),
        ],
    )

    assert payload["minimum_sources"] == 3
    assert payload["available_source_count"] == 2
    assert payload["source_gap"] is True


def test_answer_source_requirements_source_gap_is_false_when_enough_sources_exist():
    payload = plan_answer_source_requirements(
        "latest statistics with references",
        results=[
            {"source": "a"},
            {"metadata": {"source_name": "b"}},
            {"domain": "c.example"},
        ],
    )

    assert payload["minimum_sources"] == 2
    assert payload["available_source_count"] == 3
    assert payload["source_gap"] is False
