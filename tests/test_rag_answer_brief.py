from __future__ import annotations

from graph.rag.answer_brief import build_answer_readiness_brief


def test_build_answer_readiness_brief_scores_supported_results():
    brief = build_answer_readiness_brief(
        "retrieval quality",
        [
            {
                "id": "a",
                "title": "Alpha",
                "content": "Retrieval quality depends on cited evidence.",
                "source_project": "max",
                "url": "https://example.com/a",
                "created_at": "2026-05-01T10:00:00Z",
                "tags": ["rag"],
                "score": 0.9,
            }
        ],
        required_facets={"tags": ["rag"]},
    )

    assert brief["readiness_label"] == "ready"
    assert brief["readiness_score"] >= 0.8
    assert brief["blocking_gaps"] == []
    assert brief["coverage"]["citation"]["with_citation_count"] == 1
    assert brief["date_coverage"]["dated_results"] == 1
    assert brief["supporting_sources"][0]["source"] == "max"
    assert brief["recommended_next_actions"] == [
        "Proceed with answer generation and cite the strongest packets."
    ]


def test_build_answer_readiness_brief_reports_empty_results_as_blocked():
    brief = build_answer_readiness_brief("missing context", [])

    assert brief["readiness_label"] == "blocked"
    assert brief["readiness_score"] == 0.0
    assert brief["supporting_sources"] == []
    assert brief["blocking_gaps"][0]["type"] == "empty_results"
    assert "Run a broader retrieval before drafting the answer." in brief[
        "recommended_next_actions"
    ]
