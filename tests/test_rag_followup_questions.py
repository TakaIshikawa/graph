from __future__ import annotations

from graph.rag.followup_questions import build_followup_questions


def test_build_followup_questions_handles_empty_results():
    assert build_followup_questions("solar storage", []) == [
        {
            "question": "What broader sources or keywords should be searched for solar storage?",
            "reason": "no retrieval results were provided",
            "priority": 100,
        }
    ]


def test_build_followup_questions_uses_required_facets_for_missing_values():
    questions = build_followup_questions(
        "battery policy",
        [
            {
                "id": "a",
                "url": "https://example.test/a",
                "source_project": "web",
                "tags": ["energy"],
                "content_type": "article",
                "content": "Short",
            }
        ],
        required_facets={
            "tags": ["energy", "finance"],
            "domains": ["example.test", "agency.gov"],
            "source_projects": ["web", "notes"],
            "content_types": ["article", "dataset"],
        },
    )

    assert [question["priority"] for question in questions[:4]] == [90, 90, 90, 90]
    assert {question["reason"] for question in questions[:4]} == {
        "required tags were not present in the retrieved results",
        "required domains were not present in the retrieved results",
        "required source_projects were not present in the retrieved results",
        "required content_types were not present in the retrieved results",
    }


def test_build_followup_questions_flags_sparse_stale_and_low_diversity_results():
    questions = build_followup_questions(
        "roadmap",
        [
            {"id": "a", "url": "https://same.test/a", "content": "Short", "date": "2024-01-01"},
            {"id": "b", "url": "https://same.test/b", "content": "Also short", "date": "2023-01-01"},
        ],
    )

    assert [question["reason"] for question in questions] == [
        "2 result(s) have sparse content",
        "retrieved results have low source diversity",
        "all dated evidence is older than 2025",
        "some results are missing source project provenance",
    ]


def test_build_followup_questions_respects_max_questions():
    questions = build_followup_questions(
        "roadmap",
        [{"id": "a", "content": "Short"}],
        required_facets={"tags": ["x"], "domains": ["example.test"]},
        max_questions=2,
    )

    assert len(questions) == 2
    assert all(question["priority"] == 90 for question in questions)
