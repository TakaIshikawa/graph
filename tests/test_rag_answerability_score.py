from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graph.rag.answerability_score import score_answerability
from graph.types.models import KnowledgeUnit


NOW = datetime(2026, 5, 12, tzinfo=timezone.utc)


def test_score_answerability_returns_components_score_and_notes():
    payload = score_answerability(
        "latest adoption metrics",
        [
            {
                "id": "a",
                "source_project": "web",
                "title": "Latest adoption metrics",
                "content": "Adoption reached 42 percent.",
                "updated_at": "2026-05-10T00:00:00Z",
                "url": "https://example.test/a",
            },
            {
                "id": "b",
                "source_project": "notes",
                "title": "Adoption background",
                "content": "Older context.",
                "updated_at": "2026-02-01",
                "url": "https://example.test/b",
            },
        ],
        now=NOW,
    )

    assert payload == {
        "score": 0.925,
        "components": {
            "focus_term_coverage": 1.0,
            "source_diversity": 1.0,
            "freshness": 0.7,
            "attribution_completeness": 1.0,
        },
        "notes": [],
    }


def test_score_answerability_accepts_nested_units_objects_and_scored_tuples():
    unit = KnowledgeUnit(
        id="unit-1",
        source_project="readwise",
        source_id="source-1",
        source_entity_type="highlight",
        title="Graph search",
        content="Hybrid retrieval improves graph search.",
        metadata={"source_url": "https://book.test", "author": "Ada"},
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
    )
    wrapper = SimpleNamespace(id="wrapper", unit=unit)

    payload = score_answerability(
        "hybrid graph retrieval",
        [
            wrapper,
            (
                {
                    "source_id": "tuple",
                    "source": "archive",
                    "title": "Retrieval notes",
                    "content": "Hybrid retrieval notes.",
                    "date": "2026-04-03",
                    "link": "https://archive.test",
                },
                0.8,
            ),
        ],
        now=NOW,
    )

    assert payload["components"]["focus_term_coverage"] == 1.0
    assert payload["components"]["source_diversity"] == 1.0
    assert payload["components"]["freshness"] == 0.75
    assert payload["score"] == 0.938


def test_score_answerability_is_deterministic_for_same_inputs():
    results = [
        {"id": "a", "source_project": "notes", "title": "Alpha", "content": "Missing target"},
        {"id": "b", "source_project": "notes", "title": "Beta"},
    ]

    first = score_answerability("alpha target metric", results, now=NOW)
    second = score_answerability("alpha target metric", results, now=NOW)

    assert first == second
    assert first["notes"] == [
        "missing_focus_terms:metric",
        "limited_source_diversity",
        "no_dated_evidence",
        "incomplete_attribution",
    ]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_score_answerability_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        score_answerability(query, [], now=NOW)  # type: ignore[arg-type]
