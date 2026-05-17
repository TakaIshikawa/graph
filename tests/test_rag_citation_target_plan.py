from __future__ import annotations

from graph.rag import build_citation_target_plan


def test_citation_target_plan_merges_duplicate_urls_and_ranks_evidence():
    rows = build_citation_target_plan(
        [
            {
                "id": "a",
                "title": "Alpha",
                "url": "https://example.test/a",
                "snippets": ["one", "two"],
                "score": 0.7,
            },
            {
                "id": "b",
                "title": "Alpha duplicate",
                "url": "https://example.test/a",
                "citations": ["ref"],
                "score": 0.9,
            },
            {"id": "c", "source": "Archive", "title": "Beta", "snippet": "single"},
        ]
    )

    assert rows[0]["source_key"] == "https://example.test/a"
    assert rows[0]["result_ids"] == ["a", "b"]
    assert rows[0]["evidence_count"] == 3
    assert rows[0]["best_score"] == 0.9
    assert "evidence" in rows[0]["recommendation_reason"]
    assert rows[1]["source_key"] == "Archive: Beta"


def test_citation_target_plan_is_deterministic_for_sparse_results():
    rows = build_citation_target_plan([{"id": "b", "title": "Beta"}, {"id": "a", "title": "Alpha"}])

    assert [row["source_key"] for row in rows] == ["Alpha", "Beta"]
    assert rows[0]["recommendation_reason"] == "limited citation metadata"
