from __future__ import annotations

from graph.rag import analyze_source_evidence_coverage


def test_source_evidence_coverage_groups_sources_and_missing_source():
    summary = analyze_source_evidence_coverage(
        [
            {"id": "a", "source": "Journal", "snippets": ["one", "two"], "url": "https://example.test"},
            {"id": "b", "metadata": {"source": "Journal"}, "snippet": "single"},
            {"id": "c", "content": "No source"},
        ]
    )

    assert summary == {
        "source_count": 2,
        "sources": [
            {
                "source_key": "__missing_source__",
                "result_count": 1,
                "evidence_count": 1,
                "cited_count": 0,
                "result_ids": ["c"],
            },
            {
                "source_key": "Journal",
                "result_count": 2,
                "evidence_count": 3,
                "cited_count": 1,
                "result_ids": ["a", "b"],
            },
        ],
    }
