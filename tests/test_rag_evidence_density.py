from __future__ import annotations

from graph.rag import score_evidence_density


def test_evidence_density_ranks_rich_results_above_sparse_results():
    rows = score_evidence_density(
        [
            {
                "id": "sparse",
                "title": "Sparse",
                "snippet": "Short note.",
            },
            {
                "id": "rich",
                "title": "Rich evidence",
                "content": "Detailed evidence with measured outcomes and supporting context.",
                "metadata": {
                    "source": "Journal",
                    "url": "https://example.test/paper",
                    "author": "Ada",
                    "published_at": "2026-04-01",
                    "citations": ["Smith 2026", "Doe 2025"],
                    "relations": ["supports", "extends"],
                    "source_count": 3,
                },
            },
        ]
    )

    assert [row["result_id"] for row in rows] == ["rich", "sparse"]
    assert rows[0] == {
        "result_id": "rich",
        "title": "Rich evidence",
        "density_score": 11.1,
        "text_word_count": 10,
        "citation_count": 3,
        "metadata_field_count": 4,
        "relation_count": 2,
        "source_count": 3,
        "graph_context_count": 5,
    }
    assert rows[1]["density_score"] == 0.03


def test_evidence_density_includes_explainable_components():
    rows = score_evidence_density(
        [
            {
                "id": "components",
                "title": "Components",
                "text": "Alpha beta gamma",
                "references": ["A"],
                "doi": "10.1000/example",
                "metadata": {"source_name": "Archive", "created_at": "2026-05-01"},
                "relation_count": 4,
                "sources": ["one", "two"],
            }
        ]
    )

    assert rows == [
        {
            "result_id": "components",
            "title": "Components",
            "density_score": 8.04,
            "text_word_count": 4,
            "citation_count": 2,
            "metadata_field_count": 2,
            "relation_count": 4,
            "source_count": 2,
            "graph_context_count": 6,
        }
    ]


def test_evidence_density_supports_plain_nested_units_and_empty_input():
    assert score_evidence_density([]) == []

    rows = score_evidence_density(
        [
            {
                "unit": {
                    "id": "nested",
                    "title": "Nested",
                    "content": "Nested evidence",
                    "metadata": {"source_url": "https://example.test", "authors": ["Ada"]},
                }
            }
        ]
    )

    assert rows[0]["result_id"] == "nested"
    assert rows[0]["citation_count"] == 1
    assert rows[0]["metadata_field_count"] == 2


def test_evidence_density_sorts_deterministically():
    results = [
        {"id": "b", "title": "Beta", "content": "same"},
        {"id": "a", "title": "Alpha", "content": "same"},
    ]

    assert score_evidence_density(results) == score_evidence_density(reversed(results))
    assert [row["result_id"] for row in score_evidence_density(results)] == ["a", "b"]
