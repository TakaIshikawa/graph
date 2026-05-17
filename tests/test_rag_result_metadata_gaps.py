from __future__ import annotations

from graph.rag import summarize_result_metadata_gaps


def test_result_metadata_gaps_counts_empty_values_as_missing():
    summary = summarize_result_metadata_gaps(
        [
            {
                "id": "complete",
                "title": "Title",
                "source": "Journal",
                "url": "https://example.test",
                "author": "Ada",
                "published_at": "2026-05-01",
                "tags": ["rag"],
                "snippet": "Excerpt",
            },
            {"id": "missing", "title": " ", "metadata": {"tags": [], "author": None}},
        ],
        required_fields=["title", "author", "tags"],
    )

    assert summary == {
        "total_results": 2,
        "result_gaps": [
            {"result_id": "complete", "missing_fields": []},
            {"result_id": "missing", "missing_fields": ["title", "author", "tags"]},
        ],
        "missing_count_by_field": {"author": 1, "tags": 1, "title": 1},
    }


def test_result_metadata_gaps_supports_default_fields_and_nested_units():
    summary = summarize_result_metadata_gaps([{"unit": {"id": "nested", "title": "Nested", "metadata": {"url": ""}}}])

    assert summary["total_results"] == 1
    assert summary["result_gaps"][0]["result_id"] == "nested"
    assert "source" in summary["result_gaps"][0]["missing_fields"]
    assert summary["missing_count_by_field"]["source"] == 1
