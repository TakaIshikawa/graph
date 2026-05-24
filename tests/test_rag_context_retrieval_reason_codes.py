from __future__ import annotations

from graph.rag.context_retrieval_reason_codes import assign_context_retrieval_reason_codes


def test_context_retrieval_reason_codes_empty_input():
    assert assign_context_retrieval_reason_codes("solar policy", [])["records"] == []


def test_context_retrieval_reason_codes_assigns_compact_codes_and_labels():
    result = assign_context_retrieval_reason_codes(
        "solar policy",
        [
            {
                "id": "r1",
                "score": 0.9,
                "text": "Solar policy details",
                "date": "2025-01-01",
                "metadata": {"authority": "official", "url": "https://example.test"},
            }
        ],
    )

    row = result["records"][0]
    assert row["result_id"] == "r1"
    assert row["reason_codes"] == ["SCORE_HIGH", "QUERY_TERM", "RECENT", "AUTHORITY", "CITATION"]
    assert row["labels"][0] == "High retrieval score"


def test_context_retrieval_reason_codes_handles_missing_metadata():
    result = assign_context_retrieval_reason_codes("solar policy", [{"id": "r1"}])

    assert result["records"][0]["reason_codes"] == []


def test_context_retrieval_reason_codes_detects_entity_overlap():
    result = assign_context_retrieval_reason_codes("NASA budget", [{"id": "r1", "text": "NASA annual report"}])

    assert result["records"][0]["reason_codes"] == ["QUERY_TERM", "ENTITY"]
