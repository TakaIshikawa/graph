from __future__ import annotations

from graph.rag.result_retrieval_diagnostics import diagnose_result_retrieval


def test_result_retrieval_diagnostics_reports_aggregate_and_rows():
    report = diagnose_result_retrieval(
        "alpha beta release",
        [
            {"id": "a", "title": "Alpha beta release notes", "content": "Alpha beta release notes with detailed migration source.", "url": "https://docs.test/a"},
            {"id": "b", "title": "Alpha beta release notes", "content": "Alpha beta release notes with detailed migration source.", "url": "https://docs.test/a"},
            {"id": "c"},
        ],
    )

    assert report["result_count"] == 3
    assert report["missing_content_count"] == 1
    assert report["duplicate_like_count"] == 2
    assert report["per_result"][0]["result_id"] == "a"
    assert "duplicate_like_result" in report["per_result"][0]["warnings"]
    assert "missing_content" in report["warnings"]


def test_result_retrieval_diagnostics_flags_weak_coverage_and_missing_source():
    report = diagnose_result_retrieval("quantum battery cost", [{"id": "x", "content": "short note"}])

    assert report["per_result"][0]["warnings"] == ["weak_query_coverage", "missing_source_metadata", "shallow_evidence"]
