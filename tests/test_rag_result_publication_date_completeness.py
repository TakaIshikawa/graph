from __future__ import annotations

from graph.rag.result_publication_date_completeness import analyze_result_publication_date_completeness


def test_result_publication_date_completeness_counts_metadata_and_text_fallbacks():
    summary = analyze_result_publication_date_completeness(
        [
            {"id": "published", "publication_date": "2026-01-01"},
            {"id": "updated", "metadata": {"updated_at": "2026-02-01"}},
            {"id": "text", "snippet": "Published on March 3, 2026 by the agency."},
            {"id": "missing"},
        ]
    )

    assert summary == {"total_results": 4, "results_with_dates": 3, "results_missing_dates": 1, "missing_date_result_ids": ["missing"]}


def test_result_publication_date_completeness_handles_empty_results():
    assert analyze_result_publication_date_completeness([]) == {"total_results": 0, "results_with_dates": 0, "results_missing_dates": 0, "missing_date_result_ids": []}
