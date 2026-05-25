from __future__ import annotations

from graph.store.unit_citation_doi_coverage_summary import summarize_unit_citation_doi_coverage


def test_summarize_unit_citation_doi_coverage_counts_complete_partial_and_empty_sets():
    summary = summarize_unit_citation_doi_coverage(
        [
            {"id": "complete", "metadata": {"citations": [{"doi": "10.1000/a"}, {"DOI": "10.1000/b"}]}},
            {"id": "partial", "metadata": {"citations": [{"title": "Missing"}, {"Doi": "10.1000/c"}]}},
            {"id": "empty", "metadata": {"citations": []}},
        ]
    )

    assert summary == {
        "total_units": 3,
        "units_with_citations": 2,
        "total_citations": 4,
        "doi_citations": 3,
        "missing_doi_citations": 1,
        "coverage_ratio": 0.75,
        "units_missing_dois": ["partial"],
    }


def test_summarize_unit_citation_doi_coverage_handles_no_citations_safely():
    summary = summarize_unit_citation_doi_coverage([{"id": "empty", "metadata": {}}])

    assert summary["coverage_ratio"] == 0.0
    assert summary["units_missing_dois"] == []
