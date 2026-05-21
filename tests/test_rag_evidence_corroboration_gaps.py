from __future__ import annotations

from graph.rag.evidence_corroboration_gaps import analyze_evidence_corroboration_gaps


def test_evidence_corroboration_gaps_counts_independent_sources():
    report = analyze_evidence_corroboration_gaps(
        ["Alpha launch happened in Paris", "Beta remains unsupported"],
        [
            {"id": "a", "content": "Alpha launch happened in Paris", "source": "docs"},
            {"id": "b", "content": "Alpha launch happened in Paris", "url": "https://news.test/a"},
        ],
    )

    assert report["claims"][0]["support_count"] == 2
    assert report["claims"][0]["independent_source_count"] == 2
    assert report["claims"][0]["supporting_result_ids"] == ["a", "b"]
    assert report["claims"][1]["warnings"] == ["unsupported_claim"]


def test_evidence_corroboration_gaps_distinguishes_single_source_support():
    report = analyze_evidence_corroboration_gaps(
        ["Gamma release was delayed"],
        [{"id": "a", "content": "Gamma release was delayed", "source_project": "docs"}],
    )

    assert report["claims"][0]["warnings"] == ["single_source_support"]
    assert report["warnings"] == ["single_source_claims_high"]
