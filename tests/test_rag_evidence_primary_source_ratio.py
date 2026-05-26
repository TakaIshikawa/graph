from __future__ import annotations

from graph.rag import analyze_evidence_primary_source_ratio


def test_primary_source_ratio_prefers_explicit_metadata_over_domain_heuristics():
    result = analyze_evidence_primary_source_ratio([{"source_type": "secondary", "url": "https://agency.gov/report"}, {"url": "https://nih.gov/study"}])

    assert result["counts"] == {"primary": 1, "secondary": 1, "tertiary": 0, "unknown": 0}
    assert result["primary_ratio"] == 0.5


def test_primary_source_ratio_counts_unknown_separately():
    result = analyze_evidence_primary_source_ratio([{"url": "https://example.com/post"}])

    assert result["counts"]["unknown"] == 1
    assert result["flagged_gaps"] == ["no_primary_sources", "unknown_source_level"]


def test_primary_source_ratio_handles_empty_input():
    assert analyze_evidence_primary_source_ratio([]) == {
        "counts": {"primary": 0, "secondary": 0, "tertiary": 0, "unknown": 0},
        "primary_ratio": 0.0,
        "flagged_gaps": ["no_evidence"],
    }
