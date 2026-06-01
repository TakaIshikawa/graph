from graph.rag.evidence_source_independence import analyze_evidence_source_independence


def test_repeated_domains_form_duplicate_groups():
    summary = analyze_evidence_source_independence(
        [{"id": "a", "url": "https://www.example.com/a"}, {"id": "b", "url": "https://example.com/b"}, {"id": "c", "domain": "other.org"}]
    )

    assert summary["record_count"] == 3
    assert summary["independent_source_count"] == 2
    assert summary["duplicate_source_group_count"] == 1
    assert summary["largest_group_size"] == 2
    assert summary["concentration_ratio"] == 0.667


def test_distinct_sources_have_no_duplicate_groups():
    assert analyze_evidence_source_independence([{"source": "a"}, {"source": "b"}])["duplicate_source_group_count"] == 0


def test_empty_evidence_returns_zero_counts():
    assert analyze_evidence_source_independence([])["record_count"] == 0
