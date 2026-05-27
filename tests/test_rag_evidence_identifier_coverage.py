from types import SimpleNamespace

from graph.rag.evidence_identifier_coverage import audit_evidence_identifier_coverage


def test_identifier_coverage_counts_multiple_identifier_types_without_double_counting_items():
    report = audit_evidence_identifier_coverage(
        [
            {"id": "a", "url": "https://example.com/a"},
            {"source_id": "s2", "doi": "10.1/x"},
            SimpleNamespace(citation_key="smith2025"),
            {"title": "Missing", "content": "No stable identifier here"},
        ]
    )

    assert report == {
        "evidence_count": 4,
        "identified_count": 3,
        "unidentified_count": 1,
        "coverage_ratio": 0.75,
        "counts_by_identifier_type": {"id": 1, "source_id": 1, "url": 1, "doi": 1, "citation_key": 1},
        "samples": [{"index": 3, "snippet": "Missing No stable identifier here"}],
    }
