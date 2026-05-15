from __future__ import annotations

from graph.rag.citation_anchor_audit import audit_citation_anchors


def test_audit_citation_anchors_counts_missing_and_weak_anchors():
    audit = audit_citation_anchors(
        [
            {"id": "missing"},
            {"id": "title", "title": "Only title"},
            {"id": "url", "url": "https://example.test/page"},
            {"id": "strong", "title": "Title", "url": "https://strong.test/page"},
        ]
    )

    assert audit["total_results"] == 4
    assert audit["anchored_count"] == 3
    assert audit["weak_anchor_count"] == 3
    assert [(issue["result_id"], issue["type"]) for issue in audit["issues"]] == [
        ("missing", "missing"),
        ("title", "title-only"),
        ("url", "url-only"),
    ]


def test_audit_citation_anchors_reads_metadata_and_nested_unit_values():
    audit = audit_citation_anchors(
        [
            {"metadata": {"unit_id": "meta", "citation": "Example citation", "url": "example.test/a"}},
            {"unit": {"id": "unit", "metadata": {"source_url": "https://unit.test/a"}, "title": "Unit title"}},
        ]
    )

    assert audit["anchored_count"] == 2
    assert audit["weak_anchor_count"] == 0


def test_audit_citation_anchors_normalizes_duplicate_urls():
    audit = audit_citation_anchors(
        [
            {"id": "a", "title": "A", "url": "HTTPS://Example.TEST/path/"},
            {"id": "b", "title": "B", "source_url": "https://example.test/path"},
        ]
    )

    assert audit["duplicate_anchors"] == {"example.test/path": ["a", "b"]}
    assert [(issue["result_id"], issue["type"]) for issue in audit["issues"]] == [
        ("a", "duplicate"),
        ("b", "duplicate"),
    ]
