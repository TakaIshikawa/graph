from __future__ import annotations

from graph.rag.answer_source_attribution_integrity import audit_answer_source_attribution_integrity


def test_source_attribution_integrity_flags_unknown_citation_ids():
    result = audit_answer_source_attribution_integrity("Claim [missing].", [{"id": "e1", "title": "Doc"}])

    assert result["issues"] == [{"type": "unknown_citation_id", "label": "missing", "severity": "high"}]


def test_source_attribution_integrity_flags_title_url_mismatch_and_duplicate_labels():
    result = audit_answer_source_attribution_integrity(
        "See Policy at https://wrong.example/policy [p1].",
        [
            {"id": "p1", "title": "Policy", "url": "https://example.com/policy"},
            {"id": "p2", "title": "Policy", "url": "https://example.com/policy-v2"},
        ],
    )

    assert {issue["type"] for issue in result["issues"]} == {"duplicate_conflicting_label", "title_url_mismatch"}


def test_source_attribution_integrity_accepts_clean_citations():
    result = audit_answer_source_attribution_integrity("See Policy at https://example.com/policy [p1].", [{"id": "p1", "title": "Policy", "url": "https://example.com/policy"}])

    assert result["issues"] == []
