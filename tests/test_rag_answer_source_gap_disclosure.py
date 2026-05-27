from __future__ import annotations

from graph.rag.answer_source_gap_disclosure import audit_answer_source_gap_disclosure


def test_audit_answer_source_gap_disclosure_detects_gap_phrases():
    result = audit_answer_source_gap_disclosure(
        "This answer excludes paywalled sources. Public records were unavailable, so the conclusion is partial."
    )

    assert result["disclosure_count"] == 2
    assert result["disclosures"][0]["gap_terms"] == ["excludes"]
    assert result["coverage_score"] == 1.0


def test_audit_answer_source_gap_disclosure_matches_missing_sources_case_insensitively():
    result = audit_answer_source_gap_disclosure(
        "No CRM sources were available. We also omitted Support Tickets from retrieval.",
        missing_sources=["crm", "support tickets"],
    )

    assert result["disclosed_missing_sources"] == ["crm", "support tickets"]
    assert result["undisclosed_missing_sources"] == []
    assert result["findings"] == []
    assert result["coverage_score"] == 1.0


def test_audit_answer_source_gap_disclosure_returns_actionable_undisclosed_findings():
    result = audit_answer_source_gap_disclosure(
        "The summary uses internal notes only; no billing sources were available.",
        missing_sources=["billing", "legal memos"],
    )

    assert result["disclosed_missing_sources"] == ["billing"]
    assert result["undisclosed_missing_sources"] == ["legal memos"]
    assert result["findings"] == [
        {
            "source": "legal memos",
            "issue": "missing_source_not_disclosed",
            "message": "Disclose that legal memos sources are missing, unavailable, or excluded.",
        }
    ]
    assert result["coverage_score"] == 0.5
