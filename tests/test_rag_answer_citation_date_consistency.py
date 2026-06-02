from graph.rag.answer_citation_date_consistency import audit_answer_citation_date_consistency


def test_matching_years_do_not_create_issues():
    result = audit_answer_citation_date_consistency(
        "The policy changed in 2026 [A].",
        [{"id": "A", "published_at": "2026-03-10T00:00:00Z"}],
    )

    assert result == {"has_date_consistency_issues": False, "issues": []}


def test_mismatched_years_are_flagged():
    result = audit_answer_citation_date_consistency(
        "The benchmark was updated in 2026 [A].",
        [{"id": "A", "published_at": "2024-12-31"}],
    )

    assert result["has_date_consistency_issues"] is True
    assert result["issues"] == [
        {
            "claim_text": "The benchmark was updated in 2026 [A].",
            "citation_id": "A",
            "cited_date": "2024-12-31",
            "severity": "high",
            "issue_type": "date_mismatch",
        }
    ]


def test_missing_citation_dates_are_reported_separately():
    result = audit_answer_citation_date_consistency(
        "The standard was revised in 2025 [std].",
        [{"id": "std", "title": "Standard"}],
    )

    assert result["issues"][0]["issue_type"] == "missing_citation_date"
    assert result["issues"][0]["severity"] == "medium"
    assert result["issues"][0]["cited_date"] is None


def test_answers_without_date_claims_return_no_issues():
    result = audit_answer_citation_date_consistency("The source describes the current workflow [A].", [])

    assert result == {"has_date_consistency_issues": False, "issues": []}
