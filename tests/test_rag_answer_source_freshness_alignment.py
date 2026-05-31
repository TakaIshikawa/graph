from graph.rag.answer_source_freshness_alignment import audit_answer_source_freshness_alignment


def test_detects_freshness_sensitive_wording():
    result = audit_answer_source_freshness_alignment("The latest guidance is current as of this month.", [{"published_at": "2026-02-01"}])

    assert [claim["claim"].casefold() for claim in result["freshness_claims"]] == ["latest", "current", "as of"]
    assert result["alignment_score"] == 0.85


def test_parses_source_date_metadata_fields():
    result = audit_answer_source_freshness_alignment("Recent reports agree.", [{"year": "2023"}, {"updated_at": "2024-05-02"}, {"date": "2022-01-01"}])

    assert result["oldest_source_date"] == "2022-01-01"
    assert result["newest_source_date"] == "2024-05-02"


def test_flags_current_claims_with_only_stale_sources():
    result = audit_answer_source_freshness_alignment("This is the current policy.", [{"published_at": "2021-06-01"}])

    assert result["stale_claim_warnings"] == ["current_claim_with_stale_sources"]
    assert result["alignment_score"] < 0.5
