from graph.rag.answer_citation_recency_disclosure import audit_answer_citation_recency_disclosure


def test_flags_stale_citations_without_age_disclosure():
    rows = audit_answer_citation_recency_disclosure(
        "This remains the recommended approach.",
        [{"id": "old", "publication_date": "2020-01-01"}, {"id": "new", "metadata": {"date": "2025-01-01"}}],
        current_date="2026-06-01",
    )

    assert rows == [{"citation_id": "old", "citation_date": "2020-01-01", "age_days": 2343, "severity": "medium"}]


def test_suppresses_when_answer_discloses_source_age():
    assert audit_answer_citation_recency_disclosure("As of 2020, this was recommended.", [{"date": "2020-01-01"}], current_date="2026-06-01") == []
