from graph.rag.evidence_provenance_completeness import analyze_evidence_provenance_completeness


def test_complete_records_have_full_completeness():
    result = analyze_evidence_provenance_completeness(
        [
            {
                "id": "a",
                "source": "Docs",
                "url": "https://example.test",
                "title": "Guide",
                "author": "Team",
                "published_at": "2025-01-01",
                "retrieved_at": "2025-01-02",
                "source_type": "documentation",
            }
        ]
    )

    assert result["record_count"] == 1
    assert result["complete_record_count"] == 1
    assert result["average_completeness"] == 1.0
    assert result["samples"] == []


def test_partial_records_report_missing_counts_and_samples():
    result = analyze_evidence_provenance_completeness(
        [{"id": "a", "source": "Docs", "title": "Guide"}, {"id": "b", "url": "https://example.test"}],
        sample_limit=1,
    )

    assert result["complete_record_count"] == 0
    assert result["average_completeness"] == 0.214
    assert result["missing_field_counts"]["source"] == 1
    assert result["missing_field_counts"]["url"] == 1
    assert result["missing_field_counts"]["author"] == 2
    assert result["samples"] == [
        {"result_id": "a", "missing_fields": ["url", "author", "published_at", "retrieved_at", "source_type"]}
    ]


def test_custom_required_fields_are_honored():
    result = analyze_evidence_provenance_completeness(
        [{"id": "a", "metadata": {"publisher": "Lab"}}],
        required_fields=["publisher", "dataset"],
    )

    assert result["required_fields"] == ["publisher", "dataset"]
    assert result["missing_field_counts"] == {"publisher": 0, "dataset": 1}
    assert result["samples"] == [{"result_id": "a", "missing_fields": ["dataset"]}]
