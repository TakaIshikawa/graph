from graph.rag.evidence_access_method_signal import analyze_evidence_access_methods


def test_metadata_method_fields_take_precedence_over_text_cues():
    result = analyze_evidence_access_methods(
        [
            {"id": "a", "metadata": {"method": "API"}, "snippet": "Downloaded as a CSV export."},
            {"id": "b", "metadata": {"source_type": "PDF"}, "text": "Retrieved from the web."},
        ]
    )

    assert result["record_count"] == 2
    assert result["method_counts"]["api"] == 1
    assert result["method_counts"]["pdf"] == 1
    assert result["method_counts"]["csv"] == 0
    assert result["method_counts"]["web"] == 0
    assert result["method_diversity"] == 2
    assert result["unknown_method_count"] == 0
    assert result["samples"] == [{"result_id": "a", "method": "api"}, {"result_id": "b", "method": "pdf"}]


def test_text_cue_fallback_detects_access_methods():
    result = analyze_evidence_access_methods(
        [
            {"id": "a", "snippet": "Collected from an RSS feed."},
            {"id": "b", "content": "The HTML page was reviewed."},
            {"id": "c", "text": "Rows were checked in the customer database."},
        ]
    )

    assert result["method_counts"]["rss"] == 1
    assert result["method_counts"]["html"] == 1
    assert result["method_counts"]["database"] == 1
    assert result["method_diversity"] == 3
    assert result["unknown_method_count"] == 0
    assert result["samples"][0] == {"result_id": "a", "method": "rss"}


def test_unknown_methods_are_counted_and_sample_limit_is_honored():
    result = analyze_evidence_access_methods(
        [
            {"id": "a", "title": "No access hints"},
            {"id": "b", "metadata": {"source_type": "interview"}},
            {"id": "c", "metadata": {"access_method": "manual review"}},
        ],
        sample_limit=2,
    )

    assert result["record_count"] == 3
    assert result["method_counts"]["manual"] == 1
    assert result["method_diversity"] == 1
    assert result["unknown_method_count"] == 2
    assert result["samples"] == [{"result_id": "a", "method": "unknown"}, {"result_id": "b", "method": "unknown"}]
