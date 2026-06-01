from graph.rag.result_format_diversity import analyze_result_format_diversity


def test_classifies_formats_from_metadata_url_and_title_hints():
    summary = analyze_result_format_diversity(
        [
            {"id": "pdf", "metadata": {"content_type": "application/pdf"}},
            {"id": "vid", "url": "https://youtube.com/watch?v=1"},
            {"id": "data", "title": "CSV dataset download"},
            {"id": "docs", "metadata": {"source_type": "documentation"}},
        ]
    )

    assert summary["format_counts"] == {"PDF": 1, "dataset": 1, "docs": 1, "video": 1}
    assert summary["dominant_format"] == "PDF"
    assert summary["diversity_score"] == 0.375
    assert summary["samples"] == [
        {"result_id": "pdf", "format": "PDF"},
        {"result_id": "data", "format": "dataset"},
        {"result_id": "docs", "format": "docs"},
        {"result_id": "vid", "format": "video"},
    ]
