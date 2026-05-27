from graph.store import summarize_collection_empty_titles


def test_collection_empty_titles_counts_missing_blank_and_present_aliases():
    report = summarize_collection_empty_titles(
        [
            {"id": "missing"},
            {"id": "blank", "title": "   "},
            {"id": "name", "name": "Named"},
            {"id": "label", "metadata": {"label": "Labeled"}},
        ]
    )

    assert report["missing_title_count"] == 1
    assert report["blank_title_count"] == 1
    assert report["present_title_count"] == 2
    assert report["completeness_ratio"] == 0.5
    assert report["sample_collection_ids"] == ["blank", "missing"]


def test_collection_empty_titles_zero_safe_ratio():
    assert summarize_collection_empty_titles([])["completeness_ratio"] == 0
