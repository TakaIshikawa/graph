from graph.store import summarize_source_accept_languages


def test_accept_language_summary_parses_tags_q_values_and_export():
    summary = summarize_source_accept_languages(
        [
            {"id": "a", "request_headers": {"Accept-Language": "en-US,en;q=0.9, ja"}},
            {"id": "b", "metadata": {"request_headers": {"accept_language": "*, fr-FR;q=bad"}}},
            {"id": "c"},
        ]
    )

    assert summary["language_counts"] == {"en": 2, "ja": 1}
    assert summary["region_counts"] == {"us": 1}
    assert summary["wildcard_count"] == 1
    assert summary["malformed_count"] == 1
    assert summary["missing_accept_language_count"] == 1
