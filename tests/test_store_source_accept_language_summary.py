from graph.store import summarize_source_accept_languages


def test_accept_language_summary_parses_direct_and_nested_headers():
    summary = summarize_source_accept_languages(
        [
            {"id": "a", "accept_language": "en-US,en;q=0.9, ja"},
            {"id": "b", "metadata": {"response_headers": {"accept_language": "fr-FR;q=0.8, *;q=0.1"}}},
            {"id": "c", "headers": {"accept-language": "de-DE;q=1.0"}},
        ],
    )

    assert summary["total_sources"] == 3
    assert summary["sources_with_accept_language"] == 3
    assert summary["missing_accept_language_count"] == 0
    assert summary["language_counts"] == {"de": 1, "en": 2, "fr": 1, "ja": 1}
    assert summary["max_language_count"] == 3
    assert summary["rows"] == [
        {
            "source_id": "a",
            "accept_language": "en-US,en;q=0.9, ja",
            "language_ranges": ["en-us", "en", "ja"],
            "primary_languages": ["en", "en", "ja"],
            "language_count": 3,
        },
        {
            "source_id": "b",
            "accept_language": "fr-FR;q=0.8, *;q=0.1",
            "language_ranges": ["fr-fr", "*"],
            "primary_languages": ["fr"],
            "language_count": 2,
        },
        {
            "source_id": "c",
            "accept_language": "de-DE;q=1.0",
            "language_ranges": ["de-de"],
            "primary_languages": ["de"],
            "language_count": 1,
        },
    ]


def test_accept_language_summary_skips_invalid_q_weights_and_counts_missing():
    summary = summarize_source_accept_languages(
        [
            {"id": "a", "headers": {"accept-language": "de-DE;q=1.0, bad;q=2"}},
            {"id": "b"},
        ]
    )

    assert summary["sources_with_accept_language"] == 1
    assert summary["missing_accept_language_count"] == 1
    assert summary["rows"][0]["language_ranges"] == ["de-de"]
    assert summary["rows"][0]["primary_languages"] == ["de"]


def test_accept_language_summary_limits_samples():
    summary = summarize_source_accept_languages(
        [
            {"id": "a", "request_headers": {"Accept-Language": "en"}},
            {"id": "b", "request_headers": {"Accept-Language": "fr"}},
            {"id": "c", "request_headers": {"Accept-Language": "ja"}},
        ],
        sample_limit=2,
    )

    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "b"]
