from graph.store import summarize_source_nel_headers


def test_nel_summary_parses_header_maps_and_samples_deterministically():
    summary = summarize_source_nel_headers(
        [
            {"source_id": "b", "headers": {"NEL": '{"report_to":"default","include_subdomains":true,"failure_fraction":1.0}'}},
            {"source_id": "a", "metadata": {"response_headers": {"nel": '{"report_to":"cdn","success_fraction":0.1}'}}},
            {"source_id": "c", "nel": "not json"},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_nel"] == 3
    assert summary["report_to_counts"] == {"cdn": 1, "default": 1}
    assert summary["include_subdomains_count"] == 1
    assert summary["fraction_configured_count"] == 2
    assert summary["malformed_count"] == 1
    assert summary["missing_nel_count"] == 1
    assert summary["samples"][0]["source_id"] == "a"
