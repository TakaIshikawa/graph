from graph.store import summarize_source_nel_policies


def test_nel_policy_summary_counts_fields_and_samples():
    summary = summarize_source_nel_policies(
        [
            {"source_id": "b", "headers": {"NEL": '{"report_to":"default","include_subdomains":true,"failure_fraction":1.0}'}},
            {"source_id": "a", "metadata": {"response_headers": {"nel": '{"report_to":"cdn","success_fraction":0.1}'}}},
            {"source_id": "c", "nel": "not json"},
            {"source_id": "d"},
        ],
        sample_limit=2,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_nel"] == 3
    assert summary["report_to_counts"] == {"cdn": 1, "default": 1}
    assert summary["include_subdomains_count"] == 1
    assert summary["success_fraction_count"] == 1
    assert summary["failure_fraction_count"] == 1
    assert summary["malformed_count"] == 1
    assert summary["missing_count"] == 1
    assert summary["samples"][0]["source_id"] == "a"
    assert summary["samples"][0]["report_to"] == "cdn"


def test_nel_policy_summary_counts_non_object_json_as_malformed():
    summary = summarize_source_nel_policies([{"source_id": "a", "nel": "[]"}])

    assert summary["malformed_count"] == 1
    assert summary["report_to_counts"] == {}
