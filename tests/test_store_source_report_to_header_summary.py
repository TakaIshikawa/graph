from __future__ import annotations

from graph.store.source_report_to_header_summary import summarize_source_report_to_headers


def test_report_to_summary_parses_json_object_array_and_malformed_values():
    summary = summarize_source_report_to_headers(
        [
            {"id": "a", "headers": {"Report-To": '{"group":"default","endpoints":[{"url":"https://r.example.test/a"}]}'}},
            {"id": "b", "metadata": {"report_to": '[{"group":"csp","endpoints":[{"url":"https://csp.example.test/r"}]}]'}},
            {"id": "c", "response_headers": {"REPORT_TO": "not-json"}},
            {"id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_report_to"] == 3
    assert summary["group_counts"] == {"csp": 1, "default": 1}
    assert summary["endpoint_host_counts"] == {"csp.example.test": 1, "r.example.test": 1}
    assert summary["malformed_count"] == 1
    assert summary["missing_report_to_count"] == 1


def test_report_to_samples_are_sorted_and_limited():
    summary = summarize_source_report_to_headers(
        [{"id": "z", "report-to": "{}"}, {"id": "a", "report_to": "broken"}],
        sample_limit=1,
    )

    assert summary["samples"] == [{"source_id": "a", "value": "broken", "malformed": True}]
