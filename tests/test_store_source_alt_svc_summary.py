from graph.store import summarize_source_alt_svc_headers


def test_alt_svc_summary_extracts_protocols_and_clear_directives():
    summary = summarize_source_alt_svc_headers(
        [
            {"source_id": "a", "Alt-Svc": 'h3=":443"; ma=86400, h2=":443"'},
            {"source_id": "b", "metadata": {"response_headers": {"alt-svc": "clear"}}},
            {"source_id": "c", "headers": {"Alt-Svc": 'h3-29=":443"'}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_alt_svc"] == 3
    assert summary["protocol_counts"] == {"h2": 1, "h3": 1, "h3-29": 1}
    assert summary["clear_count"] == 1
    assert summary["missing_alt_svc_count"] == 1
    assert summary["samples"][0] == {"source_id": "a", "protocol": "h3", "value": 'h3=":443"; ma=86400, h2=":443"'}
