from graph.store.source_alt_svc_summary import summarize_source_alt_svcs


def test_alt_svc_summary_counts_protocols_clear_and_max_age():
    summary = summarize_source_alt_svcs(
        [
            {"source_id": "a", "Alt-Svc": 'h3=":443"; ma=86400, h2=":443"'},
            {"source_id": "b", "metadata": {"response_headers": {"alt-svc": "clear"}}},
            {"source_id": "c", "headers": {"ALT_SVC": 'h3-29=":443"; ma="60"'}},
            {"source_id": "d"},
        ]
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_alt_svc"] == 3
    assert summary["missing_alt_svc_count"] == 1
    assert summary["protocol_counts"] == {"clear": 1, "h2": 1, "h3": 1, "h3-29": 1}
    assert summary["max_age_counts"] == {"60": 1, "86400": 1}
    assert summary["max_age_samples"] == [
        {"source_id": "a", "protocol": "h3", "ma": "86400", "value": 'h3=":443"; ma=86400'},
        {"source_id": "c", "protocol": "h3-29", "ma": "60", "value": 'h3-29=":443"; ma="60"'},
    ]


def test_alt_svc_summary_ignores_blank_and_malformed_segments_deterministically():
    summary = summarize_source_alt_svcs(
        [
            {"source_id": "z", "Alt-Svc": 'malformed, h3=":443"; ma=bad'},
            {"source_id": "a", "Alt-Svc": 'h2=":443"; ma=100, clear'},
            {"source_id": "blank", "Alt-Svc": ""},
        ],
        sample_limit=3,
    )

    assert summary["sources_with_alt_svc"] == 2
    assert summary["protocol_counts"] == {"clear": 1, "h2": 1, "h3": 1}
    assert summary["max_age_counts"] == {"100": 1, "bad": 1}
    assert summary["samples"] == [
        {"source_id": "a", "protocol": "h2", "value": 'h2=":443"; ma=100', "ma": "100"},
        {"source_id": "a", "protocol": "clear", "value": "clear"},
        {"source_id": "z", "protocol": "h3", "value": 'h3=":443"; ma=bad', "ma": "bad"},
    ]
