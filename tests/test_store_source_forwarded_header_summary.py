from graph.store.source_forwarded_header_summary import summarize_source_forwarded_headers


def test_forwarded_header_summary_counts_params_proto_identifiers_and_malformed():
    summary = summarize_source_forwarded_headers(
        [
            {"source_id": "a", "Forwarded": 'for=192.168.0.1;proto=https;host="example.test", for="[2001:db8::1]";by=_proxy'},
            {"source_id": "b", "metadata": {"response_headers": {"FORWARDED": "by=10.0.0.1;proto=http, broken"}}},
            {"source_id": "c"},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_forwarded"] == 2
    assert summary["parameter_presence_counts"] == {"by": 2, "for": 2, "host": 1, "proto": 2}
    assert summary["proto_value_counts"] == {"http": 1, "https": 1}
    assert summary["obfuscated_identifier_count"] == 1
    assert summary["private_identifier_count"] == 3
    assert summary["malformed_count"] == 1
    assert summary["missing_forwarded_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "params": {"for": "192.168.0.1", "proto": "https", "host": "example.test"}, "value": 'for=192.168.0.1;proto=https;host="example.test"'},
        {"source_id": "a", "params": {"for": "[2001:db8::1]", "by": "_proxy"}, "value": 'for="[2001:db8::1]";by=_proxy'},
    ]
