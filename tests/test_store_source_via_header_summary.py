from graph.store import summarize_source_via_headers


def test_via_summary_counts_hops_protocols_hosts_and_malformed_values():
    summary = summarize_source_via_headers(
        [
            {"source_id": "a", "Via": "1.1 proxy-a"},
            {"source_id": "b", "metadata": {"headers": {"VIA": "2.0 edge, HTTP/1.1 cache"}}},
            {"source_id": "c", "response_headers": {"via": "broken"}},
            {"source_id": "d"},
        ],
        sample_limit=1,
    )

    assert summary["sources_with_via"] == 3
    assert summary["sources_missing_via"] == 1
    assert summary["proxy_hop_count_distribution"] == {"1": 2, "2": 1}
    assert summary["protocol_counts"] == {"1.1": 1, "2.0": 1, "http/1.1": 1}
    assert summary["top_via_hosts"] == {"cache": 1, "edge": 1, "proxy-a": 1}
    assert summary["malformed_via_samples"] == [{"source_id": "c", "value": "broken"}]
