from graph.store.source_proxy_status_summary import summarize_source_proxy_statuses


def test_proxy_status_summary_counts_proxies_errors_protocols_malformed_and_samples():
    summary = summarize_source_proxy_statuses(
        [
            {"source_id": "a", "Proxy-Status": 'edge; error="dns_timeout"; next-hop-protocol=h2, gateway; error=connection_refused'},
            {"source_id": "b", "metadata": {"headers": {"PROXY_STATUS": "; error=bad"}}},
            {"source_id": "c"},
        ],
        sample_limit=2,
    )

    assert summary["sources_with_proxy_status"] == 2
    assert summary["proxy_counts"] == {"edge": 1, "gateway": 1}
    assert summary["error_counts"] == {"connection_refused": 1, "dns_timeout": 1}
    assert summary["next_hop_protocol_counts"] == {"h2": 1}
    assert summary["malformed_entry_count"] == 1
    assert summary["missing_proxy_status_count"] == 1
    assert summary["samples"] == [
        {"source_id": "a", "proxy": "edge", "value": 'edge; error="dns_timeout"; next-hop-protocol=h2'},
        {"source_id": "a", "proxy": "gateway", "value": "gateway; error=connection_refused"},
    ]
