from graph.store import summarize_source_dns_prefetch_controls


def test_dns_prefetch_control_summary_counts_on_off_unknown_and_missing():
    summary = summarize_source_dns_prefetch_controls(
        [
            {"source_id": "a", "X-DNS-Prefetch-Control": "on"},
            {"source_id": "b", "metadata": {"headers": {"x-dns-prefetch-control": "off"}}},
            {"source_id": "c", "response_headers": {"X-DNS-Prefetch-Control": "sometimes"}},
            {"source_id": "d"},
        ]
    )

    assert summary["sources_with_header"] == 3
    assert summary["value_counts"] == {"off": 1, "on": 1, "sometimes": 1}
    assert summary["missing_header_count"] == 1
    assert summary["enabled_count"] == 1
    assert summary["disabled_count"] == 1
    assert summary["unknown_value_count"] == 1
    assert [sample["source_id"] for sample in summary["samples"]] == ["a", "c"]
