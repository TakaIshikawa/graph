from graph.store import summarize_source_cache_status_headers


def test_cache_status_summary_counts_buckets_ttl_detail_malformed_and_missing():
    summary = summarize_source_cache_status_headers(
        [
            {"source_id": "a", "Cache-Status": 'cdn; hit; ttl=120; collapsed; detail="fresh"'},
            {"source_id": "b", "metadata": {"cache_status": "edge; fwd=uri-miss; ttl=0"}},
            {"source_id": "c", "headers": {"Cache-Status": 'browser; fwd=stale; ttl=7200; detail="revalidated"'}},
            {"source_id": "d", "response_headers": {"cache-status": "; ttl=12"}},
            {"source_id": "e"},
        ]
    )

    assert summary["cache_counts"] == {"browser": 1, "cdn": 1, "edge": 1}
    assert summary["bucket_counts"] == {"hit": 1, "miss": 1, "pass": 0, "stale": 1}
    assert summary["ttl_bucket_counts"] == {"expired": 1, "short": 1, "medium": 1, "long": 1}
    assert summary["collapsed_forwarding_count"] == 1
    assert summary["detail_counts"] == {"fresh": 1, "revalidated": 1}
    assert summary["malformed_entry_count"] == 1
    assert summary["missing_cache_status_count"] == 1
