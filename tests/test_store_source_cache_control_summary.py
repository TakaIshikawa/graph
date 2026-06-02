from graph.store import summarize_source_cache_control_headers


def test_cache_control_summary_parses_headers_and_limits_unusual_samples():
    summary = summarize_source_cache_control_headers(
        [
            {"source_id": "a", "Cache-Control": "No-Cache, max-age=60; Immutable; stale-if-error=30"},
            {"source_id": "b", "metadata": {"response_headers": {"CACHE_CONTROL": "private, no-store; custom-one"}}},
            {"source_id": "c", "metadata": {"cache_control": "public; custom-two"}},
            {"source_id": "d"},
        ],
        sample_limit=1,
    )

    assert summary["total_sources"] == 4
    assert summary["sources_with_cache_control"] == 3
    assert summary["missing_cache_control_count"] == 1
    assert summary["directive_counts"] == {"custom-one": 1, "custom-two": 1, "immutable": 1, "max-age": 1, "no-cache": 1, "no-store": 1, "private": 1, "public": 1, "stale-if-error": 1}
    assert summary["noteworthy_samples"] == [{"source_id": "a", "directive": "stale-if-error=30"}]
