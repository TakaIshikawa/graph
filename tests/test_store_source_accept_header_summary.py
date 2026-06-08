from graph.store import summarize_source_accept_headers


def test_accept_header_summary_supports_direct_metadata_and_header_containers():
    summary = summarize_source_accept_headers(
        [
            {"id": "d", "Accept": "text/html, application/json;q=0.9, text/html"},
            {"id": "a", "metadata": {"accept": "application/xml"}},
            {"id": "b", "headers": {"ACCEPT": "image/png"}},
            {"id": "c", "metadata": {"response_headers": {"accept": "text/plain"}}},
            {"id": "e", "response_headers": {"Accept": ""}},
            {"id": "f"},
        ]
    )

    assert summary["total_sources"] == 6
    assert summary["sources_with_accept"] == 4
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["media_type_counts"] == {
        "application/json": 1,
        "application/xml": 1,
        "image/png": 1,
        "text/html": 1,
        "text/plain": 1,
    }
    assert summary["samples"][0]["source_id"] == "a"
    assert summary["samples"][0]["media_types"] == ["application/xml"]


def test_accept_header_summary_respects_sample_limit_while_counting():
    summary = summarize_source_accept_headers([{"id": "a", "Accept": "text/html, application/json"}], sample_limit=0)

    assert summary["sources_with_accept"] == 1
    assert summary["media_type_counts"] == {"application/json": 1, "text/html": 1}
    assert summary["samples"] == []
