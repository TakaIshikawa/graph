from graph.store import summarize_source_accept_ch_headers


def test_accept_ch_summary_supports_direct_metadata_and_header_containers():
    summary = summarize_source_accept_ch_headers(
        [
            {"id": "d", "Accept-CH": "DPR, Width, DPR"},
            {"id": "a", "metadata": {"accept_ch": "Viewport-Width"}},
            {"id": "b", "headers": {"ACCEPT_CH": "Sec-CH-UA"}},
            {"id": "c", "metadata": {"response_headers": {"accept-ch": "Device-Memory"}}},
            {"id": "e", "response_headers": {"Accept-CH": ""}},
            {"id": "f"},
        ]
    )

    assert summary["total_sources"] == 6
    assert summary["sources_with_accept_ch"] == 4
    assert summary["missing_header_count"] == 1
    assert summary["empty_value_count"] == 1
    assert summary["hint_counts"] == {
        "device-memory": 1,
        "dpr": 1,
        "sec-ch-ua": 1,
        "viewport-width": 1,
        "width": 1,
    }
    assert summary["samples"][0]["source_id"] == "a"
    assert summary["samples"][0]["hints"] == ["viewport-width"]


def test_accept_ch_summary_respects_sample_limit_while_counting():
    summary = summarize_source_accept_ch_headers([{"id": "a", "Accept-CH": "DPR, Width"}], sample_limit=0)

    assert summary["sources_with_accept_ch"] == 1
    assert summary["hint_counts"] == {"dpr": 1, "width": 1}
    assert summary["samples"] == []
